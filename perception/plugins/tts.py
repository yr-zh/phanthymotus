#!/usr/bin/env python3
"""
plugins/tts.py — Gentleman TTS (PhoneTone + Matcha + Vocos Python ORT).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
from abc import ABC, abstractmethod
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_BYTES = 3200  # 100ms @ 16kHz 16-bit mono
# 0 = do not length-split; only cut at sentence punctuation (。！？；).
# Forced N-char cuts were adding mid-clause pauses in ranking copy.
MAX_SEGMENT_CHARS = 0
# Local synthesis buffer. 600 frames is about 60 seconds / 1.9 MB of PCM.
# It lets the producer synthesize the next sentence while the current one plays.
SYNTH_QUEUE_FRAMES = 600


def _maybe_set_cpu_affinity() -> None:
    """Optional CPU pinning for Jetson benchmarks (TTS_CPU_AFFINITY=0,1,2,3)."""
    import os

    if not hasattr(os, "sched_setaffinity"):
        return
    spec = os.environ.get("TTS_CPU_AFFINITY", "").strip()
    if not spec:
        return
    cores = {int(x.strip()) for x in spec.split(",") if x.strip()}
    if cores:
        os.sched_setaffinity(0, cores)
        log.info(f"[tts] CPU affinity set to {sorted(cores)}")


def _process_rss_mb() -> float:
    """Current process RSS in MB (for Jetson memory benchmarking)."""
    import os

    import psutil

    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def _maybe_malloc_trim(where: str) -> None:
    """Return free glibc heap pages to the OS (GPT 'I'). Opt-in via TTS_MALLOC_TRIM=1."""
    import ctypes
    import os

    if os.environ.get("TTS_MALLOC_TRIM", "0") != "1":
        return
    before = _process_rss_mb()
    try:
        ret = ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception as e:
        log.warning("[tts] malloc_trim failed at %s: %s", where, e)
        return
    after = _process_rss_mb()
    log.info(
        "[tts] malloc_trim(%s) ret=%s rss_mb=%.1f->%.1f (delta=%.1f)",
        where,
        ret,
        before,
        after,
        after - before,
    )


def _piper_run(sess, ort_module, feeds):
    """ORT session.run. GPU arena shrinkage is opt-in via TTS_ORT_ARENA_SHRINK=1.

    Shrinkage MUST be a RunOptions entry (gpu:0), not SessionOptions.
    """
    import os
    import threading

    if os.environ.get("TTS_ORT_DUMP_THREAD", "0") == "1":
        print("TTS ORT THREAD", threading.get_ident(), flush=True)
    if os.environ.get("TTS_ORT_ARENA_SHRINK", "0") != "1":
        return sess.run(None, feeds)
    run_options = ort_module.RunOptions()
    run_options.add_run_config_entry(
        "memory.enable_memory_arena_shrinkage",
        "gpu:0",
    )
    if not getattr(_piper_run, "_logged_shrink", False):
        _piper_run._logged_shrink = True
        log.info("[tts] ORT RunOptions memory.enable_memory_arena_shrinkage=gpu:0")
        print("ARENA_SHRINK_APPLIED gpu:0", flush=True)
    return sess.run(None, feeds, run_options=run_options)



def _piper_ort_providers(hw_provider: str, gpu_mem_limit_mb: int | None = None) -> tuple:
    """Return (onnxruntime module, provider list) for Gentleman InferenceSession.

    CUDA EP only. No TensorRT EP.
    """
    import os

    import onnxruntime as ort

    hw = (hw_provider or "cpu").lower().strip()
    want_gpu = hw == "cuda"

    if want_gpu:
        available = ort.get_available_providers()
        max_ws = os.environ.get("TTS_ORT_CUDNN_MAX_WORKSPACE", "0").strip() or "0"
        algo = os.environ.get("TTS_ORT_CUDNN_ALGO", "HEURISTIC").strip() or "HEURISTIC"
        cuda_opts = {
            "device_id": 0,
            "cudnn_conv_use_max_workspace": max_ws,
            "cudnn_conv_algo_search": algo,
            "arena_extend_strategy": "kSameAsRequested",
        }
        arena = os.environ.get("TTS_ORT_ARENA_EXTEND", "kSameAsRequested").strip()
        if arena in ("kSameAsRequested", "kNextPowerOfTwo"):
            cuda_opts["arena_extend_strategy"] = arena
        if gpu_mem_limit_mb is not None and int(gpu_mem_limit_mb) > 0:
            mem_mb = str(int(gpu_mem_limit_mb))
        else:
            mem_mb = os.environ.get("TTS_ORT_GPU_MEM_LIMIT_MB", "512").strip()
        if mem_mb.isdigit() and int(mem_mb) > 0:
            cuda_opts["gpu_mem_limit"] = int(mem_mb) * 1024 * 1024

        if "CUDAExecutionProvider" in available:
            log.info("[tts] CUDA EP options: %s", cuda_opts)
            return ort, [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]

        require = os.environ.get("TTS_REQUIRE_CUDA", "1") == "1"
        msg = (
            "[tts] GPU requested but CUDAExecutionProvider missing; "
            f"available={available}"
        )
        if require:
            raise RuntimeError(
                msg
                + " (TTS_REQUIRE_CUDA=1). Rebuild with onnxruntime-gpu "
                "(see Dockerfile.jetson)."
            )
        log.warning(msg + "; using CPU")
    return ort, ["CPUExecutionProvider"]


def _ort_lowmem_session_options(ort, num_threads: int):
    """SessionOptions used by Gentleman Matcha/BigVGAN."""
    import os

    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = os.environ.get("TTS_ORT_CPU_ARENA", "0") == "1"
    so.enable_mem_pattern = os.environ.get("TTS_ORT_MEM_PATTERN", "1") != "0"
    so.intra_op_num_threads = max(1, int(num_threads))
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if os.environ.get("TTS_ORT_DISABLE_PREPACKING", "1") == "1":
        so.add_session_config_entry("session.disable_prepacking", "1")
    if os.environ.get("TTS_ORT_DEVICE_INITIALIZERS", "0") == "1":
        so.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
    level = os.environ.get("TTS_ORT_GRAPH_OPT", "all").strip().lower()
    mapping = {
        "off": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    if level in mapping:
        so.graph_optimization_level = mapping[level]
    return so


def _load_onnx_session(
    onnx_path: str,
    hw_provider: str,
    num_threads: int,
    gpu_mem_limit_mb: int | None = None,
):
    """CUDA session with a per-session arena cap. Still sess.run, no IOBinding.

    Uncapped CUDA EP on JP5 grabs several GB of unified memory on first Run
    (cuDNN workspace + BFC). Default 512MB Matcha / 128MB BigVGAN.
    """
    import os

    ort, providers = _piper_ort_providers(
        hw_provider, gpu_mem_limit_mb=gpu_mem_limit_mb
    )
    so = _ort_lowmem_session_options(ort, num_threads)
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    log.info(
        "[tts] onnx %s providers=%s",
        os.path.basename(onnx_path),
        sess.get_providers(),
    )
    return ort, sess


def _ort_outputs(sess, feeds: dict) -> dict:
    import onnxruntime as ort

    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, _piper_run(sess, ort, feeds)))


def _dump_stage(tag: str) -> None:
    return


def _phonetone_acoustic_path(model_dir: str) -> str:
    import os

    for name in ("model-steps-3.onnx", "model-steps-10.onnx"):
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            return path
    return os.path.join(model_dir, "model-steps-3.onnx")


def _phonetone_vocoder_path(model_dir: str) -> str:
    """Gentleman ranking uses Vocos only (no BigVGAN fallback)."""
    import os

    env = os.environ.get("TTS_VOCODER_ONNX", "").strip()
    if env:
        return env
    for name in ("gentleman-vocos.onnx", "vocos.onnx", "vocos-16khz-univ.onnx"):
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            return path
    return os.path.join(model_dir, "gentleman-vocos.onnx")


def _phonetone_e2e_path(model_dir: str) -> str:
    """Optional merged Matcha+BigVGAN ONNX. Never required for ranking.

    Priority: TTS_GENTLEMAN_E2E_ONNX > {model_dir}/gentleman-e2e.onnx > "".
    Empty means the caller must keep the two-session Matcha + BigVGAN path.
    """
    import os

    env = os.environ.get("TTS_GENTLEMAN_E2E_ONNX", "").strip()
    if env:
        return env
    path = os.path.join(model_dir, "gentleman-e2e.onnx")
    return path if os.path.isfile(path) else ""


class _WaveformOrt:
    """Mel→wav via Gentleman Vocos ONNX (mag/x/y) + CPU iSTFT."""

    def __init__(self, onnx_path: str, hw_provider: str, num_threads: int = 2):
        voc_mb = os.environ.get("TTS_VOCODER_GPU_MEM_LIMIT_MB", "128").strip()
        gpu_mb = int(voc_mb) if voc_mb.isdigit() and int(voc_mb) > 0 else 128
        _ort, self._sess = _load_onnx_session(
            onnx_path, hw_provider, num_threads, gpu_mem_limit_mb=gpu_mb
        )
        self._in = self._sess.get_inputs()[0].name
        outs = [o.name for o in self._sess.get_outputs()]
        if not {"mag", "x", "y"}.issubset(outs):
            raise RuntimeError(
                "Gentleman vocoder must be Vocos (outputs mag/x/y), got %s from %s"
                % (outs, onnx_path)
            )
        log.info(
            "[tts] vocoder onnx=%s kind=vocos outs=%s gpu_mem_limit_mb=%s",
            os.path.basename(onnx_path),
            outs,
            gpu_mb,
        )

    def infer(self, mel_bct):
        import numpy as np
        from utils.matcha_ort import vocos_istft

        mel = np.ascontiguousarray(mel_bct, dtype=np.float32)
        out = _ort_outputs(self._sess, {self._in: mel})
        wav = vocos_istft(out["mag"], out["x"], out["y"])
        cap = int(mel.shape[-1]) * 256
        return wav[:cap]

_maybe_set_cpu_affinity()

def _normalize_tts_text(text: str) -> str:
    """PhoneTone does TN itself. Keep a no-op for the generic adapter API."""
    return text or ""


_LOW_LAT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=200,
    durability=DurabilityPolicy.VOLATILE,
)

TOOLS = [
    {
        "name": "tts",
        "type": "processor",
        "multiInstance": True,
        "description": "TTS — start/stop speech synthesis, speak text, or get status",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "speak", "info", "config"],
                    "description": "Action to perform"
                },
                "input_topic": {
                    "type": "string",
                    "description": "ROS2 topic for text input (data/json, required for action=start)"
                },
                "text": {
                    "type": "string",
                    "description": "Text to synthesize (required for action=speak)"
                },
            },
            "required": ["action"]
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "speaker_id": {"type": "integer", "description": "Speaker ID", "default": 0, "scope": "shared"},
                "speed":      {"type": "number", "description": "Speech speed (1.0 = normal)", "default": 1.0, "scope": "shared"},
            },
            "required": []
        },
        "topic_in":  [{"format": "data/json",     "desc": "text to synthesize"}],
        "topic_out": [{"format": "audio/pcm-16k", "desc": "synthesized PCM audio"}],
    }
]


# ── TTS Adapter ──────────────────────────────────────────────────────────────

# Split priority (prosody first, memory second):
#   。！？； / .!?  → always
#   ，,            → only when the clause is already long
#   、              → last-resort backstop, not a normal cut
_STRONG_SENTENCE_END = frozenset("。！？；;!?")
_COMMA_CHARS = frozenset("，,")
_WEAK_SENTENCE_END = frozenset("、：:")
_CLOSING_PUNCTUATION = frozenset("”’\"'》〉】〕）)]}」』")
_PAUSE_MS = {
    "，": 120,
    ",": 120,
    "、": 80,
    # Timeline 分号 in ranking copy is closer to a comma than a period.
    "；": 100,
    ";": 100,
    "：": 150,
    ":": 150,
    "。": 280,
    "．": 280,
    ".": 220,
    "！": 280,
    "？": 280,
    "!": 280,
    "?": 280,
}


def _is_cjk(char: str) -> bool:
    """Return True for common CJK code-point ranges."""
    if not char:
        return False
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
    )


def _is_single_letter_abbrev_dot(text: str, dot_index: int) -> bool:
    """True for U.S. / e.g. style dots: a single letter immediately before '.'."""
    if dot_index <= 0:
        return False
    prev = text[dot_index - 1]
    if not (prev.isascii() and prev.isalpha()):
        return False
    if dot_index == 1:
        return True
    return not text[dot_index - 2].isalpha()


def _find_cut(remaining: str, max_chars: int, marks: frozenset) -> int:
    """Last mark in [max/2, max], else the next mark a bit past max. Else -1."""
    if max_chars <= 0:
        return -1
    min_cut = max(1, max_chars // 2)
    hard = min(len(remaining), max(max_chars, int(max_chars * 1.5)))
    for index in range(min(max_chars, len(remaining)) - 1, min_cut - 1, -1):
        if remaining[index] in marks:
            return index + 1
    for index in range(max_chars, hard):
        if remaining[index] in marks:
            return index + 1
    return -1


def _split_long_segment(segment: str, max_chars: int) -> list[str]:
    """Only used when a strongly-split sentence is still over max_chars.

    Commas first; whitespace next; 、/： only if nothing else works.
    """
    if max_chars <= 0 or len(segment) <= max_chars:
        return [segment]

    parts: list[str] = []
    remaining = segment
    min_cut = max(1, max_chars // 2)

    while len(remaining) > max_chars:
        cut = _find_cut(remaining, max_chars, _COMMA_CHARS)

        if cut < 0:
            space_index = remaining.rfind(" ", min_cut, max_chars + 1)
            if space_index >= 0:
                cut = space_index + 1

        if cut < 0:
            cut = _find_cut(remaining, max_chars, _WEAK_SENTENCE_END)

        if cut < 0:
            # Slightly over max is better than cutting through a word.
            if len(remaining) <= int(max_chars * 1.5):
                break
            cut = max_chars

        part = remaining[:cut].strip()
        if part:
            parts.append(part)
        remaining = remaining[cut:].strip()

    if remaining:
        parts.append(remaining)
    return parts


def _resolve_max_segment_chars(adapter=None) -> int:
    env = os.environ.get("TTS_MAX_SEGMENT_CHARS", "").strip()
    if env:
        return max(1, int(env))
    if adapter is not None:
        return int(getattr(adapter, "max_segment_chars", MAX_SEGMENT_CHARS))
    return MAX_SEGMENT_CHARS


def _ending_pause_ms(segment: str) -> int:
    s = (segment or "").rstrip()
    while s and s[-1] in _CLOSING_PUNCTUATION:
        s = s[:-1]
    if not s:
        return 0
    # Only punctuated joins get silence. Length cuts (if any) concatenate.
    return int(_PAUSE_MS.get(s[-1], 0))


def _silence_pcm16(ms: int) -> bytes:
    if ms <= 0:
        return b""
    n = int(SAMPLE_RATE * int(ms) / 1000)
    return b"\x00\x00" * max(0, n)


def _split_utterance(adapter, text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if os.environ.get("TTS_NO_SPLIT", "0") == "1":
        return [text]
    fit_tokens = bool(getattr(adapter, "needs_token_split", True))
    return _split_text_for_tts(text, _resolve_max_segment_chars(adapter), fit_tokens=fit_tokens)


def _split_to_fit_tokens(sentence: str, budget: int = 240) -> list[str]:
    """Keep clauses under Matcha MAX_TOKENS. Cut at ，/; only when needed."""
    if not sentence:
        return []
    from utils.phonetone import encode_for_matcha

    spoken = _strip_sentence_punct(sentence)
    if encode_for_matcha(spoken)["real_len"] < budget:
        return [sentence]
    parts = _split_long_segment(sentence, max(24, len(sentence) // 2))
    if len(parts) == 1:
        cut = max(24, len(sentence) // 2)
        parts = [sentence[:cut], sentence[cut:]]
    out: list[str] = []
    for part in parts:
        part = part.strip()
        if part:
            out.extend(_split_to_fit_tokens(part, budget))
    return out or [sentence]


def _split_text_for_tts(text: str, max_chars: int = MAX_SEGMENT_CHARS, fit_tokens: bool = True) -> list[str]:
    """Split for TTS without chopping every comma.

    Always cut at 。！？； / newline / English .!? (not 3.14 or U.S.).
    Length-splitting at ， is off unless max_chars > 0.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    segments: list[str] = []
    current: list[str] = []
    text_len = len(normalized)
    index = 0

    while index < text_len:
        char = normalized[index]
        current.append(char)
        is_boundary = char == "\n" or char in _STRONG_SENTENCE_END

        if char == ".":
            previous = normalized[index - 1] if index > 0 else ""
            following = normalized[index + 1] if index + 1 < text_len else ""
            is_decimal = previous.isdigit() and following.isdigit()
            is_abbrev = _is_single_letter_abbrev_dot(normalized, index)
            is_boundary = (not is_decimal) and (not is_abbrev) and (
                not following
                or following.isspace()
                or following in _CLOSING_PUNCTUATION
                or _is_cjk(following)
            )

        if is_boundary:
            next_index = index + 1
            while (
                next_index < text_len
                and normalized[next_index] in _CLOSING_PUNCTUATION
            ):
                current.append(normalized[next_index])
                next_index += 1
            index = next_index - 1

            sentence = "".join(current).strip()
            if sentence:
                if max_chars > 0:
                    segments.extend(_split_long_segment(sentence, max_chars))
                elif fit_tokens:
                    segments.extend(_split_to_fit_tokens(sentence))
                else:
                    segments.append(sentence)
            current = []

        index += 1

    tail = "".join(current).strip()
    if tail:
        if max_chars > 0:
            segments.extend(_split_long_segment(tail, max_chars))
        elif fit_tokens:
            segments.extend(_split_to_fit_tokens(tail))
        else:
            segments.append(tail)

    return segments


def _strip_sentence_punct(segment: str) -> str:
    """After split: drop sentence-end marks; keep commas and numeric/version dots.

    Never treat ``42.3`` / ``V2.8.1`` dots as sentence ends — that made ranking
    read versions/decimals differently from local PhoneTone TN tests.
    """
    import re

    if not segment:
        return segment
    protected: list[str] = []

    def _hold(match: re.Match) -> str:
        protected.append(match.group(0))
        return f"\0{len(protected) - 1}\0"

    # Keep URLs intact — stripping '.' before/without TN kills 点/斜杠 readings.
    out = re.sub(r"(?i)\bhttps?://[^\s，。；！？]+", _hold, segment)
    out = re.sub(r"(?i)\bwww\.[^\s，。；！？]+", _hold, out)
    # Keep decimals / dotted versions / NO.0917-style refs intact.
    out = re.sub(r"(?i)\b[a-z]*\d+(?:\.\d+)+\b", _hold, out)
    out = re.sub(r"(?i)\b[a-z]{1,8}\.\d+\b", _hold, out)
    out = re.sub(r"[。！？；!?;]+", " ", out)
    # ASCII / fullwidth '.' only when not inside a protected number/version/URL.
    out = re.sub(r"[．.]+", " ", out)
    for index, value in enumerate(protected):
        out = out.replace(f"\0{index}\0", value)
    return re.sub(r"\s+", " ", out).strip()


class TTSAdapter(ABC):
    @abstractmethod
    def _synthesize_segment(self, text: str) -> bytes: ...

    def split_text(self, text: str) -> list[str]:
        if getattr(self, "text_normalize", True):
            text = _normalize_tts_text(text)
        return _split_utterance(self, text)

    def synthesize(self, text: str) -> bytes:
        """Synthesize all segments and return one concatenated PCM stream."""
        return b"".join(self.synthesize_stream(text))

    def synthesize_stream(self, text: str):
        """Yield concatenated PCM chunks, synthesized one sentence at a time."""
        yield from self.synthesize_segments_stream(self.split_text(text))

    def synthesize_segments_stream(self, segments: list[str]):
        """Synthesize one clause at a time, insert a short pause, yield PCM.

        Mel/wav tensors from a clause are dropped before the next ORT run so
        BigVGAN peak tracks the longest clause, not the full paragraph.
        """
        import gc

        buffer = b""
        spoken_parts = []
        for segment in segments:
            spoken = _strip_sentence_punct(segment)
            if spoken:
                spoken_parts.append((segment, spoken))
        for i, (segment, spoken) in enumerate(spoken_parts):
            log.info("[tts] synth spoken[%d/%d]: %s", i + 1, len(spoken_parts), spoken)
            buffer += self._synthesize_segment(spoken)
            if i + 1 < len(spoken_parts) and os.environ.get("TTS_SEGMENT_PAUSE", "1") != "0":
                buffer += _silence_pcm16(_ending_pause_ms(segment))
            gc.collect()
            while len(buffer) >= CHUNK_BYTES:
                yield buffer[:CHUNK_BYTES]
                buffer = buffer[CHUNK_BYTES:]
        if buffer:
            yield buffer


def _resample_to_16k(samples, src_rate: int):
    """Gentleman models are 16 kHz. Reject any other rate instead of pulling scipy."""
    if int(src_rate) == SAMPLE_RATE:
        return samples
    raise RuntimeError("Gentleman TTS expected 16 kHz, got %s" % src_rate)


def _float_samples_to_pcm16(samples) -> bytes:
    import numpy as np

    x = np.asarray(samples, dtype=np.float32)
    x = np.clip(x * 32767.0, -32768, 32767).astype(np.int16)
    return x.tobytes()



class MatchaPhoneToneOrtAdapter(TTSAdapter):
    """PhoneTone frontend + Matcha ONNX + Vocos ONNX.

    Default is two CUDA sessions (Matcha then Vocos, sess.run). If
    TTS_GENTLEMAN_E2E_ONNX or model_dir/gentleman-e2e.onnx exists, one
    merged session is used instead. Missing e2e file always falls back.
    """

    def __init__(
        self,
        model_dir: str,
        speaker_id: int = 0,
        speed: float = 1.0,
        model_name: str = "tts_matcha_gentleman",
        hw_provider: str = "cuda",
        num_threads: int = 2,
        noise_scale: float = 0.667,
    ):
        import os

        from utils.model_downloader import ensure_model
        from utils.phonetone import PhoneToneFrontend, encode_for_matcha

        ensure_model(model_name, model_dir)
        self._frontend = PhoneToneFrontend(model_dir)
        self._encode = encode_for_matcha
        self._sid = speaker_id
        self._speed = speed
        self._noise_scale = float(noise_scale)
        self.max_segment_chars = MAX_SEGMENT_CHARS
        self.text_normalize = False
        self._model_sr = 16000
        acoustic = _phonetone_acoustic_path(model_dir)
        vocoder = _phonetone_vocoder_path(model_dir)
        e2e = _phonetone_e2e_path(model_dir)
        if e2e:
            if not os.path.isfile(e2e):
                raise FileNotFoundError(e2e)
            e2e_mb = os.environ.get("TTS_E2E_GPU_MEM_LIMIT_MB", "640").strip()
            gpu_mb = int(e2e_mb) if e2e_mb.isdigit() and int(e2e_mb) > 0 else 640
            self._sess = _load_onnx_session(
                e2e, hw_provider, num_threads, gpu_mem_limit_mb=gpu_mb
            )[1]
            self._vocoder = None
            self._e2e = True
            log.info(
                "[tts] Gentleman e2e loaded: onnx=%s providers=%s frontend=%s gpu_mem_limit_mb=%s",
                e2e,
                self._sess.get_providers(),
                self._frontend.release,
                gpu_mb,
            )
            return
        self._e2e = False
        if not os.path.isfile(acoustic):
            raise FileNotFoundError(acoustic)
        if not os.path.isfile(vocoder):
            raise FileNotFoundError(vocoder)
        self._sess = _load_onnx_session(acoustic, hw_provider, num_threads)[1]
        self._vocoder = _WaveformOrt(vocoder, hw_provider, num_threads)
        log.info(
            "[tts] Gentleman loaded: acoustic=%s vocoder=%s providers=%s frontend=%s",
            os.path.basename(acoustic),
            os.path.basename(vocoder),
            self._sess.get_providers(),
            self._frontend.release,
        )

    def split_text(self, text: str) -> list[str]:
        from utils.phonetone.speak_patch import patch_speak_text

        raw = (text or "").strip()
        patched = patch_speak_text(raw)
        tn_text = self._frontend.normalize(raw)
        log.info("[tts] frontend input: %s", raw)
        log.info("[tts] frontend patched: %s", patched)
        log.info("[tts] frontend TN: %s", tn_text)
        segments = _split_utterance(self, tn_text)
        for index, segment in enumerate(segments):
            log.info("[tts] segment[%d/%d]: %s", index + 1, len(segments), segment)
        return segments

    def _synthesize_segment(self, text: str) -> bytes:
        import numpy as np
        from utils.matcha_ort import crop_mel

        packed = self._encode(
            text,
            temperature=self._noise_scale,
            length_scale=1.0 / max(self._speed, 1e-3),
        )
        feeds = {name: packed[name] for name in ("x", "x_lengths", "tones", "languages", "scales")}
        in_names = {i.name for i in self._sess.get_inputs()}
        if "x_length" in in_names and "x_lengths" not in in_names:
            feeds["x_length"] = feeds.pop("x_lengths")
        out = _ort_outputs(self._sess, feeds)
        if getattr(self, "_e2e", False):
            wav = out.get("v/wav", out.get("wav"))
            if wav is None:
                raise RuntimeError("no wav in %s" % list(out))
            samples = np.asarray(wav, dtype=np.float32).reshape(-1)
            mel_lengths = out.get("mel_lengths")
            if mel_lengths is not None:
                cap = max(1, int(np.asarray(mel_lengths).reshape(-1)[0]) * 256)
                samples = samples[:cap]
            samples = _resample_to_16k(samples, self._model_sr)
            return _float_samples_to_pcm16(samples)
        mel = out.get("mel")
        if mel is None:
            raise RuntimeError("no mel in %s" % list(out))
        cropped = crop_mel(mel, packed["real_len"], out.get("mel_lengths"))
        mel_bct = np.ascontiguousarray(cropped[None, ...], dtype=np.float32)
        samples = self._vocoder.infer(mel_bct)
        samples = _resample_to_16k(samples, self._model_sr)
        return _float_samples_to_pcm16(samples)


class MossOrtAdapter(TTSAdapter):
    """MOSS-TTS-Nano (Audio Tokenizer + LM) ONNX, CPU-only.

    Multilingual board backend: language is auto-detected per segment and
    mapped to a builtin voice (default Saki for every language, configurable
    per language via cfg["voices"]). The codec outputs 48 kHz stereo; we
    lowpass-decimate to 16 kHz mono PCM16 to match the bundle contract
    (audio/pcm-16k).

    MOSS does its own punctuation-driven prosody, so sentence marks are kept
    (the Matcha path strips them because PhoneTone re-adds its own).
    """

    needs_token_split = False

    def __init__(
        self,
        model_dir: str,
        model_name: str = "tts_moss_nano",
        num_threads: int = 4,
        voices: dict | None = None,
        sample_mode: str | None = None,
        do_sample: bool | None = None,
        max_new_frames: int | None = None,
        hw_provider: str = "cpu",
    ):
        import os

        from utils.model_downloader import ensure_model
        from utils.moss_ort import MossOrtTTS

        ensure_model(model_name, model_dir)
        provider = "cuda" if str(hw_provider).lower() == "cuda" else "cpu"
        try:
            self._tts = MossOrtTTS(
                model_dir,
                num_threads=num_threads,
                voices=voices,
                sample_mode=sample_mode,
                do_sample=do_sample,
                max_new_frames=max_new_frames,
                execution_provider=provider,
            )
        except Exception:
            if provider != "cuda":
                raise
            log.warning("[tts] MOSS CUDA init failed; falling back to CPU", exc_info=True)
            provider = "cpu"
            self._tts = MossOrtTTS(
                model_dir,
                num_threads=num_threads,
                voices=voices,
                sample_mode=sample_mode,
                do_sample=do_sample,
                max_new_frames=max_new_frames,
                execution_provider="cpu",
            )
        self._model_dir = model_dir
        self.max_segment_chars = 0
        self.text_normalize = False
        self._model_sr = SAMPLE_RATE
        log.info(
            "[tts] MOSS loaded: model_dir=%s voices=%s threads=%s",
            model_dir,
            voices,
            num_threads,
        )

    def split_text(self, text: str) -> list[str]:
        raw = (text or "").strip()
        log.info("[tts] moss input: %s", raw)
        segments = _split_utterance(self, raw)
        for index, segment in enumerate(segments):
            log.info("[tts] segment[%d/%d]: %s", index + 1, len(segments), segment)
        return segments

    def synthesize_segments_stream(self, segments: list[str]):
        """Streaming per segment: first 16k PCM chunk is yielded after ~1 s
        (prefill + first frame), not after the whole sentence.
        """
        import gc

        buffer = b""
        parts = [s.strip() for s in segments if s and s.strip()]
        for i, segment in enumerate(parts):
            log.info("[tts] moss synth[%d/%d]: %s", i + 1, len(parts), segment)
            for chunk_f32 in self._tts.synthesize_stream(segment):
                buffer += _float_samples_to_pcm16(chunk_f32)
                while len(buffer) >= CHUNK_BYTES:
                    yield buffer[:CHUNK_BYTES]
                    buffer = buffer[CHUNK_BYTES:]
            if i + 1 < len(parts) and os.environ.get("TTS_SEGMENT_PAUSE", "1") != "0":
                buffer += _silence_pcm16(_ending_pause_ms(segment))
            gc.collect()
        if buffer:
            yield buffer

    def _synthesize_segment(self, text: str) -> bytes:
        return self._tts.synthesize(text)


def _build_tts_adapter(cfg: dict) -> TTSAdapter:
    backend = cfg.get("backend", "matcha")
    if backend == "moss":
        adapter = MossOrtAdapter(
            model_dir=cfg.get("model_dir", "/models/moss-tts-nano-16k"),
            model_name=cfg.get("model_name", "tts_moss_nano"),
            num_threads=int(cfg.get("num_threads", 4)),
            voices=cfg.get("voices") or None,
            sample_mode=cfg.get("sample_mode") or None,
            do_sample=cfg.get("do_sample"),
            max_new_frames=int(cfg["max_new_frames"]) if cfg.get("max_new_frames") else None,
            hw_provider=cfg.get("hw_provider", "cpu"),
        )
        adapter.max_segment_chars = int(cfg.get("max_segment_chars", 0))
        adapter.text_normalize = False
        return adapter
    if backend != "matcha":
        raise ValueError("Gentleman image only supports backend=matcha|moss, got %r" % backend)
    adapter = MatchaPhoneToneOrtAdapter(
        model_dir=cfg.get("model_dir", "/models/matcha-gentleman-phonetone-16k"),
        speaker_id=int(cfg.get("speaker_id", 0)),
        speed=float(cfg.get("speed", 1.0)),
        model_name=cfg.get("model_name", "tts_matcha_gentleman"),
        hw_provider=cfg.get("hw_provider", "cuda"),
        num_threads=int(cfg.get("num_threads", 2)),
        noise_scale=float(cfg.get("noise_scale", 0.667)),
    )
    adapter.max_segment_chars = int(cfg.get("max_segment_chars", MAX_SEGMENT_CHARS))
    adapter.prefer_single_pass = bool(cfg.get("prefer_single_pass", True))
    adapter.text_normalize = False
    return adapter


# ── ROS2 Node ─────────────────────────────────────────────────────────────────

class _TTSNode(Node):
    def __init__(
        self,
        input_topic: Optional[str],
        adapter: Optional[TTSAdapter],
        node_suffix: str = '',
        realtime_pacing: bool = False,
    ):
        node_name = f"tts_{node_suffix}" if node_suffix else "tts"
        super().__init__(node_name)
        self._input_topic  = input_topic or ''
        self._output_topic = f"{input_topic}/tts" if input_topic else '/perception/tts'
        self._adapter      = adapter
        self._realtime_pacing = realtime_pacing
        self.state         = "idle"
        self._text_queue   = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event   = threading.Event()
        from audio_msgs.msg import AudioChunk
        self._pub = self.create_publisher(AudioChunk, self._output_topic, _LOW_LAT_QOS)
        if input_topic:
            self._sub = self.create_subscription(String, self._input_topic, self._text_cb, _LOW_LAT_QOS)
        else:
            self._sub = None
        log.info(f"[tts] node created: subscribing={self._input_topic or '(none)'}, publishing={self._output_topic}")

    def start(self) -> dict:
        while not self._text_queue.empty():
            try: self._text_queue.get_nowait()
            except Exception: break
        if self.state == "running":
            return self._status_dict()
        if not self._adapter:
            raise RuntimeError("TTS adapter not configured")
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        self.state = "running"
        return self._status_dict()

    def stop(self) -> dict:
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3)
        self.state = "idle"
        return {"state": "idle"}

    def enqueue(self, text: str):
        if self.state != "running":
            raise RuntimeError("TTS not running; call start first")
        self._text_queue.put(text)

    def _wait_subscriber_then_settle(self) -> None:
        """Block until ranking side has subscribed, then settle 200ms before first frame.

        Blind TTS_PUBLISH_DELAY_MS is not used: discovery is dynamic, so we wait for
        get_subscription_count() >= 1, then a short settle so the listener is ready.
        """
        import time as _time

        timeout_s = float(os.environ.get("TTS_SUBSCRIBER_WAIT_S", "5"))
        settle_ms = float(os.environ.get("TTS_SUBSCRIBER_SETTLE_MS", "200"))
        t_wait0 = _time.monotonic()
        deadline = t_wait0 + max(0.0, timeout_s)
        while not self._stop_event.is_set():
            count = int(self._pub.get_subscription_count())
            if count >= 1:
                waited_ms = (_time.monotonic() - t_wait0) * 1000.0
                log.info(
                    "[tts] subscriber ready count=%d after %.0fms; settle %.0fms",
                    count,
                    waited_ms,
                    settle_ms,
                )
                if settle_ms > 0:
                    _time.sleep(settle_ms / 1000.0)
                return
            if _time.monotonic() >= deadline:
                log.warning(
                    "[tts] no subscriber within %.1fs (count=0); publishing anyway",
                    timeout_s,
                )
                return
            _time.sleep(0.02)
    def _publish_frame(self, frame: bytes, frames_sent: int, t0: Optional[float], frame_duration: float):
        from audio_msgs.msg import AudioChunk
        import time as _time

        if self._realtime_pacing and t0 is not None:
            target = t0 + frames_sent * frame_duration
            now = _time.monotonic()
            if now < target:
                _time.sleep(target - now)
        msg = AudioChunk()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "audio/pcm-16k"
        msg.data = list(frame)
        self._pub.publish(msg)
        return frames_sent + 1

    def _text_cb(self, msg: String):
        if self.state != "running": return
        try:
            text = json.loads(msg.data).get("text","")
        except Exception:
            text = msg.data.strip()
        if text:
            log.info(f"[tts] received text from topic: {text[:50]}...")
            self._text_queue.put(text)

    def _worker(self):
        from audio_msgs.msg import AudioChunk
        import time as _time

        # Real-time pacing: publish frames at playback rate to avoid bursts/gaps
        FRAME_DURATION = CHUNK_BYTES / (SAMPLE_RATE * 2)  # 0.1s per 3200-byte frame
        PREBUF_FRAMES  = 1  # 1 frame (~100ms); was 3 (~300ms) for lower judged TTFT

        while not self._stop_event.is_set():
            try:
                text = self._text_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                import time as _time
                t_start = _time.monotonic()
                total = 0
                buf   = b''
                t0    = None  # wall-clock start of playback
                frames_sent = 0
                prebuf = []   # pre-buffer queue
                first_audio_latency = None
                segments = self._adapter.split_text(text)
                if not segments:
                    continue

                log.info(
                    f"[tts] split {len(text)} chars into {len(segments)} segment(s): "
                    f"{[len(segment) for segment in segments]}"
                )

                subscriber_gate_pending = True

                # Decouple offline sentence synthesis from real-time publishing.
                # The producer can generate the next sentence while audio from
                # the current sentence is being paced to the ROS2 topic.
                audio_queue = queue.Queue(maxsize=SYNTH_QUEUE_FRAMES)
                stream_end = object()
                producer_error = []
                synth_elapsed = [0.0]

                def _queue_put(item) -> bool:
                    while not self._stop_event.is_set():
                        try:
                            audio_queue.put(item, timeout=0.1)
                            return True
                        except queue.Full:
                            continue
                    return False

                def _produce_audio():
                    synth_t0 = _time.monotonic()
                    try:
                        for chunk in self._adapter.synthesize_segments_stream(segments):
                            if self._stop_event.is_set() or not _queue_put(chunk):
                                break
                    except Exception as exc:
                        producer_error.append(exc)
                    finally:
                        synth_elapsed[0] = _time.monotonic() - synth_t0
                        _queue_put(stream_end)

                producer_thread = threading.Thread(target=_produce_audio, daemon=True)
                producer_thread.start()

                while not self._stop_event.is_set():
                    try:
                        raw_chunk = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        if not producer_thread.is_alive() and audio_queue.empty():
                            break
                        continue

                    if raw_chunk is stream_end:
                        break
                    if first_audio_latency is None:
                        first_audio_latency = _time.monotonic() - t_start

                    if self._stop_event.is_set():
                        break
                    buf  += raw_chunk
                    total += len(raw_chunk)
                    # split into CHUNK_BYTES frames
                    while len(buf) >= CHUNK_BYTES:
                        frame = buf[:CHUNK_BYTES]
                        buf   = buf[CHUNK_BYTES:]

                        # Pre-buffer phase: accumulate a few frames before pacing
                        if t0 is None:
                            prebuf.append(frame)
                            if len(prebuf) >= PREBUF_FRAMES:
                                if subscriber_gate_pending:
                                    self._wait_subscriber_then_settle()
                                    subscriber_gate_pending = False
                                t0 = _time.monotonic() if self._realtime_pacing else 0.0
                                log.info(
                                    "[tts] first publish: prebuf=%d frames pacing=%s",
                                    len(prebuf),
                                    self._realtime_pacing,
                                )
                                for pf in prebuf:
                                    frames_sent = self._publish_frame(
                                        pf, frames_sent, t0, FRAME_DURATION
                                    )
                                prebuf = []
                            continue

                        frames_sent = self._publish_frame(
                            frame, frames_sent, t0, FRAME_DURATION
                        )

                # Flush any remaining pre-buffer (short utterances < PREBUF_FRAMES)
                if prebuf and not self._stop_event.is_set():
                    if t0 is None:
                        if subscriber_gate_pending:
                            self._wait_subscriber_then_settle()
                            subscriber_gate_pending = False
                        t0 = _time.monotonic() if self._realtime_pacing else 0.0
                    for pf in prebuf:
                        frames_sent = self._publish_frame(
                            pf, frames_sent, t0, FRAME_DURATION
                        )

                # flush remainder
                if buf and not self._stop_event.is_set():
                    if self._realtime_pacing and t0 is not None:
                        target = t0 + frames_sent * FRAME_DURATION
                        now = _time.monotonic()
                        if now < target:
                            _time.sleep(target - now)
                    from audio_msgs.msg import AudioChunk
                    msg = AudioChunk()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.format = "audio/pcm-16k"
                    msg.data = list(buf)
                    self._pub.publish(msg)
                    frames_sent += 1

                producer_thread.join(timeout=0.2)
                if producer_error:
                    raise producer_error[0]

                elapsed = _time.monotonic() - t_start
                audio_duration = total / (SAMPLE_RATE * 2) if total else 0.0
                synth_rtf = (synth_elapsed[0] / audio_duration) if audio_duration > 0 else 0.0
                e2e_rtf = (elapsed / audio_duration) if audio_duration > 0 else 0.0
                mem_mb = _process_rss_mb()
                first_audio_text = (
                    f", TTFT={first_audio_latency:.2f}s"
                    if first_audio_latency is not None
                    else ""
                )
                log.info(
                    f"[tts] spoke {len(text)} chars in {len(segments)} segment(s) "
                    f"→ {total} bytes ({frames_sent} frames) in {elapsed:.2f}s"
                    f"{first_audio_text}, "
                    f"audio={audio_duration:.2f}s, synth_RTF={synth_rtf:.2f}, "
                    f"e2e_RTF={e2e_rtf:.2f}, memory_mb={mem_mb:.1f}"
                )
            except Exception as e:
                log.error(f"[tts] synthesis error: {e}", exc_info=True)

    def _status_dict(self) -> dict:
        return {
            "state":     self.state,
            "topic_in":  [{"topic": self._input_topic,  "format": "data/json",     "desc": "text to synthesize"}],
            "topic_out": [{"topic": self._output_topic, "format": "audio/pcm-16k", "desc": "synthesized PCM audio"}],
        }


def _warmup_tts_adapter(adapter: TTSAdapter, text: str = "。") -> None:
    """Run one silent synthesis to warm up ORT/CUDA before the first speak request."""
    import time as _time

    if getattr(adapter, "text_normalize", True):
        text = _normalize_tts_text(text)
    log.info(f"[tts] warmup starting: text={text!r}")
    t0 = _time.monotonic()
    pcm = adapter._synthesize_segment(text)
    elapsed = _time.monotonic() - t0
    log.info(f"[tts] warmup done in {elapsed:.2f}s ({len(pcm)} bytes)")


def _run_tts_warmup(adapter: TTSAdapter, plugin_cfg: dict) -> None:
    """Warm up ORT/CUDA (+ ZH and ZH/EN G2P paths) before first speak."""
    if not plugin_cfg.get("warmup", True):
        return
    texts = plugin_cfg.get("warmup_texts")
    if texts is None:
        single = plugin_cfg.get(
            "warmup_text",
            "你好，欢迎使用语音合成服务，这是一段预热测试文本。",
        )
        texts = [single] if isinstance(single, str) else list(single or [])
    elif isinstance(texts, str):
        texts = [texts]
    else:
        texts = [t for t in texts if t]
    if not texts:
        texts = ["你好，欢迎使用语音合成服务，这是一段预热测试文本。"]
    infer_ok = False
    try:
        for i, text in enumerate(texts):
            log.info(f"[tts] warmup [{i + 1}/{len(texts)}]")
            _warmup_tts_adapter(adapter, text)
            _dump_stage("after_warmup_%d" % (i + 1))
        infer_ok = True
        _maybe_malloc_trim("after_warmup")
        from utils.model_downloader import drop_file_pages

        for path in (
            getattr(adapter, "_model_dir", None),
            getattr(adapter, "_acoustic_path", None),
            getattr(adapter, "_vocoder_path", None),
        ):
            if path:
                drop_file_pages(path)
    except Exception as e:
        log.warning(f"[tts] warmup failed (non-fatal): {e}", exc_info=True)


def _start_warmup_background(adapter: TTSAdapter, plugin_cfg: dict) -> None:
    """Optional async warmup (warmup_async=true)."""
    def _run() -> None:
        _run_tts_warmup(adapter, plugin_cfg)

    threading.Thread(target=_run, daemon=True, name="tts-warmup").start()


# ── Plugin ────────────────────────────────────────────────────────────────────

class TTSPlugin:
    PREFIX = "tts"

    def __init__(self, plugin_cfg: dict, executor):
        self._cfg      = plugin_cfg
        self._loading  = False
        self._load_error = None
        self._realtime_pacing = bool(plugin_cfg.get("realtime_pacing", False))
        _dump_stage("ros_python")
        try:
            self._adapter  = _build_tts_adapter(plugin_cfg)
        except Exception as e:
            log.error(f"[tts] failed to load model: {e}", exc_info=True)
            self._adapter = None
            self._load_error = str(e)
        if self._adapter:
            if plugin_cfg.get("warmup_async", False):
                _start_warmup_background(self._adapter, plugin_cfg)
            else:
                _run_tts_warmup(self._adapter, plugin_cfg)
        self._nodes: dict[str, _TTSNode] = {}
        self._instance_configs: dict[str, dict] = {}
        self._executor = executor
        log.info(
            "[tts] plugin init: gentleman-ort backend=%s speaker_id=%s speed=%s "
            "max_segment_chars=%s realtime_pacing=%s",
            plugin_cfg.get("backend", "matcha"),
            plugin_cfg.get("speaker_id", 0),
            plugin_cfg.get("speed", 1.0),
            plugin_cfg.get("max_segment_chars", MAX_SEGMENT_CHARS),
            self._realtime_pacing,
        )

    def get_tools(self) -> list:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name == "tts" else name
        instance_id = args.get("instance_id", "")

        if action == "info":
            if self._loading:
                return {
                    "name": "TTS", "manufacture": "Embodied", "model": "tts",
                    "state": "loading",
                    "desc": "Downloading TTS model...",
                }
            if self._load_error:
                return {
                    "name": "TTS", "manufacture": "Embodied", "model": "tts",
                    "state": "error",
                    "desc": f"Model load failed: {self._load_error}",
                }
            input_topic = args.get("input_topic", "")
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                return {
                    "name": "TTS", "manufacture": "Embodied", "model": "tts",
                    "state": node.state,
                    "topic_in":  [{"topic": node._input_topic,  "format": "data/json",     "desc": ""}],
                    "topic_out": [{"topic": node._output_topic, "format": "audio/pcm-16k", "desc": ""}],
                    "desc": "TTS service — converts text to audio/pcm-16k",
                }
            if instance_id:
                # Instance requested but not running — return inferred topics for this instance only.
                inferred_out = f"{input_topic}/tts" if input_topic else "/perception/tts"
                return {
                    "name": "TTS", "manufacture": "Embodied", "model": "tts",
                    "state": "idle",
                    "topic_in":  [{"topic": input_topic,  "format": "data/json",     "desc": ""}] if input_topic else [],
                    "topic_out": [{"topic": inferred_out, "format": "audio/pcm-16k", "desc": ""}],
                    "desc": "TTS service — converts text to audio/pcm-16k",
                }
            # Aggregate info (no instance_id = ping/overview only)
            if self._nodes:
                topics_in = [{"topic": n._input_topic, "format": "data/json", "desc": ""} for n in self._nodes.values()]
                topics_out = [{"topic": n._output_topic, "format": "audio/pcm-16k", "desc": ""} for n in self._nodes.values()]
                states = list(set(n.state for n in self._nodes.values()))
                state = "running" if "running" in states else states[0] if states else "idle"
            else:
                inferred_out = f"{input_topic}/tts" if input_topic else "/perception/tts"
                topics_in = [{"topic": input_topic, "format": "data/json", "desc": ""}]
                topics_out = [{"topic": inferred_out, "format": "audio/pcm-16k", "desc": ""}]
                state = "idle"
            return {
                "name": "TTS", "manufacture": "Embodied", "model": "tts",
                "state": state,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "TTS service — converts text to audio/pcm-16k",
            }

        elif action == "start":
            if self._loading:
                return {"state": "loading", "message": "TTS model is being downloaded, please wait..."}
            if self._load_error:
                return {"state": "error", "message": f"TTS model failed to load: {self._load_error}"}
            if not self._adapter:
                return {"state": "error", "message": "TTS model not loaded"}
            input_topic = args.get("input_topic") or ''
            node_key = instance_id or input_topic or '_default'
            # Clean up _default node if it would conflict with this instance
            if '_default' in self._nodes and node_key != '_default':
                default_node = self._nodes['_default']
                if default_node._input_topic == input_topic or default_node._output_topic == (f"{input_topic}/tts" if input_topic else '/perception/tts'):
                    default_node.stop()
                    self._executor.remove_node(default_node)
                    del self._nodes['_default']
            if node_key not in self._nodes:
                node = _TTSNode(
                    input_topic or None,
                    self._adapter,
                    node_suffix=node_key.replace('/', '_').replace('-', '_'),
                    realtime_pacing=self._realtime_pacing,
                )
                self._executor.add_node(node)
                self._nodes[node_key] = node
            elif input_topic and self._nodes[node_key]._input_topic != input_topic:
                # Input topic changed for existing instance — recreate
                old_node = self._nodes[node_key]
                old_node.stop()
                self._executor.remove_node(old_node)
                node = _TTSNode(
                    input_topic,
                    self._adapter,
                    node_suffix=node_key.replace('/', '_').replace('-', '_'),
                    realtime_pacing=self._realtime_pacing,
                )
                self._executor.add_node(node)
                self._nodes[node_key] = node
            return self._nodes[node_key].start()

        elif action == "stop":
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                result = node.stop()
                self._executor.remove_node(node)
                del self._nodes[instance_id]
                return result
            elif not instance_id and self._nodes:
                for key in list(self._nodes.keys()):
                    self._nodes[key].stop()
                    self._executor.remove_node(self._nodes[key])
                    del self._nodes[key]
                return {"state": "idle"}
            return {"state": "idle"}

        elif action == "speak":
            if self._loading:
                return {"state": "loading", "message": "TTS model is being downloaded, please wait..."}
            if self._load_error or not self._adapter:
                return {"state": "error", "message": f"TTS model not available: {self._load_error or 'not loaded'}"}
            text = args.get("text", "")
            if not text:
                raise ValueError("text is required")
            # Find any existing running node to reuse
            node = None
            for n in self._nodes.values():
                if n.state == "running":
                    node = n
                    break
            if node is None:
                # No running node — use instance key or fallback
                node_key = instance_id or '_default'
                if node_key not in self._nodes:
                    input_topic = args.get("input_topic") or None
                    adapter = self._adapter
                    if instance_id and instance_id in self._instance_configs:
                        inst_adapter = _build_tts_adapter(self._instance_configs[instance_id])
                        if inst_adapter:
                            adapter = inst_adapter
                    node = _TTSNode(
                        input_topic,
                        adapter,
                        node_suffix=node_key.replace('/', '_').replace('-', '_'),
                        realtime_pacing=self._realtime_pacing,
                    )
                    self._executor.add_node(node)
                    self._nodes[node_key] = node
                else:
                    node = self._nodes[node_key]
                if node.state != "running":
                    node.start()
            node.enqueue(text)
            return {"status": "queued", "text": text}

        elif action == "config":
            cfg = {
                k: v for k, v in args.items()
                if k not in ('action', 'instance_id') and v is not None and v != ''
            }
            if 'speaker_id' in cfg:
                self._cfg['speaker_id'] = int(cfg['speaker_id'])
            if 'speed' in cfg:
                self._cfg['speed'] = float(cfg['speed'])
            # Prefer in-place update (eval only tweaks sid/speed). Full rebuild
            # only when adapter missing (e.g. init load failed).
            if self._adapter is not None:
                if hasattr(self._adapter, '_sid'):
                    self._adapter._sid = int(self._cfg.get('speaker_id', 0))
                if hasattr(self._adapter, '_speed'):
                    self._adapter._speed = float(self._cfg.get('speed', 1.0))
            else:
                try:
                    self._adapter = _build_tts_adapter(self._cfg)
                    self._load_error = None
                    # Init load failed earlier; warm now so first speak is not cold.
                    _run_tts_warmup(self._adapter, self._cfg)
                except Exception as e:
                    self._load_error = str(e)
                    log.error(f"[tts] config rebuild failed: {e}", exc_info=True)
                    raise
            for key in list(self._nodes.keys()):
                self._nodes[key].stop()
                self._executor.remove_node(self._nodes[key])
                del self._nodes[key]
            return {"status": "configured"}

        return None

    def synthesize_raw(self, text: str) -> bytes:
        """Synthesize text and return raw PCM bytes (16kHz 16-bit mono)."""
        if not self._adapter:
            raise RuntimeError("TTS adapter not configured")
        return self._adapter.synthesize(text)

