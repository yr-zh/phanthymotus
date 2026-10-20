"""
utils/moss_ort.py — MOSS-TTS-Nano (Audio Tokenizer + LM) ONNX runtime.

The core inference class below is vendored from the OpenMOSS MOSS-TTS-Nano
project (ort_cpu_runtime.py), licensed under the Apache License 2.0:
  https://github.com/OpenMOSS/MOSS-TTS-Nano
Modifications: disable GatherSliceToSplitFusion (ORT 1.23 cannot parse
external-data Split tensors), plus the phanthymotus MossOrtTTS wrapper at the
bottom (language-aware voice selection, 48k stereo -> 16k mono resampling).
"""

from __future__ import annotations

import json
import os
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnxruntime as ort

SAMPLE_MODE_GREEDY = "greedy"
SAMPLE_MODE_FIXED = "fixed"
SAMPLE_MODE_FULL = "full"
EXECUTION_PROVIDER_CPU = "cpu"
EXECUTION_PROVIDER_CUDA = "cuda"

MANIFEST_CANDIDATE_RELATIVE_PATHS = (
    "browser_poc_manifest.json",
    "MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json",
    "MOSS-TTS-Nano-ONNX-CPU/browser_poc_manifest.json",
)
MODEL_DIR_ALIAS_MAP = {
    "MOSS-TTS-Nano-ONNX-CPU": "MOSS-TTS-Nano-100M-ONNX",
    "MOSS-Audio-Tokenizer-Nano-ONNX-CPU": "MOSS-Audio-Tokenizer-Nano-ONNX",
}


def _argmax(values: np.ndarray) -> int:
    return int(np.argmax(values))


def _normalize_execution_provider(raw_execution_provider: str | None) -> str:
    normalized = str(raw_execution_provider or EXECUTION_PROVIDER_CPU).strip().lower()
    if normalized in {EXECUTION_PROVIDER_CPU, "CPUExecutionProvider".lower()}:
        return EXECUTION_PROVIDER_CPU
    if normalized in {EXECUTION_PROVIDER_CUDA, "gpu", "CUDAExecutionProvider".lower()}:
        return EXECUTION_PROVIDER_CUDA
    raise ValueError("execution_provider must be one of: cpu, cuda")


def _resolve_ort_providers(execution_provider: str) -> list[Any]:
    normalized = _normalize_execution_provider(execution_provider)
    if normalized == EXECUTION_PROVIDER_CPU:
        return ["CPUExecutionProvider"]
    available_providers = set(ort.get_available_providers())
    if "CUDAExecutionProvider" not in available_providers:
        available = ", ".join(ort.get_available_providers()) or "none"
        raise RuntimeError(
            "CUDAExecutionProvider was requested, but this onnxruntime build does not expose it. "
            "Install onnxruntime-gpu that matches your CUDA/cuDNN runtime. "
            f"Available providers: {available}"
        )
    preload_dlls = getattr(ort, "preload_dlls", None)
    if callable(preload_dlls):
        preload_dlls()
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]




def _ort_run(sess, feeds):
    """session.run with periodic GPU arena shrinkage.

    ORT CUDA EP sessions in one process share one BFC arena; without shrinkage
    the arena only grows and a few requests exhaust gpu_mem_limit. Shrinking
    every 16th run keeps steady-state at the working set without paying the
    shrink cost on every 80 ms frame.
    """
    _ort_run.counter = getattr(_ort_run, "counter", 0) + 1
    if _ort_run.counter % 8 == 0:
        run_options = ort.RunOptions()
        run_options.add_run_config_entry(
            "memory.enable_memory_arena_shrinkage", "gpu:0"
        )
        return sess.run(None, feeds, run_options=run_options)
    return sess.run(None, feeds)

def _flatten3d_int32(nested: list[list[list[int]]]) -> tuple[np.ndarray, list[int]]:
    dim0 = len(nested)
    dim1 = len(nested[0])
    dim2 = len(nested[0][0])
    data = np.zeros((dim0 * dim1 * dim2,), dtype=np.int32)
    offset = 0
    for i in range(dim0):
        for j in range(dim1):
            for k in range(dim2):
                data[offset] = int(nested[i][j][k])
                offset += 1
    return data, [dim0, dim1, dim2]


def _flatten2d_int32(nested: list[list[int]]) -> tuple[np.ndarray, list[int]]:
    dim0 = len(nested)
    dim1 = len(nested[0])
    data = np.zeros((dim0 * dim1,), dtype=np.int32)
    offset = 0
    for i in range(dim0):
        for j in range(dim1):
            data[offset] = int(nested[i][j])
            offset += 1
    return data, [dim0, dim1]


def _slice_channel_major_audio(audio: np.ndarray, start_sample: int = 0, end_sample: int | None = None) -> list[np.ndarray]:
    if audio.ndim != 3 or audio.shape[0] != 1:
        raise ValueError(f"Unexpected audio tensor shape: {audio.shape}")
    channels = int(audio.shape[1])
    total_samples = int(audio.shape[2])
    start = max(0, int(start_sample))
    end = total_samples if end_sample is None else max(start, min(int(end_sample), total_samples))
    return [audio[0, channel_index, start:end].astype(np.float32, copy=False) for channel_index in range(channels)]


def _extract_last_hidden(hidden_states: np.ndarray) -> np.ndarray:
    if hidden_states.ndim == 2:
        return hidden_states.astype(np.float32, copy=False)
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError(f"Unexpected global_hidden shape: {hidden_states.shape}")
    return hidden_states[:, -1, :].astype(np.float32, copy=False)


def _apply_repetition_penalty(values: np.ndarray, previous_token_ids: list[int], repetition_penalty: float) -> np.ndarray:
    if not previous_token_ids or repetition_penalty == 1.0:
        return values
    result = values.copy()
    for token_id in set(int(item) for item in previous_token_ids):
        if token_id < 0 or token_id >= result.shape[0]:
            continue
        result[token_id] = result[token_id] * repetition_penalty if result[token_id] < 0 else result[token_id] / repetition_penalty
    return result


def _argmax_with_repetition_penalty(values: np.ndarray, previous_token_set: set[int], repetition_penalty: float) -> int:
    best_index = 0
    best_value = float("-inf")
    apply_penalty = bool(previous_token_set) and repetition_penalty != 1.0
    for index, value in enumerate(values):
        score = float(value)
        if apply_penalty and index in previous_token_set:
            score = score * repetition_penalty if score < 0 else score / repetition_penalty
        if score > best_value:
            best_value = score
            best_index = index
    return int(best_index)


def _softmax(values: np.ndarray) -> np.ndarray:
    max_value = float(np.max(values))
    shifted = np.asarray(values - max_value, dtype=np.float64)
    exps = np.exp(shifted)
    return exps / np.sum(exps, dtype=np.float64)


def _sample_from_scores(
    values: np.ndarray,
    *,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    rng: np.random.Generator,
) -> int:
    if not do_sample:
        return _argmax(values)
    if not (temperature > 0):
        raise ValueError("temperature must be positive when do_sample=True")
    scores = np.asarray(values, dtype=np.float32).copy() / float(temperature)
    if top_k > 0 and top_k < scores.shape[0]:
        sorted_desc = np.sort(scores)[::-1]
        threshold = float(sorted_desc[top_k - 1])
        scores[scores < threshold] = float("-inf")
    if top_p > 0 and top_p < 1:
        indexed = list(enumerate(scores.tolist()))
        indexed.sort(key=lambda item: item[1], reverse=True)
        sorted_scores = np.asarray([item[1] for item in indexed], dtype=np.float32)
        sorted_probs = _softmax(sorted_scores)
        remove_mask = [False] * len(indexed)
        cumulative = 0.0
        for index, probability in enumerate(sorted_probs):
            cumulative += float(probability)
            if cumulative > float(top_p):
                remove_mask[index] = True
        for index in range(len(remove_mask) - 1, 0, -1):
            remove_mask[index] = remove_mask[index - 1]
        if remove_mask:
            remove_mask[0] = False
        for index, should_remove in enumerate(remove_mask):
            if should_remove:
                scores[indexed[index][0]] = float("-inf")
    probabilities = _softmax(scores)
    random_value = float(rng.random())
    for index, probability in enumerate(probabilities):
        random_value -= float(probability)
        if random_value <= 0:
            return int(index)
    return _argmax(scores)


def _sample_assistant_text_token(
    text_logits: np.ndarray,
    manifest: dict[str, Any],
    generation_defaults: dict[str, Any],
    rng: np.random.Generator,
) -> int:
    candidate_ids = np.asarray(
        [
            int(manifest["tts_config"]["audio_assistant_slot_token_id"]),
            int(manifest["tts_config"]["audio_end_token_id"]),
        ],
        dtype=np.int32,
    )
    candidate_scores = text_logits[candidate_ids]
    sampled_index = _sample_from_scores(
        candidate_scores,
        do_sample=bool(generation_defaults["do_sample"]),
        temperature=float(generation_defaults["text_temperature"]),
        top_k=min(int(generation_defaults["text_top_k"]), int(candidate_scores.shape[0])),
        top_p=float(generation_defaults["text_top_p"]),
        rng=rng,
    )
    return int(candidate_ids[sampled_index])


def _sample_audio_token(
    audio_logits: np.ndarray,
    previous_token_ids: list[int],
    previous_token_set: set[int],
    generation_defaults: dict[str, Any],
    rng: np.random.Generator,
) -> int:
    repetition_penalty = float(generation_defaults["audio_repetition_penalty"])
    if not bool(generation_defaults["do_sample"]):
        return _argmax_with_repetition_penalty(audio_logits, previous_token_set, repetition_penalty)
    penalized_scores = _apply_repetition_penalty(audio_logits, previous_token_ids, repetition_penalty)
    return _sample_from_scores(
        penalized_scores,
        do_sample=True,
        temperature=float(generation_defaults["audio_temperature"]),
        top_k=int(generation_defaults["audio_top_k"]),
        top_p=float(generation_defaults["audio_top_p"]),
        rng=rng,
    )


def _normalize_sample_mode(raw_sample_mode: str | None, raw_do_sample: bool = True) -> str:
    normalized = str(raw_sample_mode or "").strip()
    if normalized in {SAMPLE_MODE_GREEDY, SAMPLE_MODE_FIXED, SAMPLE_MODE_FULL}:
        return normalized
    if normalized == "mixed3":
        return SAMPLE_MODE_FIXED if raw_do_sample else SAMPLE_MODE_GREEDY
    return SAMPLE_MODE_GREEDY if not raw_do_sample else SAMPLE_MODE_FIXED


def _compute_stream_lead_seconds(emitted_samples_total: int, sample_rate: int, first_audio_emitted_at_seconds: float | None) -> float:
    if not first_audio_emitted_at_seconds or sample_rate <= 0:
        return 0.0
    elapsed_seconds = max(0.0, time.perf_counter() - first_audio_emitted_at_seconds)
    emitted_seconds = emitted_samples_total / float(sample_rate)
    return emitted_seconds - elapsed_seconds


def _resolve_stream_decode_frame_budget(
    emitted_samples_total: int,
    sample_rate: int,
    first_audio_emitted_at_seconds: float | None,
) -> int:
    lead_seconds = _compute_stream_lead_seconds(emitted_samples_total, sample_rate, first_audio_emitted_at_seconds)
    if not first_audio_emitted_at_seconds or lead_seconds < 0.20:
        return 1
    if lead_seconds < 0.55:
        return 2
    if lead_seconds < 1.10:
        return 4
    return 8


@dataclass
class CodecStreamingDecodeSession:
    codec_meta: dict[str, Any]
    session: ort.InferenceSession

    def __post_init__(self) -> None:
        self.transformer_specs = list(self.codec_meta.get("streaming_decode", {}).get("transformer_offsets", []))
        self.attention_specs = list(self.codec_meta.get("streaming_decode", {}).get("attention_caches", []))
        self.state_feeds: dict[str, np.ndarray] = {}
        self.reset()

    def reset(self) -> None:
        self.state_feeds = {}
        for spec in self.transformer_specs:
            self.state_feeds[str(spec["input_name"])] = np.zeros(tuple(spec["shape"]), dtype=np.int32)
        for spec in self.attention_specs:
            self.state_feeds[str(spec["offset_input_name"])] = np.zeros(tuple(spec["offset_shape"]), dtype=np.int32)
            self.state_feeds[str(spec["cached_keys_input_name"])] = np.zeros(tuple(spec["cache_shape"]), dtype=np.float32)
            self.state_feeds[str(spec["cached_values_input_name"])] = np.zeros(tuple(spec["cache_shape"]), dtype=np.float32)
            positions = np.full(tuple(spec["positions_shape"]), -1, dtype=np.int32)
            self.state_feeds[str(spec["cached_positions_input_name"])] = positions

    def run_frames(self, frame_rows: list[list[int]]) -> tuple[np.ndarray, int] | None:
        if not frame_rows:
            return None
        num_quantizers = int(self.codec_meta["codec_config"]["num_quantizers"])
        frame_count = len(frame_rows)
        audio_codes = np.zeros((1, frame_count, num_quantizers), dtype=np.int32)
        for frame_index, frame_row in enumerate(frame_rows):
            for channel_index in range(num_quantizers):
                audio_codes[0, frame_index, channel_index] = int(frame_row[channel_index] if channel_index < len(frame_row) else 0)
        feeds: dict[str, np.ndarray] = {
            "audio_codes": audio_codes,
            "audio_code_lengths": np.asarray([frame_count], dtype=np.int32),
        }
        feeds.update(self.state_feeds)
        outputs = _ort_run(self.session, feeds)
        output_names = [output.name for output in self.session.get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        for spec in self.transformer_specs:
            self.state_feeds[str(spec["input_name"])] = named_outputs[str(spec["output_name"])]
        for spec in self.attention_specs:
            self.state_feeds[str(spec["offset_input_name"])] = named_outputs[str(spec["offset_output_name"])]
            self.state_feeds[str(spec["cached_keys_input_name"])] = named_outputs[str(spec["cached_keys_output_name"])]
            self.state_feeds[str(spec["cached_values_input_name"])] = named_outputs[str(spec["cached_values_output_name"])]
            self.state_feeds[str(spec["cached_positions_input_name"])] = named_outputs[str(spec["cached_positions_output_name"])]
        return (
            named_outputs["audio"],
            int(named_outputs["audio_lengths"].reshape(-1)[0]),
        )


class OrtCpuRuntime:
    def __init__(
        self,
        model_dir: str | Path,
        thread_count: int = 4,
        max_new_frames: int | None = None,
        do_sample: bool | None = None,
        sample_mode: str | None = None,
        execution_provider: str = EXECUTION_PROVIDER_CPU,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.thread_count = max(1, int(thread_count))
        self.execution_provider = _normalize_execution_provider(execution_provider)
        self.ort_providers = _resolve_ort_providers(self.execution_provider)
        self.manifest_path = self._resolve_manifest_path(self.model_dir)
        self.manifest_dir = self.manifest_path.parent
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.manifest = manifest
        if max_new_frames is not None:
            self.manifest["generation_defaults"]["max_new_frames"] = int(max_new_frames)
        if do_sample is not None:
            self.manifest["generation_defaults"]["do_sample"] = bool(do_sample)
        self.manifest["generation_defaults"]["sample_mode"] = _normalize_sample_mode(
            sample_mode if sample_mode is not None else self.manifest["generation_defaults"].get("sample_mode"),
            bool(self.manifest["generation_defaults"]["do_sample"]),
        )
        self.manifest["generation_defaults"]["do_sample"] = (
            self.manifest["generation_defaults"]["sample_mode"] != SAMPLE_MODE_GREEDY
        )
        self.tts_meta_path = self.resolve_manifest_relative_path(manifest["model_files"]["tts_meta"])
        self.codec_meta_path = self.resolve_manifest_relative_path(manifest["model_files"]["codec_meta"])
        self.tts_meta = json.loads(self.tts_meta_path.read_text(encoding="utf-8"))
        self.codec_meta = json.loads(self.codec_meta_path.read_text(encoding="utf-8"))
        self.rng = np.random.default_rng(1234)
        self.sessions = self._create_sessions()
        self.codec_streaming_session = CodecStreamingDecodeSession(
            codec_meta=self.codec_meta,
            session=self.sessions["codec_decode_step"],
        )

    @staticmethod
    def _resolve_manifest_path(model_dir: Path) -> Path:
        tried_paths: list[Path] = []
        for relative_path in MANIFEST_CANDIDATE_RELATIVE_PATHS:
            candidate = (model_dir / relative_path).resolve()
            tried_paths.append(candidate)
            if candidate.is_file():
                return candidate
        joined = ", ".join(str(path_value) for path_value in tried_paths)
        raise FileNotFoundError(f"browser_poc_manifest.json not found. tried: {joined}")

    def resolve_manifest_relative_path(self, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        resolved = (self.manifest_dir / relative).resolve()
        if resolved.exists():
            return resolved
        relative_text = str(relative).replace("\\", "/")
        for legacy_name, canonical_name in MODEL_DIR_ALIAS_MAP.items():
            legacy_fragment = f"/{legacy_name}/"
            if legacy_fragment not in f"/{relative_text}/":
                continue
            rewritten_text = relative_text.replace(legacy_name, canonical_name)
            rewritten = (self.manifest_dir / Path(rewritten_text)).resolve()
            if rewritten.exists():
                return rewritten
        return resolved

    def _session(self, path_value: Path) -> ort.InferenceSession:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = self.thread_count
        options.inter_op_num_threads = 1
        # Keep large external-data constants mmap-backed instead of prepacking
        # a copy per session (three ranking instances share one JuiceFS page cache).
        # TTS_MOSS_PREPACKING=1 restores eager prepacking (standalone-test behavior).
        if os.environ.get("TTS_MOSS_PREPACKING", "0") != "1":
            options.add_session_config_entry("session.disable_prepacking", "1")
        # Free arena blocks back to the OS after each run; three instances on an
        # 8 GB Orin cannot each hold a multi-GB arena.
        options.enable_cpu_mem_arena = os.environ.get("TTS_MOSS_CPU_ARENA", "0") == "1"
        providers = self.ort_providers
        if self.execution_provider == EXECUTION_PROVIDER_CUDA:
            gpu_mb = int(os.environ.get("TTS_MOSS_GPU_MEM_LIMIT_MB", "1536"))
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "gpu_mem_limit": gpu_mb * 1024 * 1024,
                        "arena_extend_strategy": "kSameAsRequested",
                    },
                ),
                "CPUExecutionProvider",
            ]
        session = ort.InferenceSession(str(path_value), sess_options=options, providers=providers, disabled_optimizers=["GatherSliceToSplitFusion"])
        if self.execution_provider == EXECUTION_PROVIDER_CUDA and "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError(
                "CUDAExecutionProvider was requested, but ONNX Runtime created a session without CUDA support "
                f"for {path_value}. Actual providers: {session.get_providers()}"
            )
        return session

    def _create_sessions(self) -> dict[str, ort.InferenceSession]:
        # Load only the graphs the fixed-sample generation path actually uses.
        # Skipping codec_encode / local_decoder / cached-step keeps the big
        # external .data files from being duplicated across sessions.
        tts_dir = self.tts_meta_path.parent
        codec_dir = self.codec_meta_path.parent
        files = self.tts_meta["files"]
        sample_mode = self.manifest["generation_defaults"]["sample_mode"]
        sessions: dict[str, ort.InferenceSession] = {
            "prefill": self._session(tts_dir / files["prefill"]),
            "decode": self._session(tts_dir / files["decode_step"]),
        }
        if sample_mode == SAMPLE_MODE_FIXED and files.get("local_fixed_sampled_frame"):
            sessions["local_fixed_sampled_frame"] = self._session(
                tts_dir / files["local_fixed_sampled_frame"]
            )
        if sample_mode == SAMPLE_MODE_GREEDY and files.get("local_greedy_frame"):
            sessions["local_greedy_frame"] = self._session(
                tts_dir / files["local_greedy_frame"]
            )
        sessions["codec_decode"] = self._session(
            codec_dir / self.codec_meta["files"]["decode_full"]
        )
        sessions["codec_decode_step"] = self._session(
            codec_dir / self.codec_meta["files"]["decode_step"]
        )
        return sessions

    def list_builtin_voices(self) -> list[dict[str, Any]]:
        return list(self.manifest["builtin_voices"])

    def list_text_samples(self) -> list[dict[str, Any]]:
        return list(self.manifest["text_samples"])

    def warmup(self) -> None:
        voice = self.list_builtin_voices()[0]
        text_sample = self.list_text_samples()[0]
        request_rows = self.build_voice_clone_request_rows(voice["prompt_audio_codes"], text_sample["text_token_ids"])
        prefill_ids, prefill_dims = _flatten3d_int32([request_rows["inputIds"]])
        prefill_mask, prefill_mask_dims = _flatten2d_int32(request_rows["attentionMask"])
        outputs = _ort_run(self.sessions["prefill"], {
                "input_ids": prefill_ids.reshape(prefill_dims),
                "attention_mask": prefill_mask.reshape(prefill_mask_dims),
            },
        )
        output_names = [output.name for output in self.sessions["prefill"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        global_hidden = _extract_last_hidden(named_outputs["global_hidden"])
        if "local_cached_step" in self.sessions:
            local_past_by_name = self.create_empty_local_cached_past()
            _text_logits, _audio_logits, _next_local_past = self.run_local_cached_step(
                global_hidden,
                text_token_id=0,
                audio_token_id=0,
                channel_index=0,
                step_type=0,
                past_valid_lengths=0,
                local_past_by_name=local_past_by_name,
            )
        if "local_fixed_sampled_frame" in self.sessions and self.manifest["generation_defaults"]["sample_mode"] == SAMPLE_MODE_FIXED:
            self.run_local_fixed_sampled_frame(
                global_hidden,
                previous_token_sets_by_channel=[set() for _ in range(int(self.manifest["tts_config"]["n_vq"]))],
            )
        elif "local_greedy_frame" in self.sessions and not bool(self.manifest["generation_defaults"]["do_sample"]):
            self.run_local_greedy_frame(
                global_hidden,
                previous_token_sets_by_channel=[set() for _ in range(int(self.manifest["tts_config"]["n_vq"]))],
                repetition_penalty=float(self.manifest["generation_defaults"]["audio_repetition_penalty"]),
            )
        else:
            self.run_local_decoder(global_hidden, self.manifest["tts_config"]["audio_assistant_slot_token_id"], [])
        empty_frames = [([0] * int(self.manifest["tts_config"]["n_vq"]))]
        self.decode_full_audio(empty_frames)
        self.codec_streaming_session.reset()
        self.codec_streaming_session.run_frames(empty_frames)
        self.codec_streaming_session.reset()

    def build_text_rows(self, token_ids: list[int]) -> list[list[int]]:
        rows: list[list[int]] = []
        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1
        for token_id in token_ids:
            row = [int(self.manifest["tts_config"]["audio_pad_token_id"])] * row_width
            row[0] = int(token_id)
            rows.append(row)
        return rows

    def build_audio_prefix_rows(self, prompt_audio_codes: list[list[int]], slot_token_id: int | None = None) -> list[list[int]]:
        rows: list[list[int]] = []
        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1
        resolved_slot_token_id = int(
            self.manifest["tts_config"]["audio_user_slot_token_id"] if slot_token_id is None else slot_token_id
        )
        for code_row in prompt_audio_codes:
            row = [int(self.manifest["tts_config"]["audio_pad_token_id"])] * row_width
            row[0] = resolved_slot_token_id
            for index in range(min(len(code_row), int(self.manifest["tts_config"]["n_vq"]))):
                row[index + 1] = int(code_row[index])
            rows.append(row)
        return rows

    def build_voice_clone_request_rows(self, prompt_audio_codes: list[list[int]], text_token_ids: list[int]) -> dict[str, list[list[int]]]:
        prefix_text_token_ids = [
            *self.manifest["prompt_templates"]["user_prompt_prefix_token_ids"],
            int(self.manifest["tts_config"]["audio_start_token_id"]),
        ]
        suffix_text_token_ids = [
            int(self.manifest["tts_config"]["audio_end_token_id"]),
            *self.manifest["prompt_templates"]["user_prompt_after_reference_token_ids"],
            *text_token_ids,
            *self.manifest["prompt_templates"]["assistant_prompt_prefix_token_ids"],
            int(self.manifest["tts_config"]["audio_start_token_id"]),
        ]
        rows = [
            *self.build_text_rows(prefix_text_token_ids),
            *self.build_audio_prefix_rows(prompt_audio_codes),
            *self.build_text_rows(suffix_text_token_ids),
        ]
        return {
            "inputIds": rows,
            "attentionMask": [[1 for _ in rows]],
        }

    def run_local_decoder(self, global_hidden: np.ndarray, text_token_id: int, frame_prefix: list[int]) -> tuple[np.ndarray, np.ndarray]:
        n_vq = int(self.manifest["tts_config"]["n_vq"])
        audio_pad = int(self.manifest["tts_config"]["audio_pad_token_id"])
        padded_prefix = np.full((1, n_vq - 1), audio_pad, dtype=np.int32)
        for index in range(min(len(frame_prefix), n_vq - 1)):
            padded_prefix[0, index] = int(frame_prefix[index])
        outputs = _ort_run(self.sessions["local_decoder"], {
                "global_hidden": global_hidden.astype(np.float32, copy=False),
                "text_token_id": np.asarray([int(text_token_id)], dtype=np.int32),
                "audio_prefix_token_ids": padded_prefix,
            },
        )
        output_names = [output.name for output in self.sessions["local_decoder"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        return named_outputs["text_logits"].reshape(-1), named_outputs["audio_logits"]

    def create_empty_local_cached_past(self) -> dict[str, np.ndarray]:
        local_layers = int(self.tts_meta["model_config"]["local_layers"])
        local_heads = int(self.tts_meta["model_config"]["local_heads"])
        local_head_dim = int(self.tts_meta["model_config"]["local_head_dim"])
        return {
            name: np.zeros((1, 0, local_heads, local_head_dim), dtype=np.float32)
            for layer_index in range(local_layers)
            for name in (f"local_past_key_{layer_index}", f"local_past_value_{layer_index}")
        }

    def run_local_cached_step(
        self,
        global_hidden: np.ndarray,
        *,
        text_token_id: int,
        audio_token_id: int,
        channel_index: int,
        step_type: int,
        past_valid_lengths: int,
        local_past_by_name: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        outputs = _ort_run(self.sessions["local_cached_step"], {
                "global_hidden": global_hidden.astype(np.float32, copy=False),
                "text_token_id": np.asarray([int(text_token_id)], dtype=np.int32),
                "audio_token_id": np.asarray([int(audio_token_id)], dtype=np.int32),
                "channel_index": np.asarray([int(channel_index)], dtype=np.int32),
                "step_type": np.asarray([int(step_type)], dtype=np.int32),
                "past_valid_lengths": np.asarray([int(past_valid_lengths)], dtype=np.int32),
                **local_past_by_name,
            },
        )
        output_names = [output.name for output in self.sessions["local_cached_step"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        next_local_past = {
            output_name.replace("local_present_", "local_past_"): named_outputs[output_name]
            for output_name in self.tts_meta["onnx"]["local_cached_output_names"][2:]
        }
        return named_outputs["text_logits"].reshape(-1), named_outputs["audio_logits"], next_local_past

    def run_local_greedy_frame(
        self,
        global_hidden: np.ndarray,
        *,
        previous_token_sets_by_channel: list[set[int]],
        repetition_penalty: float,
    ) -> tuple[bool, list[int]]:
        audio_codebook_size = int(self.tts_meta["model_config"]["audio_codebook_sizes"][0])
        n_vq = int(self.manifest["tts_config"]["n_vq"])
        repetition_seen_mask = np.zeros((1, n_vq, audio_codebook_size), dtype=np.int32)
        for channel_index, token_ids in enumerate(previous_token_sets_by_channel):
            for token_id in token_ids:
                if 0 <= token_id < audio_codebook_size:
                    repetition_seen_mask[0, channel_index, token_id] = 1
        outputs = _ort_run(self.sessions["local_greedy_frame"], {
                "global_hidden": global_hidden.astype(np.float32, copy=False),
                "repetition_seen_mask": repetition_seen_mask,
                "repetition_penalty": np.asarray([float(repetition_penalty)], dtype=np.float32),
            },
        )
        output_names = [output.name for output in self.sessions["local_greedy_frame"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        should_continue = bool(int(np.asarray(named_outputs["should_continue"]).reshape(-1)[0]))
        frame_token_ids = np.asarray(named_outputs["frame_token_ids"]).reshape(-1).astype(np.int32, copy=False).tolist()
        return should_continue, [int(item) for item in frame_token_ids]

    def run_local_fixed_sampled_frame(
        self,
        global_hidden: np.ndarray,
        *,
        previous_token_sets_by_channel: list[set[int]],
    ) -> tuple[bool, list[int]]:
        audio_codebook_size = int(self.tts_meta["model_config"]["audio_codebook_sizes"][0])
        n_vq = int(self.manifest["tts_config"]["n_vq"])
        repetition_seen_mask = np.zeros((1, n_vq, audio_codebook_size), dtype=np.int32)
        for channel_index, token_ids in enumerate(previous_token_sets_by_channel):
            for token_id in token_ids:
                if 0 <= token_id < audio_codebook_size:
                    repetition_seen_mask[0, channel_index, token_id] = 1
        assistant_random_u = np.asarray([min(0.99999994, max(0.0, float(self.rng.random())))], dtype=np.float32)
        audio_random_u = np.asarray(
            [[min(0.99999994, max(0.0, float(self.rng.random()))) for _ in range(n_vq)]],
            dtype=np.float32,
        )
        outputs = _ort_run(self.sessions["local_fixed_sampled_frame"], {
                "global_hidden": global_hidden.astype(np.float32, copy=False),
                "repetition_seen_mask": repetition_seen_mask,
                "assistant_random_u": assistant_random_u,
                "audio_random_u": audio_random_u,
            },
        )
        output_names = [output.name for output in self.sessions["local_fixed_sampled_frame"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        frame_token_ids = np.asarray(named_outputs["frame_token_ids"]).reshape(-1).astype(np.int32, copy=False).tolist()
        should_continue = bool(int(np.asarray(named_outputs["should_continue"]).reshape(-1)[0]))
        return should_continue, [int(item) for item in frame_token_ids]

    def slice_audio_channel_logits(self, audio_logits: np.ndarray, channel_index: int) -> np.ndarray:
        per_channel = int(audio_logits.shape[-1])
        flat = audio_logits.reshape(-1)
        start = channel_index * per_channel
        end = start + per_channel
        return flat[start:end]

    def decode_full_audio(self, generated_frames: list[list[int]]) -> tuple[list[np.ndarray], int]:
        if not generated_frames:
            return [], 0
        audio_codes, dims = _flatten3d_int32([generated_frames])
        outputs = _ort_run(self.sessions["codec_decode"], {
                "audio_codes": audio_codes.reshape(dims),
                "audio_code_lengths": np.asarray([len(generated_frames)], dtype=np.int32),
            },
        )
        output_names = [output.name for output in self.sessions["codec_decode"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        audio_length = int(named_outputs["audio_lengths"].reshape(-1)[0])
        return _slice_channel_major_audio(named_outputs["audio"], 0, audio_length), audio_length

    def generate_audio_frames(
        self,
        request_rows: dict[str, list[list[int]]],
        on_frame: Callable[[list[list[int]], int, list[int]], None] | None = None,
    ) -> list[list[int]]:
        generation_defaults = self.manifest["generation_defaults"]
        row_width = int(self.manifest["tts_config"]["n_vq"]) + 1
        prefill_ids, prefill_dims = _flatten3d_int32([request_rows["inputIds"]])
        prefill_mask, prefill_mask_dims = _flatten2d_int32(request_rows["attentionMask"])
        outputs = _ort_run(self.sessions["prefill"], {
                "input_ids": prefill_ids.reshape(prefill_dims),
                "attention_mask": prefill_mask.reshape(prefill_mask_dims),
            },
        )
        output_names = [output.name for output in self.sessions["prefill"].get_outputs()]
        named_outputs = dict(zip(output_names, outputs, strict=True))
        global_hidden = _extract_last_hidden(named_outputs["global_hidden"])
        past_valid_length = sum(int(item) for item in request_rows["attentionMask"][0])
        past_by_name = {
            output_name.replace("present_", "past_"): named_outputs[output_name]
            for output_name in self.tts_meta["onnx"]["prefill_output_names"][1:]
        }
        generated_frames: list[list[int]] = []
        previous_tokens_by_channel = [[] for _ in range(int(self.manifest["tts_config"]["n_vq"]))]
        previous_token_sets_by_channel = [set() for _ in range(int(self.manifest["tts_config"]["n_vq"]))]

        for step_index in range(int(generation_defaults["max_new_frames"])):
            frame: list[int] = []
            if "local_greedy_frame" in self.sessions and not bool(generation_defaults["do_sample"]):
                should_continue, frame = self.run_local_greedy_frame(
                    global_hidden,
                    previous_token_sets_by_channel=previous_token_sets_by_channel,
                    repetition_penalty=float(generation_defaults["audio_repetition_penalty"]),
                )
                if not should_continue:
                    break
                for channel_index, sampled_token in enumerate(frame):
                    previous_tokens_by_channel[channel_index].append(sampled_token)
                    previous_token_sets_by_channel[channel_index].add(sampled_token)
            elif "local_fixed_sampled_frame" in self.sessions and generation_defaults["sample_mode"] == SAMPLE_MODE_FIXED:
                should_continue, frame = self.run_local_fixed_sampled_frame(
                    global_hidden,
                    previous_token_sets_by_channel=previous_token_sets_by_channel,
                )
                if not should_continue:
                    break
                for channel_index, sampled_token in enumerate(frame):
                    previous_tokens_by_channel[channel_index].append(sampled_token)
                    previous_token_sets_by_channel[channel_index].add(sampled_token)
            elif "local_cached_step" in self.sessions:
                local_past_by_name = self.create_empty_local_cached_past()
                local_past_valid_length = 0
                local_text_logits, _ignored_audio_logits, local_past_by_name = self.run_local_cached_step(
                    global_hidden,
                    text_token_id=0,
                    audio_token_id=0,
                    channel_index=0,
                    step_type=0,
                    past_valid_lengths=local_past_valid_length,
                    local_past_by_name=local_past_by_name,
                )
                local_past_valid_length += 1
                next_text_token = _sample_assistant_text_token(
                    local_text_logits,
                    self.manifest,
                    generation_defaults,
                    self.rng,
                )
                if next_text_token != int(self.manifest["tts_config"]["audio_assistant_slot_token_id"]):
                    break
                _unused_text_logits, audio_logits, local_past_by_name = self.run_local_cached_step(
                    global_hidden,
                    text_token_id=next_text_token,
                    audio_token_id=0,
                    channel_index=0,
                    step_type=1,
                    past_valid_lengths=local_past_valid_length,
                    local_past_by_name=local_past_by_name,
                )
                local_past_valid_length += 1
                first_channel_logits = self.slice_audio_channel_logits(audio_logits, 0).astype(np.float32, copy=False)
                sampled_token = _sample_audio_token(
                    first_channel_logits,
                    previous_tokens_by_channel[0],
                    previous_token_sets_by_channel[0],
                    generation_defaults,
                    self.rng,
                )
                frame.append(sampled_token)
                previous_tokens_by_channel[0].append(sampled_token)
                previous_token_sets_by_channel[0].add(sampled_token)

                previous_token = sampled_token
                host_sampled_channel_limit = int(self.manifest["tts_config"]["n_vq"])
                for channel_index in range(1, host_sampled_channel_limit):
                    _unused_text_logits, audio_logits, local_past_by_name = self.run_local_cached_step(
                        global_hidden,
                        text_token_id=0,
                        audio_token_id=previous_token,
                        channel_index=channel_index - 1,
                        step_type=2,
                        past_valid_lengths=local_past_valid_length,
                        local_past_by_name=local_past_by_name,
                    )
                    local_past_valid_length += 1
                    channel_logits = self.slice_audio_channel_logits(audio_logits, channel_index).astype(np.float32, copy=False)
                    sampled_token = _sample_audio_token(
                        channel_logits,
                        previous_tokens_by_channel[channel_index],
                        previous_token_sets_by_channel[channel_index],
                        generation_defaults,
                        self.rng,
                    )
                    frame.append(sampled_token)
                    previous_tokens_by_channel[channel_index].append(sampled_token)
                    previous_token_sets_by_channel[channel_index].add(sampled_token)
                    previous_token = sampled_token
            else:
                local_text_logits, _ = self.run_local_decoder(global_hidden, 0, [])
                next_text_token = _sample_assistant_text_token(
                    local_text_logits,
                    self.manifest,
                    generation_defaults,
                    self.rng,
                )
                if next_text_token != int(self.manifest["tts_config"]["audio_assistant_slot_token_id"]):
                    break
                for channel_index in range(int(self.manifest["tts_config"]["n_vq"])):
                    _, audio_logits = self.run_local_decoder(global_hidden, next_text_token, frame)
                    channel_logits = self.slice_audio_channel_logits(audio_logits, channel_index).astype(np.float32, copy=False)
                    sampled_token = _sample_audio_token(
                        channel_logits,
                        previous_tokens_by_channel[channel_index],
                        previous_token_sets_by_channel[channel_index],
                        generation_defaults,
                        self.rng,
                    )
                    frame.append(sampled_token)
                    previous_tokens_by_channel[channel_index].append(sampled_token)
                    previous_token_sets_by_channel[channel_index].add(sampled_token)
            generated_frames.append(frame)

            next_row = np.full((1, 1, row_width), int(self.manifest["tts_config"]["audio_pad_token_id"]), dtype=np.int32)
            next_row[0, 0, 0] = int(self.manifest["tts_config"]["audio_assistant_slot_token_id"])
            for index, token in enumerate(frame):
                next_row[0, 0, index + 1] = int(token)
            decode_feeds: dict[str, np.ndarray] = {
                "input_ids": next_row,
                "past_valid_lengths": np.asarray([past_valid_length], dtype=np.int32),
            }
            for input_name in self.tts_meta["onnx"]["decode_input_names"][2:]:
                decode_feeds[input_name] = past_by_name[input_name]
            decode_outputs = _ort_run(self.sessions["decode"], decode_feeds)
            decode_output_names = [output.name for output in self.sessions["decode"].get_outputs()]
            named_decode_outputs = dict(zip(decode_output_names, decode_outputs, strict=True))
            global_hidden = _extract_last_hidden(named_decode_outputs["global_hidden"])
            past_valid_length += 1
            past_by_name = {
                output_name.replace("present_", "past_"): named_decode_outputs[output_name]
                for output_name in self.tts_meta["onnx"]["decode_output_names"][1:]
            }
            if on_frame is not None:
                on_frame(generated_frames, step_index, frame)
        return generated_frames


# ── phanthymotus wrapper ─────────────────────────────────────────────────────

def _detect_language(text: str) -> str:
    """Coarse script/language classification for voice selection."""
    for ch in text or "":
        code = ord(ch)
        if 0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF:
            return "ja"
        if 0x4E00 <= code <= 0x9FFF:
            return "zh"
        if 0x0E00 <= code <= 0x0E7F:
            return "th"
        if 0x0400 <= code <= 0x04FF:
            return "ru"
        if 0x0590 <= code <= 0x05FF:
            return "he"
        if 0x0600 <= code <= 0x06FF:
            return "ar"
        if 0x0370 <= code <= 0x03FF:
            return "el"
        if 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:
            return "ko"
    lowered = (text or "").lower()
    if any(c in lowered for c in "ñ¿áéíóúü"):
        return "es"
    if any(c in lowered for c in "àâçèêëîïôùûœ"):
        return "fr"
    if any(c in lowered for c in "äöüß"):
        return "de"
    if any(c in lowered for c in "ãõ"):
        return "pt"
    return "en"


class MossOrtTTS:
    """MOSS-TTS-Nano ONNX TTS: fixed per-language voice, 16 kHz mono PCM16 out.

    model_dir layout (matches the JuiceFS tar):
      {model_dir}/MOSS-TTS-Nano-100M-ONNX/
      {model_dir}/MOSS-Audio-Tokenizer-Nano-ONNX/
    """

    def __init__(
        self,
        model_dir: str,
        num_threads: int = 4,
        voices: dict | None = None,
        sample_mode: str | None = None,
        do_sample: bool | None = None,
        max_new_frames: int | None = None,
        execution_provider: str = "cpu",
    ):
        import os
        import threading

        self._model_dir = model_dir
        self._lock = threading.Lock()
        self._voices_cfg = dict(voices or {"default": "saki"})
        self._runtime = OrtCpuRuntime(
            model_dir,
            thread_count=num_threads,
            max_new_frames=max_new_frames,
            do_sample=do_sample,
            sample_mode=sample_mode,
            execution_provider=execution_provider,
        )
        codec_sr = int(self._runtime.codec_meta["codec_config"]["sample_rate"])
        self._model_sr = codec_sr  # 48000 stereo from the codec
        self._out_sr = 16000

        import sentencepiece as spm

        tts_dir = self._runtime.tts_meta_path.parent
        tok_path = str(tts_dir / "tokenizer.model")
        if not os.path.isfile(tok_path):
            raise FileNotFoundError(tok_path)
        self._sp = spm.SentencePieceProcessor(model_file=tok_path)

        self._voices = list(self._runtime.list_builtin_voices())
        self._default_voice = self._resolve_voice(self._voices_cfg.get("default", "saki"))
        self._lang_voice_cache: dict[str, dict] = {}

    def _resolve_voice(self, spec) -> dict:
        if isinstance(spec, int):
            return self._voices[spec]
        name = str(spec or "").strip().lower()
        for voice in self._voices:
            fields = (
                str(voice.get("display_name", "")).lower(),
                str(voice.get("voice", "")).lower(),
            )
            if name and any(name in f for f in fields):
                return voice
        raise ValueError("MOSS voice not found: %r (builtin: %s)" % (
            spec, [v.get("display_name") for v in self._voices]))

    def voice_for_text(self, text: str) -> dict:
        lang = _detect_language(text)
        cached = self._lang_voice_cache.get(lang)
        if cached is None:
            spec = self._voices_cfg.get(lang, self._voices_cfg.get("default", "saki"))
            cached = self._resolve_voice(spec)
            self._lang_voice_cache[lang] = cached
        return cached

    def _prepare(self, text: str):
        """Shared setup: punctuation, language, duration cap, token ids."""
        text = (text or "").strip()
        if not text:
            return None
        if text[-1] not in "。！？!?.…":
            lang = _detect_language(text)
            text += "。" if lang in ("zh", "ja") else "."
        else:
            lang = _detect_language(text)
        cap_s = _duration_cap_seconds(text, lang)
        cap_frames = max(24, int(cap_s * 12.5))
        token_ids = self._sp.encode(text)
        if not token_ids:
            return None
        return text, cap_s, cap_frames, token_ids

    def synthesize_stream(self, text: str):
        """Streaming synthesis: yields 16 kHz mono float32 chunks (~0.08 s each)
        as audio frames are generated and codec-decoded. First chunk arrives
        after prefill + one frame (~1 s), satisfying the board's TTFT gate.
        """
        import queue as _queue
        import threading

        import numpy as np

        prepared = self._prepare(text)
        if prepared is None:
            return
        text, _cap_s, cap_frames, token_ids = prepared

        pcm_q: _queue.Queue = _queue.Queue(maxsize=256)

        def _generate() -> None:
            with self._lock:
                session = self._runtime.codec_streaming_session
                session.reset()
                # Prime the streaming codec with one zero frame: its attention
                # caches start zeroed and the first real frame would otherwise
                # come out as a loud garbage burst before the speech.
                n_vq = int(self._runtime.codec_meta["codec_config"]["num_quantizers"])
                session.run_frames([[0] * n_vq])

                def _on_frame(frames, step_index, frame):
                    out = session.run_frames([frame])
                    if out is not None:
                        pcm_q.put(out)

                defaults = self._runtime.manifest["generation_defaults"]
                saved_cap = defaults.get("max_new_frames")
                defaults["max_new_frames"] = min(
                    int(saved_cap) if saved_cap else 375, cap_frames
                )
                try:
                    voice = self.voice_for_text(text)
                    rows = self._runtime.build_voice_clone_request_rows(
                        voice["prompt_audio_codes"], token_ids
                    )
                    self._runtime.generate_audio_frames(rows, on_frame=_on_frame)
                except Exception as exc:  # surfaced to the consumer
                    pcm_q.put(exc)
                finally:
                    defaults["max_new_frames"] = saved_cap
                    pcm_q.put(None)

        worker = threading.Thread(target=_generate, daemon=True)
        worker.start()
        while True:
            item = pcm_q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                worker.join(timeout=1)
                raise item
            audio, length = item
            if length <= 0:
                continue
            arr = np.asarray(audio, dtype=np.float32)
            if arr.ndim == 3:  # (1, channels, samples) channel-major
                mono = arr[0].mean(axis=0)[:length]
            elif arr.ndim == 2 and arr.shape[0] <= 4:
                mono = arr.mean(axis=0)[:length]
            else:
                mono = arr.reshape(-1)[:length]
            yield _resample_to_16k(mono, self._model_sr, self._out_sr)
        worker.join(timeout=5)

    def synthesize(self, text: str) -> bytes:
        """Buffered synthesis with silence trims (used by /tts/test)."""
        import numpy as np

        prepared = self._prepare(text)
        if prepared is None:
            return b""
        _text, cap_s, _cap_frames, _ids = prepared
        pcm = np.concatenate(list(self.synthesize_stream(text)))
        if pcm.size == 0:
            return b""
        pcm = _trim_silence(pcm, self._out_sr)
        pcm = _trim_leading_blips(pcm, self._out_sr)
        pcm = _cap_duration(pcm, self._out_sr, cap_s)
        return _float_to_pcm16(pcm)


def _resample_to_16k(samples, src_rate: int, dst_rate: int = 16000):
    """Windowed-sinc lowpass + integer decimation (no scipy)."""
    import numpy as np

    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    src_rate = int(src_rate)
    dst_rate = int(dst_rate)
    if src_rate == dst_rate:
        return x
    if src_rate % dst_rate != 0:
        raise ValueError("MOSS resample only supports integer ratios, got %s -> %s" % (src_rate, dst_rate))
    factor = src_rate // dst_rate
    cutoff = (dst_rate / 2.0 * 0.92) / src_rate  # slightly below output Nyquist
    num_taps = 33
    t = np.arange(num_taps, dtype=np.float64) - (num_taps - 1) / 2.0
    kernel = 2.0 * cutoff * np.sinc(2.0 * cutoff * t)
    kernel *= np.hamming(num_taps)
    kernel = kernel / kernel.sum()
    filtered = np.convolve(x.astype(np.float64), kernel, mode="same")
    return filtered[::factor].astype(np.float32)


def _duration_cap_seconds(text: str, lang: str) -> float:
    """~1.8x the natural speaking duration for the script (12.5 Hz frames)."""
    chars = sum(1 for ch in text if ch.isalnum())
    rate = 0.40 if lang in ("zh", "ja", "th", "ko") else 0.25  # seconds/char
    return min(20.0, max(3.0, chars * rate))


def _trim_leading_blips(samples, sample_rate: int, gap_s: float = 0.30,
                        window_s: float = 1.5, max_iter: int = 3,
                        threshold_db: float = -40.0):
    """Drop leading garbage clusters: after the model consumes the voice-prompt
    codes, the first frames can be a low-pitched vocalization / noise blip
    before the real sentence. If a >=gap_s silence gap exists inside the
    first `window_s`, everything before the gap end is preamble — cut it and
    re-check (up to max_iter). Audio that starts with sustained speech
    (zh/ja/de) has no early gap and passes through untouched."""
    import numpy as np

    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    win = max(1, int(sample_rate * 0.05))
    gap_wins = max(1, int(gap_s / 0.05))
    for _ in range(max_iter):
        n_win = x.size // win
        limit = min(n_win, int(window_s / 0.05))
        if limit < gap_wins + 1:
            break
        rms = np.sqrt(np.mean(x[: n_win * win].reshape(n_win, win) ** 2, axis=1))
        db = 20.0 * np.log10(np.maximum(rms, 1e-6))
        run = 0
        cut = None
        for i in range(limit):
            if db[i] <= threshold_db:
                run += 1
                if run >= gap_wins:
                    cut = (i + 1) * win
                    break
            else:
                run = 0
        if cut is None:
            break
        x = x[cut:]
    return x


def _cap_duration(samples, sample_rate: int, cap_s: float,
                  gap_s: float = 0.35):
    """If audio exceeds cap_s, cut at the last >=gap_s silence gap inside the
    last 60% below the cap; hard-cut at cap_s when no gap exists."""
    import numpy as np

    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size <= cap_s * sample_rate:
        return x
    win = max(1, int(sample_rate * 0.05))
    n_win = x.size // win
    rms = np.sqrt(np.mean(x[: n_win * win].reshape(n_win, win) ** 2, axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-6))
    cap_win = int(cap_s * sample_rate / win)
    floor_win = int(gap_s * sample_rate / win)
    # 最后一个低于阈值的窗口位置（从 cap 往前找 >=gap 的连续静音）
    run = 0
    cut_win = None
    for i in range(min(cap_win, n_win) - 1, int(n_win * 0.4), -1):
        if db[i] <= -45.0:
            run += 1
            if run >= floor_win:
                cut_win = i + run - 1
                break
        else:
            run = 0
    if cut_win is None:
        cut_win = cap_win
    end = min(x.size, (cut_win + 1) * win)
    return x[:end]


def _trim_silence(samples, sample_rate: int, threshold_db: float = -45.0,
                  max_lead_s: float = 0.30, tail_s: float = 0.20):
    """Trim codec lead/trail silence. Keep at most `max_lead_s` of leading
    silence and `tail_s` of padding after the last voiced window."""
    import numpy as np

    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    win = max(1, int(sample_rate * 0.05))
    n_win = x.size // win
    if n_win == 0:
        return x
    rms = np.sqrt(np.mean(x[: n_win * win].reshape(n_win, win) ** 2, axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-6))
    voiced = np.nonzero(db > threshold_db)[0]
    if voiced.size == 0:
        return x[: int(sample_rate * 0.5)]
    first = voiced[0]
    last = voiced[-1]
    start = 0 if first * win <= max_lead_s * sample_rate else max(0, first * win - int(0.05 * sample_rate))
    end = min(x.size, (last + 1) * win + int(tail_s * sample_rate))
    return x[start:end]


def _float_to_pcm16(samples) -> bytes:
    import numpy as np

    x = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()
