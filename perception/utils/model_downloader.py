"""
utils/model_downloader.py — Fetch Gentleman Matcha pack from JuiceFS.

Always reinstall from the current JuiceFS/HTTP tar. Do not reuse a stale
/models tree (old TN / frontend / onnx) just because check_file exists.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tarfile
import tempfile
import zipfile
from urllib.request import urlretrieve

log = logging.getLogger(__name__)

JUICEFS_BASE = "http://172.28.4.81:34567/fanyi/phanthymotus_tts"
# Same JuiceFS data disk the tar lives on. Prefer this over HTTP when present.
JUICEFS_LOCAL = os.environ.get("TTS_JUICEFS_DIR", "/mnt/data/fanyi/phanthymotus_tts")


def _progress_hook(name: str):
    """Create a reporthook for urlretrieve that logs download progress."""
    last_pct = [0]

    def hook(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(int(block_num * block_size * 100 / total_size), 100)
            if pct >= last_pct[0] + 10:
                last_pct[0] = pct
                mb_done = block_num * block_size / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                log.info(
                    f"[model_downloader] {name}: {pct}% ({mb_done:.1f}/{mb_total:.1f} MB)"
                )

    return hook


MODELS = {
    "tts_moss_nano": {
        # Multilingual board (zh/ja/ru/en/es/fr/th/de). ONNX TTS + codec pack.
        "url": "http://172.28.4.81:34567/zhaoyiru/phanthymotus/tts/moss-tts-nano-16k.tar.bz2",
        "check_file": "MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json",
        "required_files": (
            "MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json",
            "MOSS-TTS-Nano-100M-ONNX/moss_tts_prefill.onnx",
            "MOSS-TTS-Nano-100M-ONNX/moss_tts_decode_step.onnx",
            "MOSS-TTS-Nano-100M-ONNX/moss_tts_local_decoder.onnx",
            "MOSS-TTS-Nano-100M-ONNX/moss_tts_local_cached_step.onnx",
            "MOSS-TTS-Nano-100M-ONNX/moss_tts_local_fixed_sampled_frame.onnx",
            "MOSS-TTS-Nano-100M-ONNX/moss_tts_global_shared.data",
            "MOSS-TTS-Nano-100M-ONNX/moss_tts_local_shared.data",
            "MOSS-TTS-Nano-100M-ONNX/tokenizer.model",
            "MOSS-Audio-Tokenizer-Nano-ONNX/codec_browser_onnx_meta.json",
            "MOSS-Audio-Tokenizer-Nano-ONNX/moss_audio_tokenizer_encode.onnx",
            "MOSS-Audio-Tokenizer-Nano-ONNX/moss_audio_tokenizer_decode_full.onnx",
            "MOSS-Audio-Tokenizer-Nano-ONNX/moss_audio_tokenizer_decode_step.onnx",
            "MOSS-Audio-Tokenizer-Nano-ONNX/moss_audio_tokenizer_decode_shared.data",
        ),
    },
    "tts_matcha_gentleman": {
        "url": f"{JUICEFS_BASE}/matcha-gentleman-phonetone-16k.tar.bz2",
        "check_file": "model-steps-3.onnx",
        # Ranking uses 3-step. Do not extract the 10-step graph into page cache.
        "skip_files": ("model-steps-10.onnx",),
        # Must match a fresh local JuiceFS pack after every install.
        "required_files": (
            "model-steps-3.onnx",
            "gentleman-vocos.onnx",
            "frontend_release/opencpop-strict.txt",
            "frontend_release/tn_cache/zh_tn_tagger.fst",
            "frontend_release/tn_cache/zh_tn_verbalizer.fst",
            "frontend_release/tn_cache/tn_manifest.json",
        ),
    },
}


def drop_file_pages(path: str, *, log_result: bool = True) -> int:
    """Evict file pages from the page cache (POSIX_FADV_DONTNEED).

    Ranking cgroup max_usage includes cache. Freshly written files are dirty;
    fsync first or DONTNEED is a no-op on Linux. After ORT copies weights,
    on-disk onnx/tar/wheel do not need to stay resident.
    """
    if not path or not os.path.exists(path):
        return 0
    files: list[str] = []
    if os.path.isfile(path):
        files = [path]
    else:
        for root, _, names in os.walk(path):
            for name in names:
                files.append(os.path.join(root, name))
    n = 0
    for file_path in files:
        fd = -1
        try:
            try:
                fd = os.open(file_path, os.O_RDWR)
            except OSError:
                fd = os.open(file_path, os.O_RDONLY)
            try:
                os.fsync(fd)
            except OSError:
                pass
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            n += 1
        except OSError:
            continue
        finally:
            if fd >= 0:
                os.close(fd)
    if n and log_result:
        log.info("[model_downloader] dropped page cache for %s files under %s", n, path)
    return n


def _unlink_skipped(model_dir: str, skip_files) -> None:
    for name in skip_files or ():
        victim = os.path.join(model_dir, name)
        if os.path.isfile(victim):
            try:
                drop_file_pages(victim)
                os.unlink(victim)
                log.info("[model_downloader] removed unused %s", victim)
            except OSError as e:
                log.warning("[model_downloader] could not remove %s: %s", victim, e)


def _local_juicefs_src(url: str) -> str | None:
    name = os.path.basename(url.split("?", 1)[0])
    path = os.path.join(JUICEFS_LOCAL, name)
    return path if os.path.isfile(path) else None


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _wipe_model_dir(model_dir: str) -> None:
    """Remove any previous install so leftover TN/frontend files cannot linger."""
    if os.path.isdir(model_dir):
        log.info("[model_downloader] wiping stale model dir: %s", model_dir)
        shutil.rmtree(model_dir)
    elif os.path.exists(model_dir):
        os.unlink(model_dir)
    os.makedirs(model_dir, exist_ok=True)


def _verify_required(name: str, model_dir: str, required_files) -> None:
    missing = [
        rel
        for rel in required_files or ()
        if not os.path.isfile(os.path.join(model_dir, rel))
    ]
    if missing:
        raise RuntimeError(
            f"[model_downloader] {name}: install incomplete, missing: {missing}"
        )
    manifest = os.path.join(
        model_dir, "frontend_release", "tn_cache", "tn_manifest.json"
    )
    if not os.path.isfile(manifest):
        return
    meta = json.loads(open(manifest, encoding="utf-8").read())
    if meta.get("schema_version") != 1:
        raise RuntimeError(
            f"[model_downloader] {name}: unsupported tn_manifest schema"
        )
    for fst_name, fst_meta in (meta.get("fst") or {}).items():
        path = os.path.join(model_dir, "frontend_release", "tn_cache", fst_name)
        if not os.path.isfile(path):
            raise RuntimeError(
                f"[model_downloader] {name}: TN fst missing after install: {fst_name}"
            )
        size = os.path.getsize(path)
        digest = _file_sha256(path)
        if size != int(fst_meta.get("bytes", -1)) or digest != fst_meta.get("sha256"):
            raise RuntimeError(
                f"[model_downloader] {name}: TN checksum mismatch: {fst_name}"
            )


def _write_install_stamp(model_dir: str, src_path: str, src_kind: str) -> None:
    stamp = {
        "src_kind": src_kind,
        "src_path": src_path,
        "src_sha256": _file_sha256(src_path),
        "src_bytes": os.path.getsize(src_path),
        "src_mtime": os.path.getmtime(src_path),
    }
    path = os.path.join(model_dir, ".juicefs_install.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stamp, f, ensure_ascii=False, indent=2)
        f.write("\n")
    log.info(
        "[model_downloader] install stamp: kind=%s sha256=%s bytes=%s",
        src_kind,
        stamp["src_sha256"][:16],
        stamp["src_bytes"],
    )


def ensure_model(name: str, model_dir: str) -> None:
    """Always reinstall model_dir from the current JuiceFS/HTTP pack.

    Ranking hosts mount persistent /models. Reusing an old tree made local
    JuiceFS updates (TN / lexicon / onnx) invisible on the leaderboard.
    """
    info = MODELS.get(name)
    if not info:
        raise ValueError(f"Unknown model name: {name}")

    check_path = os.path.join(model_dir, info["check_file"])
    skip_files = info.get("skip_files") or ()
    required_files = info.get("required_files") or (info["check_file"],)
    url = info["url"]
    local = _local_juicefs_src(url)

    _wipe_model_dir(model_dir)

    if info.get("single_file"):
        dest = os.path.join(model_dir, info["check_file"])
        if local:
            log.info(f"[model_downloader] {name}: copy from data disk {local}")
            shutil.copy2(local, dest)
            _write_install_stamp(model_dir, local, "juicefs_local_file")
        else:
            log.info(f"[model_downloader] {name}: downloading from {url} ...")
            urlretrieve(url, dest, reporthook=_progress_hook(name))
            _write_install_stamp(model_dir, dest, "http_file")
        log.info(f"[model_downloader] {name}: done.")
        return

    suffix = ".zip" if url.endswith(".zip") else ".tar.bz2"
    if local:
        log.info(f"[model_downloader] {name}: extract from data disk {local}")
        if suffix == ".zip":
            _extract_zip(local, model_dir, skip_files)
        else:
            _extract_tar(local, model_dir, skip_files)
        _unlink_skipped(model_dir, skip_files)
        _verify_required(name, model_dir, required_files)
        _write_install_stamp(model_dir, local, "juicefs_local_tar")
        drop_file_pages(model_dir)
        log.info(f"[model_downloader] {name}: done (fresh install from JuiceFS).")
        if not os.path.exists(check_path):
            raise RuntimeError(
                f"[model_downloader] {name}: local extract completed but "
                f"{info['check_file']} not found in {model_dir}"
            )
        return

    log.info(f"[model_downloader] {name}: downloading from {url} ...")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        urlretrieve(url, tmp_path, reporthook=_progress_hook(name))
        log.info(f"[model_downloader] {name}: extracting to {model_dir} ...")
        if suffix == ".zip":
            _extract_zip(tmp_path, model_dir, skip_files)
        else:
            _extract_tar(tmp_path, model_dir, skip_files)
        _write_install_stamp(model_dir, tmp_path, "http_tar")
        drop_file_pages(tmp_path)
        log.info(f"[model_downloader] {name}: done (fresh install from HTTP).")
    finally:
        if os.path.exists(tmp_path):
            drop_file_pages(tmp_path)
            os.unlink(tmp_path)
    _unlink_skipped(model_dir, skip_files)
    _verify_required(name, model_dir, required_files)
    drop_file_pages(model_dir)

    if not os.path.exists(check_path):
        raise RuntimeError(
            f"[model_downloader] {name}: download completed but {info['check_file']} "
            f"not found in {model_dir}"
        )


def _extract_zip(zip_path: str, model_dir: str, skip_files=()) -> None:
    """Extract zip, stripping common top-level directory prefix."""
    skip = set(skip_files or ())
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [
            n
            for n in zf.namelist()
            if not n.endswith("/") and not n.startswith("__MACOSX")
        ]
        if not names:
            raise RuntimeError(f"Empty archive: {zip_path}")

        prefix = _common_prefix_from_names(names)
        for name in names:
            stripped = name[len(prefix) :] if prefix else name
            if not stripped:
                continue
            if os.path.basename(stripped) in skip:
                continue
            dest = os.path.join(model_dir, stripped)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(name) as src, open(dest, "wb") as dst:
                dst.write(src.read())
                dst.flush()
                os.fsync(dst.fileno())
            drop_file_pages(dest, log_result=False)


def _extract_tar(tar_path: str, model_dir: str, skip_files=()) -> None:
    """Extract tar.bz2, stripping common top-level directory prefix."""
    skip = set(skip_files or ())
    with tarfile.open(tar_path, "r:bz2") as tf:
        members = tf.getmembers()
        if not members:
            raise RuntimeError(f"Empty archive: {tar_path}")

        names = [m.name for m in members if not m.isdir()]
        prefix = _common_prefix_from_names(names)
        for m in members:
            if m.isdir():
                continue
            # Copy member before mutating name (TarInfo is shared).
            name = m.name
            if prefix and name.startswith(prefix):
                name = name[len(prefix) :]
            name = name.lstrip("/")
            if not name:
                continue
            if os.path.basename(name) in skip:
                continue
            m.name = name
            tf.extract(m, model_dir)
            drop_file_pages(os.path.join(model_dir, m.name), log_result=False)


def _common_prefix_from_names(names: list[str]) -> str:
    """Find common top-level directory prefix from file name list."""
    dirs_with_slash = [n.split("/", 1) for n in names if "/" in n]
    if not dirs_with_slash:
        return ""
    first_parts = set(parts[0] for parts in dirs_with_slash)
    if len(first_parts) == 1:
        return first_parts.pop() + "/"
    return ""
