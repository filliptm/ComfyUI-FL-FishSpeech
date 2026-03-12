"""Model download, caching, and lifecycle management for FL FishSpeech."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

LOG_PREFIX = "[FL FishSpeech]"

# HuggingFace repo info — v2.0 (gated model, requires `huggingface-cli login`)
HF_REPO_ID = "fishaudio/openaudio-s1-mini"
CODEC_FILENAME = "codec.pth"

# Global model cache
_FS_MODEL_CACHE: dict[str, Any] = {}


def _cache_key(device: str, precision: str, compile: bool) -> str:
    return f"{device}_{precision}_{compile}"


def _format_bytes(num_bytes: int | float) -> str:
    """Format bytes into human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def _get_vram_info() -> str:
    """Get current VRAM usage string, or empty if not CUDA."""
    if not torch.cuda.is_available():
        return ""
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory
    return (
        f" | VRAM: {_format_bytes(allocated)} allocated, "
        f"{_format_bytes(reserved)} reserved, "
        f"{_format_bytes(total)} total"
    )


def _count_params(model: torch.nn.Module) -> str:
    """Count and format model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total >= 1e9:
        return f"{total/1e9:.1f}B params ({trainable/1e9:.1f}B trainable)"
    return f"{total/1e6:.0f}M params ({trainable/1e6:.0f}M trainable)"


def download_models_if_needed(models_dir: Path) -> Path:
    """Download the FishSpeech checkpoint from HuggingFace if not already present.

    Returns the path to the checkpoint directory.
    """
    print(f"{LOG_PREFIX} Models directory: {models_dir}")

    checkpoint_dir = models_dir / "openaudio-s1-mini"

    # Check if already downloaded by looking for key files
    config_file = checkpoint_dir / "config.json"
    codec_file = checkpoint_dir / CODEC_FILENAME

    if config_file.exists() and codec_file.exists():
        print(f"{LOG_PREFIX} Checkpoint found: {checkpoint_dir}")
        return checkpoint_dir

    print(f"{LOG_PREFIX} Checkpoint not found at {checkpoint_dir}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for auto-downloading. "
            "Install with: pip install huggingface-hub"
        )

    print(f"{LOG_PREFIX} Downloading model from HuggingFace: {HF_REPO_ID}")
    print(f"{LOG_PREFIX} NOTE: This is a gated model — you must run "
          f"'huggingface-cli login' first and have accepted access at "
          f"https://huggingface.co/{HF_REPO_ID}")
    print(f"{LOG_PREFIX} This is a one-time ~8GB download (transformer + codec + tokenizer)...")

    t0 = time.perf_counter()
    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=str(checkpoint_dir),
        local_dir_use_symlinks=False,
    )
    elapsed = time.perf_counter() - t0

    if not config_file.exists():
        raise FileNotFoundError(
            f"Download appeared to succeed but config.json not found at {checkpoint_dir}"
        )

    print(f"{LOG_PREFIX} Download complete in {elapsed:.1f}s")
    return checkpoint_dir


def _detect_device() -> str:
    """Detect the best available device."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory
        print(f"{LOG_PREFIX} CUDA available: {gpu_name} ({_format_bytes(total_vram)} VRAM)")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"{LOG_PREFIX} MPS (Apple Silicon) available")
        return "mps"
    print(f"{LOG_PREFIX} No GPU detected, falling back to CPU")
    return "cpu"


def _precision_to_dtype(precision: str) -> torch.dtype:
    """Map precision string to torch dtype."""
    precision_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return precision_map.get(precision, torch.bfloat16)


def load_fs_model(
    device: str | None = None,
    precision: str = "bfloat16",
    compile: bool = False,
    force_reload: bool = False,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Load (or retrieve cached) FishSpeech model bundle.

    Returns a dict with 'model', 'decode_one_token', 'codec', 'device', 'precision', 'sample_rate'.
    """
    import sys

    import fs_paths  # type: ignore[import]

    print(f"{LOG_PREFIX} {'='*60}")
    print(f"{LOG_PREFIX} Loading FishSpeech Engine (v2.0 — openaudio-s1-mini)")
    print(f"{LOG_PREFIX} {'='*60}")
    print(f"{LOG_PREFIX} Config: precision={precision}, compile={compile}, "
          f"force_reload={force_reload}")

    if device is None:
        device = _detect_device()
    else:
        print(f"{LOG_PREFIX} Using requested device: {device}")

    key = _cache_key(device, precision, compile)

    if not force_reload and key in _FS_MODEL_CACHE:
        print(f"{LOG_PREFIX} Model found in cache (key={key})")
        print(f"{LOG_PREFIX} Skipping model load — returning cached model{_get_vram_info()}")
        return _FS_MODEL_CACHE[key]

    if force_reload and key in _FS_MODEL_CACHE:
        print(f"{LOG_PREFIX} Force reload requested — clearing cached model")
        del _FS_MODEL_CACHE[key]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Ensure fish-speech is on sys.path
    fish_root = fs_paths.get_fish_speech_root()
    fish_root_str = str(fish_root)
    if fish_root_str not in sys.path:
        sys.path.insert(0, fish_root_str)
        print(f"{LOG_PREFIX} Added fish-speech to sys.path: {fish_root_str}")

    # Download checkpoint if needed
    if models_dir is None:
        models_dir = fs_paths.get_models_directory()
    checkpoint_dir = download_models_if_needed(models_dir)

    dtype = _precision_to_dtype(precision)

    # Import v2.0 inference functions
    from fish_speech.models.text2semantic.inference import (
        init_model,
        load_codec_model,
    )

    # Load DualAR transformer (includes tokenizer via from_pretrained)
    print(f"{LOG_PREFIX} Loading DualAR transformer...")
    print(f"{LOG_PREFIX}   - Checkpoint: {checkpoint_dir}")
    print(f"{LOG_PREFIX}   - Device: {device}")
    print(f"{LOG_PREFIX}   - Precision: {dtype}")
    print(f"{LOG_PREFIX}   - Compile: {'enabled (slow first run)' if compile else 'disabled'}")

    if torch.cuda.is_available():
        print(f"{LOG_PREFIX} Pre-load{_get_vram_info()}")

    t0 = time.perf_counter()
    model, decode_one_token = init_model(
        checkpoint_path=str(checkpoint_dir),
        device=device,
        precision=dtype,
        compile=compile,
    )
    t_model = time.perf_counter() - t0
    print(f"{LOG_PREFIX} Transformer loaded in {t_model:.2f}s — {_count_params(model)}")

    # Setup KV caches explicitly (matches official CLI pattern)
    print(f"{LOG_PREFIX} Setting up KV caches (max_seq_len={model.config.max_seq_len})...")
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )
    print(f"{LOG_PREFIX} KV caches ready")

    # Load DAC codec via v2.0 load_codec_model
    codec_path = str(checkpoint_dir / CODEC_FILENAME)
    print(f"{LOG_PREFIX} Loading DAC codec...")
    t1 = time.perf_counter()
    codec = load_codec_model(codec_path, device, precision=dtype)
    t_codec = time.perf_counter() - t1
    sample_rate = codec.sample_rate
    print(f"{LOG_PREFIX} Codec loaded in {t_codec:.2f}s — sample_rate={sample_rate}")

    if torch.cuda.is_available():
        print(f"{LOG_PREFIX} Post-load{_get_vram_info()}")

    total_time = time.perf_counter() - t0
    print(f"{LOG_PREFIX} Total load time: {total_time:.2f}s")

    result = {
        "model": model,
        "decode_one_token": decode_one_token,
        "codec": codec,
        "device": device,
        "precision": dtype,
        "sample_rate": sample_rate,
    }

    _FS_MODEL_CACHE[key] = result
    print(f"{LOG_PREFIX} Model cached (key={key})")
    print(f"{LOG_PREFIX} {'='*60}")
    print(f"{LOG_PREFIX} FishSpeech Engine ready!")
    print(f"{LOG_PREFIX} {'='*60}")
    return result


def clear_model_cache() -> None:
    """Clear all cached models to free memory."""
    count = len(_FS_MODEL_CACHE)
    _FS_MODEL_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"{LOG_PREFIX} Cleared {count} cached model(s)")
