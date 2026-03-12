"""Path utilities for FL FishSpeech ComfyUI node pack."""

from __future__ import annotations

import os
from pathlib import Path


def get_models_directory() -> Path:
    """Get the FishSpeech models directory under ComfyUI/models/fishspeech/.

    Creates the directory if it doesn't exist.
    """
    try:
        import folder_paths

        base = Path(folder_paths.models_dir)
    except ImportError:
        base = Path(__file__).resolve().parent.parent.parent.parent / "models"

    models_dir = base / "fishspeech"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_fish_speech_root() -> Path:
    """Get the root directory of the fish-speech repository.

    Expected at ComfyUI/fish-speech/ (sibling of custom_nodes/).
    """
    comfyui_root = Path(__file__).resolve().parent.parent.parent.parent
    fish_root = comfyui_root / "fish-speech"
    if not fish_root.exists():
        raise FileNotFoundError(
            f"fish-speech directory not found at {fish_root}. "
            f"Please clone or place the fish-speech repo there."
        )
    return fish_root


def get_package_root() -> Path:
    """Get the root directory of this node pack."""
    return Path(__file__).resolve().parent.parent
