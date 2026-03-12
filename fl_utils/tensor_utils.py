"""Audio tensor conversion utilities for FL FishSpeech ComfyUI nodes."""

from __future__ import annotations

import torch


def comfyui_audio_to_tensor(audio: dict) -> tuple[torch.Tensor, int]:
    """Extract waveform tensor and sample rate from ComfyUI AUDIO format.

    Args:
        audio: ComfyUI AUDIO dict with "waveform" [B, C, T] and "sample_rate" int.

    Returns:
        (waveform, sample_rate) where waveform is [B, C, T] float tensor.
    """
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    return waveform, sample_rate


def tensor_to_comfyui_audio(waveform: torch.Tensor, sample_rate: int) -> dict:
    """Convert a waveform tensor to ComfyUI AUDIO format.

    Args:
        waveform: [B, C, T] or [C, T] or [T] tensor.
        sample_rate: Audio sample rate.

    Returns:
        ComfyUI AUDIO dict.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)  # [1, C, T]

    return {"waveform": waveform.cpu().float(), "sample_rate": sample_rate}


def ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Ensure audio is mono by averaging channels if stereo.

    Args:
        waveform: Audio tensor (1D, 2D [C, T], or 3D [B, C, T]).

    Returns:
        Mono audio tensor.
    """
    if waveform.dim() == 1:
        return waveform
    elif waveform.dim() == 2:
        if waveform.shape[0] > 1:
            return waveform.mean(dim=0)
        return waveform[0]
    elif waveform.dim() == 3:
        if waveform.shape[1] > 1:
            return waveform.mean(dim=1, keepdim=True)
        return waveform
    return waveform


def resample_audio(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample audio waveform if sample rates differ.

    Args:
        waveform: Audio tensor (any shape, last dim is time).
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled tensor (same shape except time dimension).
    """
    if orig_sr == target_sr:
        return waveform

    import torchaudio

    return torchaudio.functional.resample(waveform, orig_sr, target_sr)
