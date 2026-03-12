"""FL FishSpeech VQ Decode node — decode VQ codes back to audio."""

from __future__ import annotations

import time

import torch

from comfy.utils import ProgressBar

import fs_tensor_utils  # type: ignore[import]

LOG_PREFIX = "[FL FishSpeech | VQ Decode]"


class FL_FishSpeech_VQDecode:
    """Decode VQ (Vector Quantized) codes back to audio using the Firefly VQGAN codec.

    Takes VQ codes from VQ Encode or other sources and produces audio output.
    """

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode"
    CATEGORY = "🐟FL FishSpeech"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fs_model": ("FS_MODEL",),
                "vq_codes": ("FS_VQ_CODES",),
            },
        }

    @torch.inference_mode()
    def decode(self, fs_model: dict, vq_codes: dict):
        pbar = ProgressBar(3)
        print(f"{LOG_PREFIX} {'='*50}")
        print(f"{LOG_PREFIX} Decoding VQ codes to audio")

        codec = fs_model["codec"]
        device = fs_model["device"]
        sample_rate = fs_model.get("sample_rate", 44100)

        codes = vq_codes["codes"].to(device=device)
        print(f"{LOG_PREFIX} Input codes: {list(codes.shape)}")

        pbar.update(1)
        t0 = time.perf_counter()

        # v2.0 DAC codec: from_indices(codes [B, n_codebooks, T]) → [B, 1, T_audio]
        audio_out = codec.from_indices(codes)
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"{LOG_PREFIX} Decoded: {list(audio_out.shape)} in {elapsed:.1f}ms")
        pbar.update(1)

        audio_trimmed = audio_out[0]  # [1, T_audio]
        num_samples = audio_trimmed.shape[-1]

        duration = num_samples / sample_rate
        print(f"{LOG_PREFIX} Output: {duration:.2f}s ({num_samples} samples at {sample_rate}Hz)")
        print(f"{LOG_PREFIX} Range: [{audio_trimmed.min().item():.4f}, {audio_trimmed.max().item():.4f}]")

        # Convert to ComfyUI AUDIO format
        result = fs_tensor_utils.tensor_to_comfyui_audio(audio_trimmed, sample_rate)
        print(f"{LOG_PREFIX} Output waveform: {list(result['waveform'].shape)}")
        print(f"{LOG_PREFIX} {'='*50}")
        pbar.update(1)

        return (result,)
