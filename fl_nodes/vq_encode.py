"""FL FishSpeech VQ Encode node — encode audio to VQ codes."""

from __future__ import annotations

import time

import torch

from comfy.utils import ProgressBar

import fs_tensor_utils  # type: ignore[import]

LOG_PREFIX = "[FL FishSpeech | VQ Encode]"


class FL_FishSpeech_VQEncode:
    """Encode audio into VQ (Vector Quantized) codes using the Firefly VQGAN codec.

    Useful for inspecting, saving, or manipulating the discrete audio representation.
    """

    RETURN_TYPES = ("FS_VQ_CODES",)
    RETURN_NAMES = ("vq_codes",)
    FUNCTION = "encode"
    CATEGORY = "🐟FL FishSpeech"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fs_model": ("FS_MODEL",),
                "audio": ("AUDIO",),
            },
        }

    @torch.inference_mode()
    def encode(self, fs_model: dict, audio: dict):
        pbar = ProgressBar(3)
        print(f"{LOG_PREFIX} {'='*50}")
        print(f"{LOG_PREFIX} Encoding audio to VQ codes")

        codec = fs_model["codec"]
        device = fs_model["device"]
        sample_rate = fs_model.get("sample_rate", 44100)

        # Extract waveform
        waveform, orig_sr = fs_tensor_utils.comfyui_audio_to_tensor(audio)
        print(f"{LOG_PREFIX} Input: {list(waveform.shape)}, sample_rate={orig_sr}")

        # Ensure mono
        if waveform.dim() == 3 and waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)
            print(f"{LOG_PREFIX} Converted to mono: {list(waveform.shape)}")

        # Resample if needed
        if orig_sr != sample_rate:
            print(f"{LOG_PREFIX} Resampling {orig_sr}Hz -> {sample_rate}Hz")
            waveform = fs_tensor_utils.resample_audio(waveform, orig_sr, sample_rate)

        num_samples = waveform.shape[-1]
        duration = num_samples / sample_rate
        print(f"{LOG_PREFIX} Audio: {duration:.2f}s ({num_samples} samples at {sample_rate}Hz)")

        # Prepare for codec: [B, 1, T] matching codec dtype
        model_dtype = next(codec.parameters()).dtype
        if waveform.dim() == 3:
            audio_data = waveform[0:1].to(device=device, dtype=model_dtype)
            if audio_data.shape[1] > 1:
                audio_data = audio_data[:, :1, :]
        elif waveform.dim() == 2:
            audio_data = waveform.unsqueeze(0).to(device=device, dtype=model_dtype)
        else:
            audio_data = waveform.unsqueeze(0).unsqueeze(0).to(device=device, dtype=model_dtype)

        audio_lengths = torch.tensor(
            [audio_data.shape[-1]], device=device, dtype=torch.long
        )

        pbar.update(1)

        # Encode using DAC codec
        t0 = time.perf_counter()
        indices, feature_lengths = codec.encode(audio_data, audio_lengths)
        elapsed = (time.perf_counter() - t0) * 1000

        # Trim to actual feature length (codec pads to frame_length boundaries)
        actual_len = feature_lengths[0].item()
        indices = indices[:, :, :actual_len]

        print(f"{LOG_PREFIX} Encoded: {list(indices.shape)} in {elapsed:.1f}ms")
        print(f"{LOG_PREFIX} Feature length: {actual_len}")

        compression_ratio = num_samples / max(indices.shape[-1], 1)
        print(f"{LOG_PREFIX} Compression ratio: {compression_ratio:.1f}x")
        pbar.update(1)

        result = {
            "codes": indices.cpu(),
            "feature_lengths": feature_lengths.cpu(),
            "sample_rate": sample_rate,
        }

        print(f"{LOG_PREFIX} {'='*50}")
        pbar.update(1)
        return (result,)
