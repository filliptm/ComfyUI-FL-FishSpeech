"""FL FishSpeech Reference Audio node — encode reference audio for voice cloning."""

from __future__ import annotations

import time

import torch

from comfy.utils import ProgressBar

import fs_tensor_utils  # type: ignore[import]

LOG_PREFIX = "[FL FishSpeech | Reference Audio]"


class FL_FishSpeech_ReferenceAudio:
    """Encode a reference audio clip + transcript into prompt tokens for voice cloning.

    The reference audio should be 10-30 seconds of clear speech.
    The transcript must match the spoken content accurately.
    """

    RETURN_TYPES = ("FS_REFERENCE",)
    RETURN_NAMES = ("fs_reference",)
    FUNCTION = "encode_reference"
    CATEGORY = "🐟FL FishSpeech"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fs_model": ("FS_MODEL",),
                "audio": ("AUDIO",),
                "transcript": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Exact transcript of the reference audio. "
                        "Must match the spoken content for best results.",
                    },
                ),
            },
        }

    @torch.inference_mode()
    def encode_reference(self, fs_model: dict, audio: dict, transcript: str):
        pbar = ProgressBar(3)
        print(f"{LOG_PREFIX} {'='*50}")
        print(f"{LOG_PREFIX} Encoding reference audio for voice cloning")

        codec = fs_model["codec"]
        device = fs_model["device"]
        sample_rate = fs_model.get("sample_rate", 44100)

        # Extract waveform from ComfyUI AUDIO
        waveform, orig_sr = fs_tensor_utils.comfyui_audio_to_tensor(audio)
        print(f"{LOG_PREFIX} Input: {waveform.shape}, sample_rate={orig_sr}")

        # Ensure mono
        if waveform.dim() == 3 and waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)
            print(f"{LOG_PREFIX} Converted to mono: {waveform.shape}")

        # Resample if needed
        if orig_sr != sample_rate:
            print(f"{LOG_PREFIX} Resampling {orig_sr}Hz -> {sample_rate}Hz")
            waveform = fs_tensor_utils.resample_audio(waveform, orig_sr, sample_rate)
            print(f"{LOG_PREFIX} Resampled: {waveform.shape}")

        # Calculate duration
        num_samples = waveform.shape[-1]
        duration = num_samples / sample_rate
        print(f"{LOG_PREFIX} Audio duration: {duration:.2f}s ({num_samples} samples)")
        print(f"{LOG_PREFIX} Transcript: {transcript[:100]}{'...' if len(transcript) > 100 else ''}")

        # Prepare for codec: [B, 1, T] matching codec dtype
        t0 = time.perf_counter()
        model_dtype = next(codec.parameters()).dtype

        if waveform.dim() == 3:
            audio_data = waveform[0:1].to(device=device, dtype=model_dtype)
            if audio_data.shape[1] > 1:
                audio_data = audio_data[:, :1, :]
        else:
            audio_data = waveform.unsqueeze(0).to(device=device, dtype=model_dtype)

        audio_lengths = torch.tensor(
            [audio_data.shape[-1]], device=device, dtype=torch.long
        )

        pbar.update(1)

        # Encode with DAC codec
        indices, feature_lengths = codec.encode(audio_data, audio_lengths)
        # Trim to actual feature length (codec pads to frame_length boundaries)
        prompt_tokens = indices[0, :, :feature_lengths[0]]  # [n_codebooks, T']

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"{LOG_PREFIX} Encoded to VQ tokens: {list(prompt_tokens.shape)} in {elapsed:.1f}ms")
        print(f"{LOG_PREFIX} Codebook groups: {prompt_tokens.shape[0]}, "
              f"Time steps: {prompt_tokens.shape[1]}")
        pbar.update(1)

        result = {
            "prompt_tokens": prompt_tokens.cpu(),
            "prompt_text": transcript,
        }

        print(f"{LOG_PREFIX} Reference audio encoded successfully")
        print(f"{LOG_PREFIX} {'='*50}")
        pbar.update(1)

        return (result,)
