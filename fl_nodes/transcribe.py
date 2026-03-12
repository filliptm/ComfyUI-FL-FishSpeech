"""FL FishSpeech Transcribe node — transcribe audio to text using Whisper."""

from __future__ import annotations

import logging
import traceback

import numpy as np
import torch

from comfy.utils import ProgressBar

import fs_tensor_utils  # type: ignore[import]

logger = logging.getLogger("FL_FishSpeech")

LOG_PREFIX = "[FL FishSpeech | Transcribe]"

# Whisper expects 16kHz audio
WHISPER_SAMPLE_RATE = 16000

# Available Whisper models
WHISPER_MODELS = [
    "openai/whisper-large-v3-turbo",
    "openai/whisper-large-v3",
    "openai/whisper-medium",
    "openai/whisper-small",
    "openai/whisper-base",
    "openai/whisper-tiny",
]

# Model cache to avoid reloading
_whisper_cache = {}


def _resolve_device(device="auto"):
    """Resolve 'auto' device to actual device string."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device


def _get_whisper_model(model_name, device):
    """Load or retrieve cached Whisper model and processor."""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    torch_dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32
    cache_key = (model_name, device, str(torch_dtype))

    if cache_key not in _whisper_cache:
        print(f"{LOG_PREFIX} Loading Whisper model: {model_name} on {device}")
        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        whisper_model.to(device)
        processor = AutoProcessor.from_pretrained(model_name)
        _whisper_cache[cache_key] = (whisper_model, processor)
        print(f"{LOG_PREFIX} Whisper model loaded successfully")
    else:
        print(f"{LOG_PREFIX} Using cached Whisper model")

    return _whisper_cache[cache_key]


def transcribe_audio(audio, model_name="openai/whisper-base", language="auto", device="auto"):
    """Transcribe ComfyUI audio dict to text using Whisper.

    Args:
        audio: ComfyUI audio dict {"waveform": Tensor, "sample_rate": int}
        model_name: Whisper model name
        language: Language code or "auto"
        device: "auto", "cuda", "cpu", or "mps"

    Returns:
        Transcription string (empty string on failure).
    """
    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "transformers library required for transcription. "
            "Install with: pip install transformers"
        )

    device = _resolve_device(device)

    try:
        whisper_model, processor = _get_whisper_model(model_name, device)
        torch_dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32

        # Prepare audio
        waveform, sr = fs_tensor_utils.comfyui_audio_to_tensor(audio)
        waveform = fs_tensor_utils.ensure_mono(waveform)

        # Flatten to 1D
        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        # Resample to 16kHz (Whisper's expected sample rate)
        if sr != WHISPER_SAMPLE_RATE:
            print(f"{LOG_PREFIX} Resampling from {sr}Hz to {WHISPER_SAMPLE_RATE}Hz")
            waveform = fs_tensor_utils.resample_audio(waveform, sr, WHISPER_SAMPLE_RATE)
            sr = WHISPER_SAMPLE_RATE

        # Convert to numpy float32
        audio_np = waveform.numpy().astype(np.float32)

        print(f"{LOG_PREFIX} Transcribing audio: {len(audio_np) / sr:.1f}s at {sr}Hz")

        # Process audio through feature extractor
        input_features = processor(
            audio_np,
            sampling_rate=sr,
            return_tensors="pt",
        ).input_features.to(device, dtype=torch_dtype)

        # Prepare generation kwargs
        generate_kwargs = {}
        if language != "auto":
            generate_kwargs["language"] = language
            generate_kwargs["task"] = "transcribe"

        # Generate transcription
        with torch.no_grad():
            predicted_ids = whisper_model.generate(input_features, **generate_kwargs)

        # Decode the output
        transcription = processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        print(
            f"{LOG_PREFIX} Transcription: {transcription[:100]}..."
            if len(transcription) > 100
            else f"{LOG_PREFIX} Transcription: {transcription}"
        )

        return transcription

    except Exception as e:
        print(f"{LOG_PREFIX} Transcription failed: {e}")
        traceback.print_exc()
        return ""


class FL_FishSpeech_Transcribe:
    """Transcribe audio to text using Whisper.

    Useful for generating transcript text for the Reference Audio node
    when you don't have a manual transcript of your reference clip.
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transcription",)
    FUNCTION = "transcribe"
    CATEGORY = "🐟FL FishSpeech"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (
                    WHISPER_MODELS,
                    {"default": "openai/whisper-large-v3-turbo"},
                ),
                "language": (
                    ["auto", "en", "zh", "ja", "ko", "de", "fr", "es", "pt", "ru", "it"],
                    {"default": "auto"},
                ),
            },
            "optional": {
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            },
        }

    def transcribe(self, audio, model, language, device="auto"):
        pbar = ProgressBar(3)
        print(f"{LOG_PREFIX} {'='*50}")
        pbar.update(1)

        transcription = transcribe_audio(
            audio, model_name=model, language=language, device=device
        )
        pbar.update(1)

        print(f"{LOG_PREFIX} {'='*50}")
        pbar.update(1)

        return (transcription,)
