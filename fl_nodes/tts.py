"""FL FishSpeech TTS node — text-to-speech generation with optional voice cloning."""

from __future__ import annotations

import re
import time

import torch

from comfy.utils import ProgressBar

import fs_tensor_utils  # type: ignore[import]
import fs_s1_compat  # type: ignore[import]

LOG_PREFIX = "[FL FishSpeech | TTS]"


class FL_FishSpeech_TTS:
    """Generate speech from text using the FishSpeech DualAR transformer.

    Optionally clone a voice by providing a reference audio encoding.
    Supports multi-speaker via <|speaker:X|> tags and emotion control
    via inline tags like [laugh], [whispers], etc.
    """

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "🐟FL FishSpeech"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fs_model": ("FS_MODEL",),
                "text": (
                    "STRING",
                    {
                        "default": "Hello, this is a test of the FishSpeech text to speech system.",
                        "multiline": True,
                        "tooltip": "Text to speak. Use <|speaker:X|> for multi-speaker, "
                        "[laugh], [whispers] for emotion control.",
                    },
                ),
            },
            "optional": {
                "fs_reference": (
                    "FS_REFERENCE",
                    {
                        "tooltip": "Optional reference audio encoding for voice cloning.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "tooltip": "Random seed. 0 = random.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Sampling temperature. Higher = more varied. v2.0 default is 1.0.",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Top-p nucleus sampling threshold. v2.0 default is 0.9.",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 100,
                        "tooltip": "Top-k sampling. Limits to top K most likely tokens.",
                    },
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1.1,
                        "min": 1.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Repetition penalty. Higher = less repetition.",
                    },
                ),
                "chunk_length": (
                    "INT",
                    {
                        "default": 512,
                        "min": 50,
                        "max": 1000,
                        "step": 50,
                        "tooltip": "Max bytes per text chunk for iterative generation.",
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 128,
                        "tooltip": "Maximum new tokens to generate per chunk. 0 = auto.",
                    },
                ),
            },
        }

    @torch.inference_mode()
    def generate(
        self,
        fs_model: dict,
        text: str,
        fs_reference: dict | None = None,
        seed: int = 0,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 30,
        repetition_penalty: float = 1.1,
        chunk_length: int = 512,
        max_new_tokens: int = 0,
    ):
        print(f"{LOG_PREFIX} {'='*50}")
        print(f"{LOG_PREFIX} Text-to-Speech generation")

        # Strip any user-provided speaker tags — generate_long now handles
        # speaker tags internally via ContentSequence.append(speaker=0)
        text = re.sub(r"<\|speaker:\d+\|>", "", text).strip()

        print(f"{LOG_PREFIX} Text: {text[:150]}{'...' if len(text) > 150 else ''}")
        print(f"{LOG_PREFIX} Text length: {len(text)} chars, {len(text.encode('utf-8'))} bytes")
        print(f"{LOG_PREFIX} Settings: temp={temperature}, top_p={top_p}, "
              f"top_k={top_k}, rep_penalty={repetition_penalty}")
        print(f"{LOG_PREFIX} chunk_length={chunk_length}, max_new_tokens={max_new_tokens}, "
              f"seed={seed}")
        print(f"{LOG_PREFIX} Voice cloning: {'YES' if fs_reference else 'NO'}")

        model = fs_model["model"]
        decode_one_token = fs_model["decode_one_token"]
        codec = fs_model["codec"]
        device = fs_model["device"]
        sample_rate = fs_model.get("sample_rate", 44100)

        # Set seed
        if seed > 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            print(f"{LOG_PREFIX} Seed set to {seed}")
        else:
            print(f"{LOG_PREFIX} Using random seed")

        # Prepare prompt tokens/text for voice cloning
        prompt_tokens = None
        prompt_text = None
        if fs_reference is not None:
            prompt_tokens = [fs_reference["prompt_tokens"].to(device)]
            prompt_text = [fs_reference["prompt_text"]]
            print(f"{LOG_PREFIX} Reference: tokens={list(prompt_tokens[0].shape)}, "
                  f"text='{prompt_text[0][:80]}...'")

        from fish_speech.models.text2semantic.inference import decode_to_audio

        generate_long = fs_s1_compat.generate_long_s1

        # Per-token progress bar: estimate max tokens, add extra steps for
        # setup (1), decode (1), and finalize (1)
        max_seq_len = getattr(model.config, "max_seq_len", 4096)
        estimated_max_tokens = max_new_tokens if max_new_tokens > 0 else max_seq_len
        extra_steps = 3  # setup + decode + finalize
        pbar = ProgressBar(estimated_max_tokens + extra_steps)

        pbar.update(1)  # Step: Setup complete

        # Progress callback — called every 5 tokens from decode_n_tokens
        _last_reported = [0]  # mutable container for closure

        def _on_token_progress(current_token, total_tokens):
            delta = current_token - _last_reported[0]
            if delta > 0:
                pbar.update(delta)
                _last_reported[0] = current_token

        # Generate
        print(f"{LOG_PREFIX} Starting generation...")
        t0 = time.perf_counter()

        all_codes = []
        chunk_count = 0

        generator = generate_long(
            model=model,
            device=device,
            decode_one_token=decode_one_token,
            text=text,
            num_samples=1,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            compile=False,
            iterative_prompt=True,
            chunk_length=chunk_length,
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
            progress_callback=_on_token_progress,
        )

        for response in generator:
            if response.action == "sample":
                all_codes.append(response.codes)
                chunk_count += 1
                print(f"{LOG_PREFIX} Chunk {chunk_count}: codes shape={list(response.codes.shape)}, "
                      f"text='{response.text[:60]}...'")
            elif response.action == "next":
                break

        t_gen = time.perf_counter() - t0

        if not all_codes:
            raise RuntimeError("No audio codes generated. Check text input and model state.")

        # Concatenate all codes and decode to audio
        merged_codes = torch.cat(all_codes, dim=1)
        total_tokens = merged_codes.shape[1]
        tokens_per_sec = total_tokens / t_gen if t_gen > 0 else 0
        print(f"{LOG_PREFIX} Generation complete: {total_tokens} tokens in {t_gen:.2f}s "
              f"({tokens_per_sec:.1f} tokens/sec)")

        # Jump progress to end of token generation phase (in case early stop
        # left the bar short of estimated_max_tokens)
        remaining_token_steps = estimated_max_tokens - _last_reported[0]
        if remaining_token_steps > 0:
            pbar.update(remaining_token_steps)

        # Decode codes to audio waveform using v2.0 DAC codec
        print(f"{LOG_PREFIX} Decoding VQ codes to audio...")
        t1 = time.perf_counter()

        # v2.0: decode_to_audio(codes, codec) → [T] mono waveform
        audio_waveform = decode_to_audio(merged_codes.to(device), codec)
        t_decode = time.perf_counter() - t1

        duration = len(audio_waveform) / sample_rate
        print(f"{LOG_PREFIX} Decoded: {len(audio_waveform)} samples, "
              f"{duration:.2f}s at {sample_rate}Hz in {t_decode*1000:.1f}ms")
        pbar.update(1)  # Step: Decode complete

        total_time = time.perf_counter() - t0
        print(f"{LOG_PREFIX} Total time: {total_time:.2f}s")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            print(f"{LOG_PREFIX} GPU memory: {allocated / 1e9:.2f} GB")

        # Convert to ComfyUI AUDIO format
        result = fs_tensor_utils.tensor_to_comfyui_audio(audio_waveform, sample_rate)
        print(f"{LOG_PREFIX} Output: waveform={list(result['waveform'].shape)}, "
              f"sample_rate={result['sample_rate']}")
        print(f"{LOG_PREFIX} {'='*50}")
        pbar.update(1)  # Step: Complete

        return (result,)
