"""FL FishSpeech Model Loader node — load transformer, codec, and tokenizer."""

from __future__ import annotations

import time

from comfy.utils import ProgressBar

import fs_model_manager  # type: ignore[import]

LOG_PREFIX = "[FL FishSpeech | Model Loader]"


class FL_FishSpeech_ModelLoader:
    """Load the FishSpeech model bundle (DualAR transformer + DAC codec + tokenizer).

    Auto-downloads from HuggingFace on first use (~8GB).
    Models are cached between runs for fast reuse.
    """

    RETURN_TYPES = ("FS_MODEL",)
    RETURN_NAMES = ("fs_model",)
    FUNCTION = "load_model"
    CATEGORY = "🐟FL FishSpeech"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "device": (
                    ["auto", "cuda", "cpu"],
                    {
                        "default": "auto",
                        "tooltip": "Device to load the model on. 'auto' detects GPU automatically.",
                    },
                ),
                "precision": (
                    ["bfloat16", "float16", "float32"],
                    {
                        "default": "bfloat16",
                        "tooltip": "Model precision. bfloat16 recommended for CUDA, float32 for CPU.",
                    },
                ),
                "compile": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable torch.compile for faster generation. "
                        "First run will be slow due to compilation.",
                    },
                ),
                "force_reload": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Force reload the model, clearing cache.",
                    },
                ),
            },
        }

    def load_model(
        self,
        device: str = "auto",
        precision: str = "bfloat16",
        compile: bool = False,
        force_reload: bool = False,
    ):
        pbar = ProgressBar(3)
        print(f"{LOG_PREFIX} {'='*50}")
        print(f"{LOG_PREFIX} Node execution started")
        print(f"{LOG_PREFIX} device={device}, precision={precision}, "
              f"compile={compile}, force_reload={force_reload}")
        pbar.update(1)

        t0 = time.perf_counter()

        device_arg = None if device == "auto" else device
        result = fs_model_manager.load_fs_model(
            device=device_arg,
            precision=precision,
            compile=compile,
            force_reload=force_reload,
        )
        pbar.update(1)

        elapsed = time.perf_counter() - t0
        print(f"{LOG_PREFIX} Model ready on {result['device']} in {elapsed:.2f}s")
        print(f"{LOG_PREFIX} {'='*50}")
        pbar.update(1)

        return (result,)
