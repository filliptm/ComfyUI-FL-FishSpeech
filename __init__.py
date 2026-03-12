"""
FL FishSpeech - AI Text-to-Speech & Voice Cloning for ComfyUI
Based on Fish Speech (fishaudio/fish-speech-1.5)

Uses DualAR Transformer for semantic token generation and DAC codec
for high-quality 44.1kHz audio synthesis with voice cloning support.
"""

import sys
import os
import importlib.util

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add fish-speech to sys.path for imports
comfyui_root = os.path.dirname(os.path.dirname(current_dir))
fish_speech_root = os.path.join(comfyui_root, "fish-speech")
if os.path.exists(fish_speech_root) and fish_speech_root not in sys.path:
    sys.path.insert(0, fish_speech_root)


def import_module_from_path(module_name, file_path):
    """Import a module from an explicit file path to avoid naming conflicts."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Register fishspeech model folder with ComfyUI
try:
    import folder_paths

    fishspeech_models_path = os.path.join(folder_paths.models_dir, "fishspeech")
    os.makedirs(fishspeech_models_path, exist_ok=True)

    supported_extensions = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
    if "fishspeech" not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths["fishspeech"] = (
            [fishspeech_models_path],
            supported_extensions,
        )
except Exception:
    pass

# Import utilities first (needed by nodes) — using fs_ prefix to avoid collisions
fs_paths = import_module_from_path(
    "fs_paths",
    os.path.join(current_dir, "fl_utils", "paths.py"),
)
fs_tensor_utils = import_module_from_path(
    "fs_tensor_utils",
    os.path.join(current_dir, "fl_utils", "tensor_utils.py"),
)
fs_model_manager = import_module_from_path(
    "fs_model_manager",
    os.path.join(current_dir, "fl_utils", "model_manager.py"),
)

# Import nodes
fs_model_loader = import_module_from_path(
    "fs_model_loader",
    os.path.join(current_dir, "fl_nodes", "model_loader.py"),
)
fs_reference_audio = import_module_from_path(
    "fs_reference_audio",
    os.path.join(current_dir, "fl_nodes", "reference_audio.py"),
)
fs_tts = import_module_from_path(
    "fs_tts",
    os.path.join(current_dir, "fl_nodes", "tts.py"),
)
fs_vq_encode = import_module_from_path(
    "fs_vq_encode",
    os.path.join(current_dir, "fl_nodes", "vq_encode.py"),
)
fs_vq_decode = import_module_from_path(
    "fs_vq_decode",
    os.path.join(current_dir, "fl_nodes", "vq_decode.py"),
)
fs_transcribe = import_module_from_path(
    "fs_transcribe",
    os.path.join(current_dir, "fl_nodes", "transcribe.py"),
)

# Get node classes
FL_FishSpeech_ModelLoader = fs_model_loader.FL_FishSpeech_ModelLoader
FL_FishSpeech_ReferenceAudio = fs_reference_audio.FL_FishSpeech_ReferenceAudio
FL_FishSpeech_TTS = fs_tts.FL_FishSpeech_TTS
FL_FishSpeech_VQEncode = fs_vq_encode.FL_FishSpeech_VQEncode
FL_FishSpeech_VQDecode = fs_vq_decode.FL_FishSpeech_VQDecode
FL_FishSpeech_Transcribe = fs_transcribe.FL_FishSpeech_Transcribe

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FL_FishSpeech_ModelLoader": FL_FishSpeech_ModelLoader,
    "FL_FishSpeech_ReferenceAudio": FL_FishSpeech_ReferenceAudio,
    "FL_FishSpeech_TTS": FL_FishSpeech_TTS,
    "FL_FishSpeech_VQEncode": FL_FishSpeech_VQEncode,
    "FL_FishSpeech_VQDecode": FL_FishSpeech_VQDecode,
    "FL_FishSpeech_Transcribe": FL_FishSpeech_Transcribe,
}

# Display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_FishSpeech_ModelLoader": "FL FishSpeech Model Loader",
    "FL_FishSpeech_ReferenceAudio": "FL FishSpeech Reference Audio",
    "FL_FishSpeech_TTS": "FL FishSpeech TTS",
    "FL_FishSpeech_VQEncode": "FL FishSpeech VQ Encode",
    "FL_FishSpeech_VQDecode": "FL FishSpeech VQ Decode",
    "FL_FishSpeech_Transcribe": "FL FishSpeech Transcribe",
}

# Version info
__version__ = "1.0.0"

print(f"\033[36m[FL FishSpeech] v{__version__} - AI Text-to-Speech & Voice Cloning (Fish Audio S2)\033[0m")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
