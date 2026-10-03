"""
config.py — Central configuration for GenAI Translator.
All paths, model names, and supported languages live here.
"""
import os
import torch
from pathlib import Path

# ── Project root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
ML_ARTIFACTS = PROJECT_ROOT / "ml" / "artifacts"

# ── Model config ─────────────────────────────────────────────────────────────
MODEL_NAME = "facebook/nllb-200-distilled-600M"
LORA_ADAPTER_PATH = str(ML_ARTIFACTS / "nllb_hinglish_adapter")
LANG_DETECTOR_PATH = str(ML_ARTIFACTS / "lang_detector.pkl")
WHISPER_MODEL_SIZE = "base"
MAX_NEW_TOKENS = 256

# ── Device detection ─────────────────────────────────────────────────────────
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()

# ── Supported languages ───────────────────────────────────────────────────────
# Format: display_name -> (nllb_code, gtts_lang_code)
SUPPORTED_LANGUAGES: dict[str, tuple[str, str]] = {
    "English":              ("eng_Latn", "en"),
    "Hindi":                ("hin_Deva", "hi"),
    "Hinglish → English":   ("hin_Deva", "en"),   # special: src treated as Hindi
    "French":               ("fra_Latn", "fr"),
    "Spanish":              ("spa_Latn", "es"),
    "German":               ("deu_Latn", "de"),
    "Chinese (Simplified)": ("zho_Hans", "zh"),
    "Arabic":               ("arb_Arab", "ar"),
    "Portuguese":           ("por_Latn", "pt"),
    "Russian":              ("rus_Cyrl", "ru"),
    "Tamil":                ("tam_Taml", "ta"),
}

# Languages available as translation SOURCE (excludes the special Hinglish entry)
SOURCE_LANGUAGES = {k: v for k, v in SUPPORTED_LANGUAGES.items() if k != "Hinglish → English"}
TARGET_LANGUAGES = SOURCE_LANGUAGES  # same set for targets

# ── NLLB lang code → display name (for Whisper detected language mapping) ────
NLLB_TO_DISPLAY: dict[str, str] = {v[0]: k for k, v in SOURCE_LANGUAGES.items()}

# ── Whisper detected lang code → NLLB code ───────────────────────────────────
WHISPER_TO_NLLB: dict[str, str] = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "de": "deu_Latn",
    "zh": "zho_Hans",
    "ar": "arb_Arab",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "ta": "tam_Taml",
}

# ── Language detector classes (must match train_lang_detector.py label order) ─
LANG_DETECTOR_CLASSES = [
    "English", "Hindi", "French", "Spanish", "German",
    "Chinese", "Arabic", "Portuguese", "Russian", "Tamil",
]
