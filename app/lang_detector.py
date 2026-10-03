"""
lang_detector.py — Inference wrapper for language detection.
Combines script-level character analysis, Romanized Hinglish heuristics,
and the trained sklearn TF-IDF + LogisticRegression model for Latin languages.
"""
import re
import pickle
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

# Lazy-loaded classifier (loaded once on first call)
_classifier = None

# Common Hinglish marker words (Romanized Hindi)
COMMON_HINGLISH_WORDS = {
    "aap", "kaise", "kaisa", "kaisi", "ho", "hai", "hain", "kya", "kyu", "kyun",
    "nahi", "nahin", "main", "hum", "tum", "mera", "meri", "mere", "tera", "teri",
    "tere", "uska", "uski", "uske", "yeh", "ye", "woh", "wo", "kuch", "bahut",
    "bohot", "accha", "acha", "theek", "thik", "kar", "karo", "karna", "karne",
    "karunga", "karungi", "raha", "rahe", "rahi", "hoga", "hogi", "hoge", "liye",
    "saath", "baat", "batao", "dekh", "dekho", "chal", "chalo", "aaj", "kal", "ab",
    "kab", "yahan", "wahan", "kaun", "koi", "sab", "sabhi", "ghar", "kaam",
    "shukriya", "dhanyawad", "kripya", "samajh", "bhai", "yaar", "namaste", "namaskar"
}

LATIN_CLASSES = ["English", "French", "German", "Spanish", "Portuguese"]


def _load_classifier():
    """Load the sklearn pipeline from disk. Returns None if not yet trained."""
    global _classifier
    if _classifier is not None:
        return _classifier

    from app.config import LANG_DETECTOR_PATH
    path = Path(LANG_DETECTOR_PATH)

    if not path.exists():
        logger.warning(
            f"Language detector model not found at {path}. "
            "Run ml/train_lang_detector.py first. Auto-detection disabled."
        )
        return None

    with open(path, "rb") as f:
        _classifier = pickle.load(f)
    logger.info(f"Language detector loaded from {path}")
    return _classifier


def is_hinglish(text: str) -> bool:
    """
    Check if the text is code-switched Hinglish (mixed Devanagari/Latin or Romanized Hindi).
    """
    text_clean = text.strip()
    if not text_clean:
        return False

    devanagari_chars = sum(1 for c in text_clean if "\u0900" <= c <= "\u097F")
    latin_chars = sum(1 for c in text_clean if c.isascii() and c.isalpha())
    total_alpha = devanagari_chars + latin_chars

    # 1. Mixed script (Devanagari + Latin)
    if devanagari_chars > 0 and latin_chars > 0:
        latin_ratio = latin_chars / total_alpha
        if 0.1 <= latin_ratio <= 0.9:
            return True

    # 2. Romanized Hinglish (pure Latin characters with Hindi lexicon tokens)
    tokens = [w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", text_clean)]
    if tokens:
        hinglish_hits = sum(1 for w in tokens if w in COMMON_HINGLISH_WORDS)
        if hinglish_hits >= 2 or (len(tokens) <= 4 and hinglish_hits >= 1):
            return True

    return False


def detect_language(text: str) -> Tuple[str, float]:
    """
    Detect the language of the input text.

    Returns:
        (language_label, confidence) where language_label matches one of the
        SOURCE_LANGUAGES keys (e.g., 'English', 'Hindi', 'French', etc.).
        If the text is empty, returns ("Unknown", 0.0).
    """
    text_clean = text.strip()
    if not text_clean:
        return ("Unknown", 0.0)

    # 1. First check scripts (Devanagari, Arabic, Tamil, Cyrillic, CJK)
    devanagari = sum(1 for c in text_clean if "\u0900" <= c <= "\u097F")
    arabic = sum(1 for c in text_clean if "\u0600" <= c <= "\u06FF")
    tamil = sum(1 for c in text_clean if "\u0B80" <= c <= "\u0BFF")
    cyrillic = sum(1 for c in text_clean if "\u0400" <= c <= "\u04FF")
    cjk = sum(1 for c in text_clean if "\u4E00" <= c <= "\u9FFF")

    total_chars = max(len(re.sub(r"\s+", "", text_clean)), 1)

    if devanagari / total_chars > 0.15:
        return ("Hindi", 0.99)
    if arabic / total_chars > 0.15:
        return ("Arabic", 0.99)
    if tamil / total_chars > 0.15:
        return ("Tamil", 0.99)
    if cyrillic / total_chars > 0.15:
        return ("Russian", 0.99)
    if cjk / total_chars > 0.15:
        return ("Chinese (Simplified)", 0.99)

    # 2. Check for Romanized Hinglish
    if is_hinglish(text_clean):
        return ("Hindi", 0.95)

    # 3. Model inference for Latin-script languages (English, French, German, Spanish, Portuguese)
    clf = _load_classifier()
    if clf is None:
        return ("Unknown", 0.0)

    try:
        classes = list(clf.classes_)
        latin_indices = [classes.index(c) for c in LATIN_CLASSES if c in classes]

        proba = clf.predict_proba([text_clean])[0]
        sub_proba = [proba[i] for i in latin_indices]
        tot = sum(sub_proba) or 1.0
        norm_proba = [p / tot for p in sub_proba]

        best_idx = max(range(len(LATIN_CLASSES)), key=lambda k: norm_proba[k])
        label = LATIN_CLASSES[best_idx]
        confidence = float(norm_proba[best_idx])
        return (label, confidence)
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return ("Unknown", 0.0)

