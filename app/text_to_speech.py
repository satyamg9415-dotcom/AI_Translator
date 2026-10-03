"""
text_to_speech.py — gTTS-based text-to-speech synthesis.
Converts translated text to an audio file for Streamlit playback.
"""
import logging
import tempfile
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# NLLB code → gTTS lang code
NLLB_TO_GTTS: dict[str, str] = {
    "eng_Latn": "en",
    "hin_Deva": "hi",
    "fra_Latn": "fr",
    "spa_Latn": "es",
    "deu_Latn": "de",
    "zho_Hans": "zh",
    "arb_Arab": "ar",
    "por_Latn": "pt",
    "rus_Cyrl": "ru",
    "tam_Taml": "ta",
}


def synthesize(text: str, lang_nllb_code: str, slow: bool = False) -> bytes:
    """
    Synthesize speech from text.

    Args:
        text:           Text to speak.
        lang_nllb_code: NLLB language code (e.g. "fra_Latn").
        slow:           If True, speak slowly.

    Returns:
        MP3 audio bytes.
    """
    from gtts import gTTS

    gtts_lang = NLLB_TO_GTTS.get(lang_nllb_code, "en")

    try:
        tts = gTTS(text=text, lang=gtts_lang, slow=slow)
    except Exception:
        # Fallback to English TTS if the target language isn't supported
        logger.warning(f"gTTS failed for lang={gtts_lang}, falling back to 'en'")
        tts = gTTS(text=text, lang="en", slow=slow)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tts.save(tmp.name)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
    finally:
        os.unlink(tmp_path)

    return audio_bytes


def synthesize_to_file(text: str, lang_nllb_code: str, output_path: str) -> str:
    """Synthesize and save to a specific file path. Returns the path."""
    from gtts import gTTS
    gtts_lang = NLLB_TO_GTTS.get(lang_nllb_code, "en")
    tts = gTTS(text=text, lang=gtts_lang)
    tts.save(output_path)
    return output_path
