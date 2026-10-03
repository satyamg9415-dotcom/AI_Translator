"""
speech_to_text.py — Whisper-based speech transcription using faster-whisper.
Returns transcribed text and detected language code.
"""
import logging
import tempfile
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_whisper_model = None


def _load_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    from faster_whisper import WhisperModel
    from app.config import WHISPER_MODEL_SIZE, DEVICE

    # faster-whisper device: cuda / cpu (MPS not supported, fallback to cpu)
    device = DEVICE if DEVICE == "cuda" else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    logger.info(f"Loading Whisper {WHISPER_MODEL_SIZE} on {device}...")
    _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)
    logger.info("Whisper loaded.")
    return _whisper_model


def transcribe(audio_bytes: bytes, audio_format: str = "wav") -> tuple[str, str]:
    """
    Transcribe audio bytes.

    Args:
        audio_bytes: Raw audio file bytes.
        audio_format: File extension hint (wav, mp3, m4a …)

    Returns:
        (transcribed_text, detected_language_code)
        e.g. ("Hello world", "en")
    """
    model = _load_whisper()

    # Write to a temp file so faster-whisper can read it
    with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(tmp_path, beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments)
        lang = info.language  # e.g. "en", "hi", "fr"
        return text.strip(), lang
    finally:
        os.unlink(tmp_path)


def transcribe_file(file_path: str) -> tuple[str, str]:
    """Transcribe from a file path (convenience wrapper)."""
    model = _load_whisper()
    segments, info = model.transcribe(file_path, beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip(), info.language
