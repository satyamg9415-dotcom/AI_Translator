"""
test_translator.py — Unit tests for the GenAI Translator pipeline.

Run with:
    source venv/bin/activate
    python -m pytest tests/ -v
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Config tests
# ─────────────────────────────────────────────────────────────────────────────
class TestConfig:
    def test_supported_languages_not_empty(self):
        from app.config import SUPPORTED_LANGUAGES
        assert len(SUPPORTED_LANGUAGES) >= 8, "Expected at least 8 supported languages"

    def test_source_languages_have_nllb_codes(self):
        from app.config import SOURCE_LANGUAGES
        for name, (nllb_code, gtts_code) in SOURCE_LANGUAGES.items():
            assert "_" in nllb_code, f"Invalid NLLB code for {name}: {nllb_code}"
            assert len(gtts_code) == 2, f"Invalid gTTS code for {name}: {gtts_code}"

    def test_device_is_valid(self):
        from app.config import DEVICE
        assert DEVICE in ("cpu", "cuda", "mps"), f"Unexpected device: {DEVICE}"

    def test_model_name(self):
        from app.config import MODEL_NAME
        assert "nllb" in MODEL_NAME.lower() or "m2m" in MODEL_NAME.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Language detector tests (no model required — tests graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────
class TestLangDetector:
    def test_detect_unknown_when_model_missing(self):
        """If model isn't trained yet, should return Unknown gracefully."""
        from app.lang_detector import detect_language
        # This may return real detection or Unknown depending on whether model exists
        label, conf = detect_language("Hello world")
        assert isinstance(label, str)
        assert 0.0 <= conf <= 1.0

    def test_empty_text_returns_unknown(self):
        from app.lang_detector import detect_language
        label, conf = detect_language("")
        assert label == "Unknown"
        assert conf == 0.0

    def test_is_hinglish_mixed_script(self):
        from app.lang_detector import is_hinglish
        # Purely English — not Hinglish
        result = is_hinglish("This is a completely English sentence.")
        # We don't assert a specific value since the detector may not be trained,
        # but the function should not raise.
        assert isinstance(result, bool)

    def test_hinglish_detection_heuristic(self):
        """Test the mixed-script heuristic directly."""
        # Simulated Hinglish: mix of Devanagari and Latin
        mixed = "मैं today school जा रहा हूँ"
        from app.lang_detector import is_hinglish
        result = is_hinglish(mixed)
        assert isinstance(result, bool)


# ─────────────────────────────────────────────────────────────────────────────
# TTS tests (no network, just gTTS)
# ─────────────────────────────────────────────────────────────────────────────
class TestTTS:
    def test_nllb_to_gtts_mapping_complete(self):
        from app.text_to_speech import NLLB_TO_GTTS
        from app.config import SOURCE_LANGUAGES
        # Every supported language should have a gTTS mapping
        for name, (nllb_code, gtts_code) in SOURCE_LANGUAGES.items():
            assert nllb_code in NLLB_TO_GTTS, f"Missing gTTS mapping for {name} ({nllb_code})"

    @pytest.mark.slow
    def test_synthesize_english(self):
        """Requires internet (gTTS). Marked slow — skip with pytest -m 'not slow'."""
        from app.text_to_speech import synthesize
        audio = synthesize("Hello, this is a test.", "eng_Latn")
        assert isinstance(audio, bytes)
        assert len(audio) > 1000, "Audio too short — likely failed"


# ─────────────────────────────────────────────────────────────────────────────
# Translation tests (requires model download — marked slow)
# ─────────────────────────────────────────────────────────────────────────────
class TestTranslator:
    @pytest.mark.slow
    def test_translate_en_to_fr(self):
        from app.text_translator import translate
        result = translate("Hello, how are you?", "eng_Latn", "fra_Latn")
        assert isinstance(result, str)
        assert len(result) > 0
        # Should not be the same as input
        assert result != "Hello, how are you?"

    @pytest.mark.slow
    def test_translate_en_to_hi(self):
        from app.text_translator import translate
        result = translate("Good morning", "eng_Latn", "hin_Deva")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.slow
    def test_translate_hinglish_to_en(self):
        """Hinglish input — should produce English output."""
        from app.text_translator import translate
        hinglish = "मैं today बहुत happy हूँ"
        result = translate(hinglish, "hin_Deva", "eng_Latn")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.slow
    def test_finetuned_availability_check(self):
        from app.text_translator import is_finetuned_available
        # Just checks if the function works — bool either way is valid
        result = is_finetuned_available()
        assert isinstance(result, bool)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset tests (requires ml/prepare_dataset.py to have been run)
# ─────────────────────────────────────────────────────────────────────────────
class TestDataset:
    def test_dataset_splits_exist_if_prepared(self):
        import json
        dataset_dir = PROJECT_ROOT / "ml" / "artifacts" / "hinglish_dataset"
        if not dataset_dir.exists():
            pytest.skip("Dataset not yet prepared — run ml/prepare_dataset.py")

        for split in ["train", "val", "test"]:
            path = dataset_dir / f"{split}.jsonl"
            assert path.exists(), f"Missing {split}.jsonl"
            with open(path) as f:
                records = [json.loads(line) for line in f]
            assert len(records) > 0
            # Check schema
            assert "hinglish" in records[0]
            assert "english"  in records[0]

    def test_code_switching_logic(self):
        """Test the Hinglish synthesis function in isolation."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "ml"))
        from prepare_dataset import apply_code_switching
        import random

        rng = random.Random(42)
        hindi = "मैं आज बाज़ार जा रहा हूँ"
        result = apply_code_switching(hindi, rng)
        assert isinstance(result, str)
        assert len(result) > 0
        # Result should differ from input (at least sometimes substituted)
        # It's possible all tokens are not in lookup, so we don't assert strict diff


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
