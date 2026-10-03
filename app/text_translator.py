"""
text_translator.py — Loads the NLLB translation model (base + optional LoRA adapter).
Exposes a translate() function used by the Streamlit UI and evaluation scripts.
"""
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level lazy cache
_base_tokenizer = None
_base_model = None
_finetuned_model = None   # base model with LoRA adapter merged/loaded
_finetuned_available = False


def _load_base_model():
    """Load and cache the base NLLB tokenizer + model."""
    global _base_tokenizer, _base_model
    if _base_tokenizer is not None:
        return _base_tokenizer, _base_model

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from app.config import MODEL_NAME, DEVICE

    logger.info(f"Loading base model {MODEL_NAME} on {DEVICE}...")
    _base_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    _base_model = _base_model.to(DEVICE)
    _base_model.eval()
    logger.info("Base model loaded.")
    return _base_tokenizer, _base_model


def _load_finetuned_model():
    """Load the base model + LoRA adapter. Returns None if adapter not found."""
    global _finetuned_model, _finetuned_available
    if _finetuned_model is not None:
        return _finetuned_model

    from app.config import LORA_ADAPTER_PATH, DEVICE

    adapter_path = Path(LORA_ADAPTER_PATH)
    if not adapter_path.exists():
        logger.warning(
            f"LoRA adapter not found at {adapter_path}. "
            "Run ml/finetune_lora.py to create it."
        )
        _finetuned_available = False
        return None

    try:
        from transformers import AutoModelForSeq2SeqLM
        from peft import PeftModel
        from app.config import MODEL_NAME

        logger.info(f"Loading fine-tuned model from {adapter_path}...")
        base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        _finetuned_model = PeftModel.from_pretrained(base, str(adapter_path))
        _finetuned_model = _finetuned_model.to(DEVICE)
        _finetuned_model.eval()
        _finetuned_available = True
        logger.info("Fine-tuned model loaded.")
        return _finetuned_model
    except Exception as e:
        logger.error(f"Failed to load fine-tuned model: {e}")
        _finetuned_available = False
        return None


def is_finetuned_available() -> bool:
    """Check (without loading) whether the LoRA adapter exists on disk."""
    from app.config import LORA_ADAPTER_PATH
    return Path(LORA_ADAPTER_PATH).exists()


def translate(
    text: str,
    src_lang: str,
    tgt_lang: str,
    use_finetuned: bool = False,
    max_new_tokens: Optional[int] = None,
) -> str:
    """
    Translate `text` from `src_lang` to `tgt_lang` using NLLB-200.

    Args:
        text:           Input text to translate.
        src_lang:       NLLB language code (e.g. "hin_Deva").
        tgt_lang:       NLLB language code (e.g. "eng_Latn").
        use_finetuned:  If True and adapter is available, use the LoRA model.
        max_new_tokens: Override for generation length.

    Returns:
        Translated string.
    """
    import torch
    from app.config import MAX_NEW_TOKENS, DEVICE

    max_tokens = max_new_tokens or MAX_NEW_TOKENS
    tokenizer, base_model = _load_base_model()

    # Choose model
    if use_finetuned:
        ft_model = _load_finetuned_model()
        model = ft_model if ft_model is not None else base_model
    else:
        model = base_model

    # Tokenize with source language forced BOS
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

    # Get forced BOS token id for target language
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=max_tokens,
            num_beams=4,
            early_stopping=True,
        )

    translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return translated.strip()
