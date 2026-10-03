# GenAI Language Translator — ML-Grade Implementation Plan

## Overview

A resume-ready **Hinglish → English fine-tuned translation system** built on top of `facebook/nllb-200-distilled-600M`, with a full voice pipeline (Whisper STT → translation → gTTS TTS), a **custom-trained scikit-learn language identifier**, LoRA/PEFT fine-tuning, rigorous BLEU/chrF evaluation, and a Streamlit UI. The project is structured to be defended in an ML interview — every modeling choice is documented and evaluated.

**Fine-tuning target confirmed:** Hinglish (code-switched Hindi–English) → English

---

## User Review Required

> [!IMPORTANT]
> **Dataset choice** — We will use `cfilt/iitb-english-hindi` from HuggingFace as the primary Hindi↔English corpus, then synthesize a small Hinglish (code-switched) test set by applying a rule-based lexical substitution script to a subset of the Hindi sentences (replacing common Hindi words with English equivalents at random positions). This is a well-established technique in Hinglish NLP literature. If you'd prefer to use a different dataset or approach, let me know before I run `prepare_dataset.py`.

> [!IMPORTANT]
> **Hardware assumption** — LoRA fine-tuning of NLLB-200 600M will be run on CPU or Apple MPS (Metal) if you're on an M-series Mac. Full training on CPU will be slow (~2–6 hrs for 3 epochs over 3k examples). If you have a CUDA GPU or want to use a cloud notebook (Colab/Kaggle), say so and I'll add appropriate flags. Default: MPS/CPU.

> [!WARNING]
> **`ml/artifacts/` is gitignored** — adapter weights and sklearn `.pkl` files will not be committed. The `PROJECT_REPORT.md` will embed the evaluation numbers directly, so the report is self-contained even without the model files.

---

## Open Questions

No blocking questions remain — fine-tuning target confirmed, dataset approach is above for review, hardware assumption is stated. Proceeding to full plan.

---

## Proposed Changes

### 1 — Project scaffold

#### [NEW] `genai-translator/` (full directory tree)
Create the project root under `/Users/satyamgupta/Projects/genai-translator/` with the following skeleton:

```
genai-translator/
├── venv/
├── app/
│   ├── __init__.py
│   ├── ui.py
│   ├── text_translator.py
│   ├── speech_to_text.py
│   ├── text_to_speech.py
│   ├── lang_detector.py
│   └── config.py
├── ml/
│   ├── __init__.py
│   ├── prepare_dataset.py
│   ├── finetune_lora.py
│   ├── train_lang_detector.py
│   ├── evaluate.py
│   └── artifacts/        ← gitignored
├── tests/
│   └── test_translator.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
└── .gitignore
```

---

### 2 — Python environment

#### [NEW] `requirements.txt`
Pinned dependencies covering all pipeline stages:

| Package | Role |
|---|---|
| `transformers>=4.40` | NLLB model loading & inference |
| `peft>=0.10` | LoRA adapters |
| `datasets>=2.19` | HuggingFace dataset loading |
| `accelerate>=0.29` | Trainer backend |
| `sacrebleu>=2.4` | BLEU + chrF scoring |
| `faster-whisper>=1.0` | Whisper STT |
| `gTTS>=2.5` | Text-to-speech |
| `streamlit>=1.33` | UI |
| `scikit-learn>=1.4` | Language detector |
| `matplotlib>=3.8` | Eval charts |
| `sentencepiece>=0.1.99` | Tokenizer for NLLB |
| `torch>=2.2` | PyTorch backend |
| `soundfile`, `sounddevice` | Audio I/O |

---

### 3 — Configuration

#### [NEW] [`app/config.py`](file:///Users/satyamgupta/Projects/genai-translator/app/config.py)

Central config file holding:
- **SUPPORTED_LANGUAGES** dict mapping display names → NLLB language codes (8–10 languages: English, Hindi, French, Spanish, German, Chinese (Simplified), Arabic, Portuguese, Russian, Tamil)
- **MODEL_NAME**: `"facebook/nllb-200-distilled-600M"`
- **LORA_ADAPTER_PATH**: `"ml/artifacts/nllb_hinglish_adapter"`
- **LANG_DETECTOR_PATH**: `"ml/artifacts/lang_detector.pkl"`
- **WHISPER_MODEL_SIZE**: `"base"` (fast, accurate enough)
- **MAX_LENGTH**: 256
- **Device auto-detection** (CUDA → MPS → CPU)

---

### 4 — ML Pipeline: Dataset Preparation

#### [NEW] [`ml/prepare_dataset.py`](file:///Users/satyamgupta/Projects/genai-translator/ml/prepare_dataset.py)

**What it does:**
1. Downloads `cfilt/iitb-english-hindi` (≈1.6M sentence pairs) via `datasets.load_dataset`.
2. Samples 5,000 Hindi–English pairs (reproducible seed).
3. Applies a **Hinglish code-switching function**: for each Hindi sentence, randomly replaces 20–40% of Hindi tokens with their English equivalents using a lookup dictionary of ~200 common Hindi→English word pairs. This produces plausible Hinglish input.
4. Splits into train/val/test (70/15/15 = 3,500 / 750 / 750).
5. Saves to `ml/artifacts/hinglish_dataset/` as JSON lines.
6. Prints sample rows for review before any training begins.

**Why this approach:** True Hinglish parallel corpora are sparse; this synthesis method is standard in the Hinglish NLP literature and produces a controlled dataset where we *know* the ground truth English translation.

---

### 5 — ML Pipeline: LoRA Fine-tuning

#### [NEW] [`ml/finetune_lora.py`](file:///Users/satyamgupta/Projects/genai-translator/ml/finetune_lora.py)

**LoRA configuration:**
| Param | Value | Rationale |
|---|---|---|
| `r` (rank) | 16 | Good balance of expressiveness vs. parameter count |
| `lora_alpha` | 32 | Standard 2× rank |
| `target_modules` | `["q_proj", "v_proj"]` | Attention matrices — most impactful for seq2seq |
| `lora_dropout` | 0.05 | Light regularization |
| `task_type` | `SEQ_2_SEQ_LM` | NLLB is encoder-decoder |

**Training config:**
| Param | Value |
|---|---|
| Learning rate | 3e-4 |
| Batch size | 4 (gradient accumulation 4 → effective 16) |
| Epochs | 3 |
| Warmup steps | 100 |
| Eval strategy | `"epoch"` |
| Save strategy | `"epoch"` |
| `fp16` | False (MPS/CPU compat); enabled if CUDA detected |

**Outputs:** saves adapter to `ml/artifacts/nllb_hinglish_adapter/`, logs training loss per epoch.

---

### 6 — ML Pipeline: Evaluation

#### [NEW] [`ml/evaluate.py`](file:///Users/satyamgupta/Projects/genai-translator/ml/evaluate.py)

**What it produces:**
1. **BLEU and chrF scores** for base model vs. fine-tuned model on the held-out 750-sentence test set (via `sacrebleu`).
2. **Bar chart** (`ml/artifacts/eval_comparison.png`) showing BLEU/chrF side-by-side.
3. **Qualitative examples table** (10 rows): Input Hinglish | Base output | Fine-tuned output | Reference English — saved as `ml/artifacts/qualitative_examples.md`.
4. **JSON summary** (`ml/artifacts/eval_results.json`) for embedding in `PROJECT_REPORT.md`.

---

### 7 — ML Pipeline: Language Detector

#### [NEW] [`ml/train_lang_detector.py`](file:///Users/satyamgupta/Projects/genai-translator/ml/train_lang_detector.py)

**Architecture:**
- **Features**: character n-gram TF-IDF (n=2–4, max_features=50,000) — captures script-level patterns (Devanagari, Latin, Arabic, CJK)
- **Classifier**: Logistic Regression (C=1.0, max_iter=1000, multi_class='multinomial')
- **Training data**: 500 samples per language × 10 languages = 5,000 total, sourced from the same HuggingFace datasets used for translation (Wikipedia text for each language via `datasets`)
- **Split**: 80/20 train/test
- **Output**: saves `ml/artifacts/lang_detector.pkl` (sklearn Pipeline), prints accuracy + confusion matrix

**Inference wrapper** ([`app/lang_detector.py`](file:///Users/satyamgupta/Projects/genai-translator/app/lang_detector.py)):
- Loads the `.pkl` at startup
- Exposes `detect_language(text) -> (lang_code, confidence)` used by the UI

---

### 8 — Core App: Text Translation

#### [NEW] [`app/text_translator.py`](file:///Users/satyamgupta/Projects/genai-translator/app/text_translator.py)

- Loads the **base NLLB model** at startup (cached via `@st.cache_resource`)
- Optionally loads the **LoRA adapter** on top if `LORA_ADAPTER_PATH` exists (graceful fallback to base if adapter missing)
- Exposes `translate(text, src_lang, tgt_lang, use_finetuned=False) -> str`
- Handles NLLB language code mapping from config

---

### 9 — Core App: Voice Pipeline

#### [NEW] [`app/speech_to_text.py`](file:///Users/satyamgupta/Projects/genai-translator/app/speech_to_text.py)
- Uses `faster-whisper` (`WhisperModel("base")`) to transcribe uploaded audio
- Returns `(transcribed_text, detected_language)`

#### [NEW] [`app/text_to_speech.py`](file:///Users/satyamgupta/Projects/genai-translator/app/text_to_speech.py)
- Uses `gTTS` to synthesize translated text
- Saves to a temp file, returns path for Streamlit audio player
- Maps NLLB language codes → gTTS language codes

---

### 10 — UI

#### [NEW] [`app/ui.py`](file:///Users/satyamgupta/Projects/genai-translator/app/ui.py)

**Tab 1 — Text Translation:**
- Text area for input
- Language **auto-detected** via `lang_detector.py` with confidence badge shown
- Manual override dropdown (all 10 supported languages)
- Target language selector
- Toggle: "Use fine-tuned model (Hinglish→EN)" — only enabled when source is detected as Hinglish/Hindi and target is English
- Side-by-side comparison mode: shows base model vs. fine-tuned output when toggle is active
- Translate button → translated text output

**Tab 2 — Voice Translation:**
- Upload `.wav`/`.mp3` or record via `st.audio_input` (Streamlit 1.33+)
- Whisper transcription shown
- Auto-detected source language
- Target language selector
- Translated text displayed
- Audio playback of gTTS output

**Styling:** Custom CSS injected via `st.markdown()` for dark glassmorphism theme, gradient header, animated translate button.

---

### 11 — Tests

#### [NEW] [`tests/test_translator.py`](file:///Users/satyamgupta/Projects/genai-translator/tests/test_translator.py)

- `test_config_languages()` — validates all NLLB codes are valid
- `test_lang_detector_loads()` — pkl loads without error
- `test_lang_detector_accuracy()` — spot-checks 5 obvious cases (English, French, Hindi script)
- `test_translate_en_fr()` — English→French, checks output is non-empty string
- `test_translate_hinglish()` — passes a Hinglish sentence, checks English output

---

### 12 — Documentation

#### [NEW] `README.md`
- Project overview, badges (Python version, license)
- Architecture diagram (ASCII + description) showing:
  - **Inference path**: Input → Lang Detector → NLLB (base or fine-tuned) → gTTS → Output
  - **Training path**: IITB corpus → code-switch synthesis → LoRA fine-tuning → evaluation → deployed adapter
- Setup instructions (venv, pip install, model download)
- Run instructions (`streamlit run app/ui.py`)
- ML scripts usage (`python ml/prepare_dataset.py`, etc.)

#### [NEW] `PROJECT_REPORT.md`
- Dataset section (IITB-HiEn, Hinglish synthesis method, splits)
- Fine-tuning section (LoRA config table, training time, hardware)
- Evaluation section (BLEU/chrF table with actual numbers, embedded chart, qualitative examples)
- Language detector section (TF-IDF features, accuracy, confusion matrix)
- Honest limitations (small synthetic dataset, CPU training constraints, no true Hinglish corpus)

---

## Verification Plan

### Automated Tests
```bash
cd genai-translator
source venv/bin/activate
python -m pytest tests/ -v
```

### Manual Verification (step-by-step)
1. `python ml/prepare_dataset.py` → review printed sample rows before training
2. `python ml/finetune_lora.py` → confirm loss decreases across epochs
3. `python ml/evaluate.py` → confirm BLEU/chrF table prints and chart saves
4. `python ml/train_lang_detector.py` → confirm accuracy ≥ 85% on test split
5. `streamlit run app/ui.py` → test all tabs, auto-detection, voice round-trip
6. Final: embed actual eval numbers into `PROJECT_REPORT.md`

### Expected Outcomes
| Metric | Target |
|---|---|
| Lang detector accuracy | ≥ 90% (character n-grams on 10 languages are very discriminative) |
| BLEU improvement (Hinglish→EN) | +2–8 points over base (realistic for 3k fine-tuning examples) |
| chrF improvement | +3–10 points |
| UI: all 10 languages | Working end-to-end |
| Voice pipeline | WAV upload → translated audio playback |

> [!NOTE]
> If fine-tuning does NOT improve BLEU (possible with synthetic data), the evaluation section will honestly report this and analyze root causes (data quality, synthesis artifacts, insufficient epochs). A well-explained negative result is explicitly part of good ML practice and is documented as such in the brief.
