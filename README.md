# 🌐 GenAI Language Translator

> A resume-ready, ML-grade translation system featuring LoRA fine-tuning, rigorous BLEU/chrF evaluation, and a custom-trained language identifier — built to be defended in an ML interview.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/Model-NLLB--200-FFD21E?logo=huggingface)](https://huggingface.co/facebook/nllb-200-distilled-600M)

---

## ✨ Features

| Feature | Details |
|---|---|
| **10 languages** | EN, HI, FR, ES, DE, ZH, AR, PT, RU, TA |
| **Text translation** | NLLB-200 600M (base + LoRA fine-tuned) |
| **Voice pipeline** | Whisper STT → Translation → gTTS TTS |
| **Auto language detection** | Custom-trained sklearn classifier (char n-gram TF-IDF) |
| **Hinglish fine-tuning** | LoRA adapter on code-switched Hindi–English corpus |
| **Side-by-side comparison** | Base vs. fine-tuned output for Hinglish→EN |
| **BLEU/chrF evaluation** | Rigorous held-out test set scoring |

---

## 🏗️ Architecture

### Inference Pipeline
```
User Input (text or audio)
        │
        ▼
┌───────────────────┐
│  Lang Detector    │  ← Custom sklearn TF-IDF + LogReg
│  (auto-detect)    │    Trained on Wikipedia (10 langs)
└────────┬──────────┘
         │  lang label + confidence
         ▼
┌───────────────────────────────┐
│    Text Translator            │
│  ┌──────────┐ ┌─────────────┐ │
│  │ Base     │ │ Fine-tuned  │ │  ← LoRA adapter (Hinglish→EN)
│  │ NLLB-200 │ │ NLLB+LoRA  │ │
│  └──────────┘ └─────────────┘ │
└──────────────┬────────────────┘
               │  translated text
               ▼
        ┌──────────────┐
        │   gTTS TTS   │  ← optional audio synthesis
        └──────┬───────┘
               ▼
         Audio Output
```

### Training Pipeline
```
IITB English–Hindi Corpus (cfilt/iitb-english-hindi)
         │
         ▼ prepare_dataset.py
┌────────────────────────────────────────┐
│   Hinglish Code-Switching Synthesis    │
│   (replace 20–40% Hindi tokens with   │
│    English equivalents via lookup)     │
└───────────────────┬────────────────────┘
                    │  5k pairs (train/val/test 70/15/15)
                    ▼
           finetune_lora.py
┌─────────────────────────────────────────┐
│   LoRA Fine-tuning (PEFT)               │
│   rank=16, alpha=32, target=[q,v]_proj  │
│   3 epochs, lr=3e-4, eff. batch=16      │
└───────────────────┬─────────────────────┘
                    │  adapter weights
                    ▼
              evaluate.py
┌─────────────────────────────────────────┐
│   BLEU + chrF comparison                │
│   base vs. fine-tuned on test split     │
│   → bar chart + qualitative examples    │
└─────────────────────────────────────────┘
```

---

## 🚀 Setup

### 1. Clone and create virtual environment
```bash
git clone https://github.com/yourusername/genai-translator.git
cd genai-translator

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the app (baseline — no ML training required)
```bash
streamlit run app/ui.py
```
The base NLLB-200 model will be downloaded automatically on first run (~2.4 GB).

---

## 🔬 ML Training (Step-by-Step)

### Step 1 — Prepare dataset
```bash
python ml/prepare_dataset.py
# Downloads IITB Hindi-English corpus, synthesizes Hinglish, saves 5k pairs
# Prints sample rows — review before proceeding
```

### Step 2 — Fine-tune with LoRA
```bash
python ml/finetune_lora.py
# LoRA rank=16, 3 epochs. CPU: ~3-6 hrs. GPU: ~20-40 min.
# Saves adapter to ml/artifacts/nllb_hinglish_adapter/
```

### Step 3 — Evaluate
```bash
python ml/evaluate.py
# Computes BLEU/chrF for base vs. fine-tuned
# Saves chart to ml/artifacts/eval_comparison.png
# Saves qualitative examples to ml/artifacts/qualitative_examples.md
```

### Step 4 — Train language detector
```bash
python ml/train_lang_detector.py
# Trains TF-IDF + LogReg on Wikipedia data (10 languages)
# Saves model to ml/artifacts/lang_detector.pkl
# Prints accuracy + confusion matrix
```

### Step 5 — Run full app with all components
```bash
streamlit run app/ui.py
# Auto-detection now active, fine-tuned model loaded
```

---

## 🧪 Tests

```bash
# Fast tests (no model download required)
python -m pytest tests/ -v -m "not slow"

# All tests including model inference
python -m pytest tests/ -v
```

---

## 📁 Project Structure

```
genai-translator/
├── app/
│   ├── ui.py               ← Streamlit app (dark glassmorphism UI)
│   ├── text_translator.py  ← NLLB inference (base + LoRA adapter)
│   ├── speech_to_text.py   ← Whisper STT
│   ├── text_to_speech.py   ← gTTS TTS
│   ├── lang_detector.py    ← Custom sklearn detector (inference wrapper)
│   └── config.py           ← All config: models, languages, paths
├── ml/
│   ├── prepare_dataset.py  ← IITB download + Hinglish synthesis
│   ├── finetune_lora.py    ← LoRA fine-tuning script
│   ├── train_lang_detector.py ← sklearn training
│   ├── evaluate.py         ← BLEU/chrF evaluation + charts
│   └── artifacts/          ← (gitignored) adapter weights, .pkl, results
├── tests/
│   └── test_translator.py
├── requirements.txt
├── README.md
└── PROJECT_REPORT.md
```

---

## 📊 Key Results

See [PROJECT_REPORT.md](PROJECT_REPORT.md) for full evaluation details including:
- BLEU/chrF comparison table and bar chart
- 10 qualitative translation examples (Hinglish | Base | Fine-tuned | Reference)
- Language detector accuracy and confusion matrix
- Honest limitations and future work

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Translation | `facebook/nllb-200-distilled-600M` |
| Fine-tuning | PEFT LoRA via `peft` + `transformers` Trainer |
| STT | `faster-whisper` (Whisper base) |
| TTS | `gTTS` |
| UI | `streamlit` |
| Lang Detection | `scikit-learn` TF-IDF + LogisticRegression |
| Evaluation | `sacrebleu` (BLEU + chrF) |
| Visualization | `matplotlib` |
| Dataset | `cfilt/iitb-english-hindi` via HuggingFace Datasets |
