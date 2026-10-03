#!/usr/bin/env python3
"""
train_lang_detector.py — Trains a custom scikit-learn language identifier.

Architecture:
  - Features:    Character n-gram TF-IDF (n=2–4, max_features=50,000)
  - Classifier:  Logistic Regression (multinomial, C=1.0)
  - Training:    500 samples per language × 10 languages = 5,000 total
  - Data source: HuggingFace Wikipedia datasets per language
  - Output:      ml/artifacts/lang_detector.pkl (sklearn Pipeline)

Supported languages:
  English, Hindi, French, Spanish, German, Chinese, Arabic, Portuguese, Russian, Tamil

Usage:
    python ml/train_lang_detector.py [--samples-per-lang N] [--seed S]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"

# ── Language → Wikipedia dataset config ──────────────────────────────────────
# HuggingFace `wikipedia` scripts are deprecated; use parquet-hosted dumps.
# Format: label -> (hf_dataset_name, hf_config, hf_split, text_column)
LANG_CONFIGS = {
    "English":    ("wikimedia/wikipedia", "20231101.en", "train", "text"),
    "Hindi":      ("wikimedia/wikipedia", "20231101.hi", "train", "text"),
    "French":     ("wikimedia/wikipedia", "20231101.fr", "train", "text"),
    "Spanish":    ("wikimedia/wikipedia", "20231101.es", "train", "text"),
    "German":     ("wikimedia/wikipedia", "20231101.de", "train", "text"),
    "Chinese":    ("wikimedia/wikipedia", "20231101.zh", "train", "text"),
    "Arabic":     ("wikimedia/wikipedia", "20231101.ar", "train", "text"),
    "Portuguese": ("wikimedia/wikipedia", "20231101.pt", "train", "text"),
    "Russian":    ("wikimedia/wikipedia", "20231101.ru", "train", "text"),
    "Tamil":      ("wikimedia/wikipedia", "20231101.ta", "train", "text"),
}


def collect_samples(label: str, config: tuple, n_samples: int, seed: int) -> list[str]:
    """Download Wikipedia text for one language and return n_samples sentences."""
    import random
    from datasets import load_dataset

    dataset_name, dataset_config, split, text_col = config

    print(f"   Downloading {label} ({dataset_name}/{dataset_config}) …", end=" ", flush=True)
    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)

    rng = random.Random(seed)
    samples = []
    for item in ds:
        text = item[text_col]
        # Split into sentences (rough), take non-trivial ones
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 30]
        samples.extend(sentences)
        if len(samples) >= n_samples * 10:  # oversample, then pick
            break

    # Pick n_samples randomly
    chosen = rng.sample(samples, min(n_samples, len(samples)))
    print(f"✅ {len(chosen)} samples")
    return chosen


def build_training_data(samples_per_lang: int, seed: int):
    """Collect samples for all languages and return (texts, labels)."""
    texts, labels = [], []
    for label, config in LANG_CONFIGS.items():
        lang_samples = collect_samples(label, config, samples_per_lang, seed)
        texts.extend(lang_samples)
        labels.extend([label] * len(lang_samples))
    return texts, labels


def train_classifier(texts: list[str], labels: list[str], seed: int):
    """Train TF-IDF + LogisticRegression pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import numpy as np

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=seed, stratify=labels
    )
    print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Build pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",   # character n-grams with word boundaries
            ngram_range=(2, 4),
            max_features=50_000,
            sublinear_tf=True,
            strip_accents=None,   # preserve Devanagari / Arabic / CJK characters
            lowercase=False,      # case-sensitive for script detection
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=seed,
        )),
    ])

    print("   Training TF-IDF + Logistic Regression …")
    pipeline.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'═'*55}")
    print(f"🎯  Language Detector Accuracy: {accuracy*100:.2f}%")
    print(f"{'═'*55}")
    print(classification_report(y_test, y_pred, target_names=sorted(set(labels))))

    # Confusion matrix
    classes = sorted(set(labels))
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    print("Confusion Matrix (rows=true, cols=pred):")
    print("         " + "  ".join(f"{c[:6]:>6}" for c in classes))
    for i, row_label in enumerate(classes):
        row_str = "  ".join(f"{cm[i,j]:>6}" for j in range(len(classes)))
        print(f"{row_label[:8]:>8} {row_str}")

    # Save confusion matrix as JSON for report
    cm_dict = {
        "classes": classes,
        "matrix": cm.tolist(),
        "accuracy": round(accuracy, 4),
    }
    with open(ARTIFACTS_DIR / "lang_detector_eval.json", "w") as f:
        json.dump(cm_dict, f, indent=2)

    return pipeline, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train sklearn language detector")
    parser.add_argument("--samples-per-lang", type=int, default=500, help="Samples per language")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"\n🚀 Training language detector ({args.samples_per_lang} samples/lang × 10 langs)\n")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Collect data ──────────────────────────────────────────────────────────
    print("📥 Collecting training data from Wikipedia …")
    texts, labels = build_training_data(args.samples_per_lang, args.seed)
    print(f"\n   Total samples: {len(texts):,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n🏋️  Training classifier …")
    pipeline, accuracy = train_classifier(texts, labels, args.seed)

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = ARTIFACTS_DIR / "lang_detector.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n✅ Model saved to {model_path}")
    print(f"   Accuracy: {accuracy*100:.2f}%")

    # Smoke-test Hinglish mixed-script handling via the app heuristic
    sys.path.insert(0, str(PROJECT_ROOT))
    from app.lang_detector import detect_language, is_hinglish

    mixed = "मैं today school जा रहा हूँ"
    hinglish_flag = is_hinglish(mixed)
    label, conf = detect_language("This is a completely English sentence.")
    print("\n🧪 Hinglish / detector smoke tests")
    print(f"   Mixed-script Hinglish: {mixed!r} → is_hinglish={hinglish_flag}")
    print(f"   Pure English: label={label} ({conf:.2f})")

    print("\n   Next step: python ml/prepare_dataset.py\n")


if __name__ == "__main__":
    main()
