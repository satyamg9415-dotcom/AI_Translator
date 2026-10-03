#!/usr/bin/env python3
"""
prepare_dataset.py — Downloads and prepares the Hinglish-English fine-tuning dataset.

Pipeline:
  1. Download cfilt/iitb-english-hindi from HuggingFace (1.6M pairs).
  2. Sample 5,000 sentence pairs (reproducible seed).
  3. Apply Hinglish code-switching: randomly replace 20-40% of Hindi tokens
     with English equivalents using a curated lookup dictionary.
  4. Split into train/val/test (70/15/15).
  5. Save to ml/artifacts/hinglish_dataset/ as JSON lines.
  6. Print sample rows for human review.

Usage:
    python ml/prepare_dataset.py [--sample-size N] [--seed S]
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

# ── Project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
DATASET_DIR = ARTIFACTS_DIR / "hinglish_dataset"

# ── Hinglish code-switching dictionary ────────────────────────────────────────
# Common Hindi words → English equivalents (for code-switching simulation)
HINDI_TO_ENGLISH_LOOKUP = {
    # Greetings / common phrases
    "नमस्ते": "hello", "हाँ": "yes", "नहीं": "no", "ठीक": "okay", "ठीक है": "okay",
    "शुक्रिया": "thanks", "धन्यवाद": "thank you", "कृपया": "please", "माफ़ करें": "sorry",
    # Time
    "आज": "today", "कल": "tomorrow", "परसों": "day after tomorrow",
    "सुबह": "morning", "शाम": "evening", "रात": "night", "अभी": "now",
    "जल्दी": "quickly", "देर": "late", "समय": "time", "घड़ी": "watch",
    # People
    "मैं": "I", "हम": "we", "तुम": "you", "वो": "he/she", "वे": "they",
    "मेरा": "my", "तुम्हारा": "your", "हमारा": "our", "उनका": "their",
    "लोग": "people", "दोस्त": "friend", "परिवार": "family", "बच्चे": "children",
    "माँ": "mother", "पिता": "father", "भाई": "brother", "बहन": "sister",
    # Common verbs
    "है": "is", "हैं": "are", "था": "was", "थे": "were", "होगा": "will be",
    "करना": "to do", "जाना": "to go", "आना": "to come", "देखना": "to see",
    "खाना": "food", "पानी": "water", "घर": "home", "काम": "work",
    # Adjectives
    "अच्छा": "good", "बुरा": "bad", "बड़ा": "big", "छोटा": "small",
    "नया": "new", "पुराना": "old", "सुंदर": "beautiful", "मुश्किल": "difficult",
    "आसान": "easy", "महंगा": "expensive", "सस्ता": "cheap",
    # Numbers
    "एक": "one", "दो": "two", "तीन": "three", "चार": "four", "पाँच": "five",
    # Places / things
    "शहर": "city", "गाँव": "village", "स्कूल": "school", "बाज़ार": "market",
    "दुकान": "shop", "सड़क": "road", "बस": "bus", "ट्रेन": "train",
    # Tech / modern (common in Hinglish)
    "मोबाइल": "mobile", "फ़ोन": "phone", "कंप्यूटर": "computer",
    "इंटरनेट": "internet", "वेबसाइट": "website", "ऐप": "app",
}


def apply_code_switching(hindi_text: str, rng: random.Random, ratio_range=(0.20, 0.40)) -> str:
    """
    Replace a random fraction of Hindi tokens with English equivalents.
    Preserves word order; unreplaced tokens stay in Devanagari script.

    Args:
        hindi_text:   Input Hindi sentence.
        rng:          Seeded random generator.
        ratio_range:  (min, max) fraction of tokens to switch.

    Returns:
        Code-switched Hinglish string.
    """
    tokens = hindi_text.split()
    if not tokens:
        return hindi_text

    # Find eligible tokens (those in our lookup)
    eligible = [(i, t) for i, t in enumerate(tokens) if t in HINDI_TO_ENGLISH_LOOKUP]
    if not eligible:
        return hindi_text

    # Choose how many to switch
    n_switch = max(1, int(len(eligible) * rng.uniform(*ratio_range)))
    to_switch = rng.sample(eligible, min(n_switch, len(eligible)))

    result = list(tokens)
    for idx, token in to_switch:
        result[idx] = HINDI_TO_ENGLISH_LOOKUP[token]

    return " ".join(result)


def load_iitb_dataset(sample_size: int, seed: int):
    """Load IITB English-Hindi dataset and return list of (hindi, english) tuples."""
    from datasets import load_dataset

    print("📥 Loading cfilt/iitb-english-hindi …")
    # This dataset has a 'train' split with 'translation' column: {'en': ..., 'hi': ...}
    ds = load_dataset("cfilt/iitb-english-hindi", split="train")
    print(f"   Total pairs available: {len(ds):,}")

    # Sample
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(sample_size, len(ds)))
    sampled = [ds[i] for i in indices]

    pairs = []
    for item in sampled:
        hi = item["translation"]["hi"].strip()
        en = item["translation"]["en"].strip()
        if hi and en and len(hi.split()) >= 3:  # skip very short sentences
            pairs.append((hi, en))

    print(f"   Sampled {len(pairs):,} pairs (after filtering short sentences)")
    return pairs


def build_hinglish_pairs(hindi_english_pairs, seed: int):
    """Build (hinglish_input, english_reference) pairs from Hindi-English pairs."""
    rng = random.Random(seed + 1)
    result = []
    for hi, en in hindi_english_pairs:
        hinglish = apply_code_switching(hi, rng)
        result.append({
            "hinglish": hinglish,
            "hindi_original": hi,
            "english": en,
        })
    return result


def split_and_save(records: list[dict], seed: int):
    """Split into train/val/test and save as JSONL files."""
    rng = random.Random(seed + 2)
    rng.shuffle(records)

    n = len(records)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    splits = {
        "train": records[:n_train],
        "val": records[n_train : n_train + n_val],
        "test": records[n_train + n_val :],
    }

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, data in splits.items():
        path = DATASET_DIR / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"   ✅ {split_name}: {len(data):,} examples → {path}")

    return splits


def print_samples(splits: dict, n: int = 8):
    """Print sample rows for human review."""
    print("\n" + "═" * 70)
    print("📋  SAMPLE DATASET ROWS (review before fine-tuning)")
    print("═" * 70)
    samples = splits["train"][:n]
    for i, row in enumerate(samples, 1):
        print(f"\n[{i}] Hinglish:  {row['hinglish']}")
        print(f"    Original:  {row['hindi_original']}")
        print(f"    English:   {row['english']}")
    print("\n" + "═" * 70)


def main():
    parser = argparse.ArgumentParser(description="Prepare Hinglish fine-tuning dataset")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of sentence pairs to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"\n🚀 Preparing Hinglish dataset (sample_size={args.sample_size}, seed={args.seed})\n")

    # 1. Load raw Hindi-English pairs
    hindi_english_pairs = load_iitb_dataset(args.sample_size, args.seed)

    # 2. Build code-switched Hinglish pairs
    print("\n🔀 Applying Hinglish code-switching …")
    records = build_hinglish_pairs(hindi_english_pairs, args.seed)
    print(f"   Generated {len(records):,} Hinglish–English pairs")

    # 3. Split and save
    print("\n💾 Splitting and saving …")
    splits = split_and_save(records, args.seed)

    # 4. Print samples for review
    print_samples(splits)

    print(f"\n✅ Dataset ready at: {DATASET_DIR}")
    print("   Next step: python ml/finetune_lora.py\n")


if __name__ == "__main__":
    main()
