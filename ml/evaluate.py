#!/usr/bin/env python3
"""
evaluate.py — Rigorous evaluation of base vs. fine-tuned NLLB model.

Produces:
  1. BLEU and chrF scores on the held-out test set (sacrebleu).
  2. Bar chart comparing base vs. fine-tuned (matplotlib).
  3. Qualitative examples table (10 rows): Hinglish | Base | Fine-tuned | Reference.
  4. JSON summary for PROJECT_REPORT.md.

Prerequisites:
  - Run ml/prepare_dataset.py (for test.jsonl)
  - Run ml/finetune_lora.py (for adapter; base-only evaluation also works without it)

Usage:
    python ml/evaluate.py [--n-examples N] [--max-test N]
"""
import argparse
import json
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
DATASET_DIR   = ARTIFACTS_DIR / "hinglish_dataset"
ADAPTER_DIR   = ARTIFACTS_DIR / "nllb_hinglish_adapter"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_inference(records: list[dict], use_finetuned: bool, tokenizer, model, device: str, max_new_tokens: int = 128) -> tuple[list[str], float]:
    """Run batch inference and return (translations, mean latency seconds)."""
    import time
    import torch
    predictions = []
    latencies = []

    TGT_LANG = "eng_Latn"
    forced_bos = tokenizer.convert_tokens_to_ids(TGT_LANG)

    model.eval()
    for i, record in enumerate(records):
        if i % 50 == 0:
            print(f"   {'FT' if use_finetuned else 'BASE'} inference: {i}/{len(records)}")

        tokenizer.src_lang = "hin_Deva"
        enc = tokenizer(
            record["hinglish"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **enc,
                forced_bos_token_id=forced_bos,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
            )
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)
        pred = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
        predictions.append(pred)

    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return predictions, mean_latency


def compute_scores(predictions: list[str], references: list[str]) -> dict:
    """Compute BLEU and chrF using sacrebleu."""
    import sacrebleu

    refs_wrapped = [[r] for r in references]  # sacrebleu expects list of reference lists

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, refs_wrapped)

    return {
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
    }


def make_bar_chart(base_scores: dict, ft_scores: dict, output_path: Path):
    """Generate a side-by-side bar chart and save to PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ["BLEU", "chrF"]
    base_vals = [base_scores["bleu"], base_scores["chrf"]]
    ft_vals   = [ft_scores["bleu"],   ft_scores["chrf"]]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    bars_base = ax.bar(x - width/2, base_vals, width, label="Base NLLB-200",
                       color="#3b82f6", alpha=0.85, edgecolor="#60a5fa", linewidth=1.2)
    bars_ft   = ax.bar(x + width/2, ft_vals,   width, label="Fine-tuned (LoRA)",
                       color="#a855f7", alpha=0.85, edgecolor="#c084fc", linewidth=1.2)

    # Value labels on bars
    for bar in bars_base:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom",
                color="#93c5fd", fontsize=11, fontweight="bold")
    for bar in bars_ft:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom",
                color="#d8b4fe", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color="#e2e8f0", fontsize=13)
    ax.set_ylabel("Score", color="#94a3b8", fontsize=11)
    ax.set_title("Base vs. Fine-tuned NLLB-200\nHinglish → English Translation Quality",
                 color="#e2e8f0", fontsize=13, pad=15)
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=10)
    ax.tick_params(colors="#64748b")
    ax.spines[:].set_color("#1e293b")
    ax.yaxis.label.set_color("#94a3b8")
    ax.set_ylim(0, max(max(base_vals), max(ft_vals)) * 1.25)
    ax.grid(axis="y", color="#1e293b", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"   📊 Chart saved to {output_path}")


def save_qualitative_examples(records, base_preds, ft_preds, output_path: Path, n: int = 10):
    """Save qualitative comparison table as markdown."""
    lines = [
        "# Qualitative Translation Examples\n",
        "Hinglish (Code-switched) → English Translation\n",
        "Base model vs. Fine-tuned (LoRA) vs. Reference\n\n",
        "| # | Hinglish Input | Base Model | Fine-tuned | Reference |\n",
        "|---|---|---|---|---|\n",
    ]
    for i in range(min(n, len(records))):
        inp = records[i]["hinglish"].replace("|", "\\|")
        base = base_preds[i].replace("|", "\\|")
        ft   = ft_preds[i].replace("|", "\\|")
        ref  = records[i]["english"].replace("|", "\\|")
        lines.append(f"| {i+1} | {inp} | {base} | {ft} | {ref} |\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"   📝 Qualitative examples saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs. fine-tuned NLLB on Hinglish→EN")
    parser.add_argument("--n-examples", type=int, default=10,  help="Qualitative examples to show")
    parser.add_argument("--max-test",   type=int, default=200, help="Max test examples to evaluate (for speed)")
    args = parser.parse_args()

    # ── Check prerequisites ───────────────────────────────────────────────────
    test_path = DATASET_DIR / "test.jsonl"
    if not test_path.exists():
        print("❌ Test set not found. Run: python ml/prepare_dataset.py")
        sys.exit(1)

    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import PeftModel

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    MODEL_NAME = "facebook/nllb-200-distilled-600M"

    # ── Load test data ────────────────────────────────────────────────────────
    all_test = load_jsonl(test_path)
    test_data = all_test[:args.max_test]
    print(f"\n🧪 Evaluating on {len(test_data)} test examples (from {len(all_test)} total)\n")

    references = [r["english"] for r in test_data]

    # ── Load tokenizer + base model ───────────────────────────────────────────
    print(f"📦 Loading base model: {MODEL_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    base_model = base_model.to(device)
    base_model.eval()

    # ── Base model inference ──────────────────────────────────────────────────
    print("\n🔵 Running BASE model inference …")
    base_preds, base_latency = run_inference(test_data, False, tokenizer, base_model, device)

    # ── Fine-tuned model inference ────────────────────────────────────────────
    ft_available = ADAPTER_DIR.exists() and (ADAPTER_DIR / "adapter_config.json").exists()
    if ft_available:
        print(f"\n🟣 Loading fine-tuned adapter from {ADAPTER_DIR} …")
        ft_model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
        ft_model = ft_model.to(device)
        ft_model.eval()
        print("   Running FINE-TUNED model inference …")
        ft_preds, ft_latency = run_inference(test_data, True, tokenizer, ft_model, device)
    else:
        print("\n⚠️  Fine-tuned adapter not found. Reporting base model scores only.")
        print("   Run ml/finetune_lora.py to generate the adapter.")
        ft_preds = base_preds  # same predictions as placeholder
        ft_latency = base_latency

    # ── Compute scores ────────────────────────────────────────────────────────
    print("\n📐 Computing BLEU and chrF scores …")
    base_scores = compute_scores(base_preds, references)
    ft_scores   = compute_scores(ft_preds,   references)

    print("\n" + "═" * 50)
    print("📊  EVALUATION RESULTS")
    print("═" * 50)
    print(f"{'Metric':<12} {'Base NLLB-200':>15} {'Fine-tuned':>15} {'Δ':>10}")
    print("─" * 50)
    for metric in ["bleu", "chrf"]:
        base_v = base_scores[metric]
        ft_v   = ft_scores[metric]
        delta  = ft_v - base_v
        sign   = "+" if delta >= 0 else ""
        print(f"{metric.upper():<12} {base_v:>15.2f} {ft_v:>15.2f} {sign}{delta:>9.2f}")
    print("═" * 50)

    if not ft_available:
        print("\n⚠️  Note: Fine-tuned model not available — run finetune_lora.py and re-evaluate.")

    # ── Save chart ────────────────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    chart_path = ARTIFACTS_DIR / "eval_comparison.png"
    make_bar_chart(base_scores, ft_scores, chart_path)

    # ── Save qualitative examples ─────────────────────────────────────────────
    qual_path = ARTIFACTS_DIR / "qualitative_examples.md"
    save_qualitative_examples(test_data, base_preds, ft_preds, qual_path, n=args.n_examples)

    # ── Save JSON summary ─────────────────────────────────────────────────────
    summary = {
        "test_examples": len(test_data),
        "base_model": MODEL_NAME,
        "finetuned_adapter": str(ADAPTER_DIR) if ft_available else None,
        "ft_available": ft_available,
        "base_scores": base_scores,
        "ft_scores": ft_scores,
        "base_mean_latency_seconds": round(base_latency, 4),
        "ft_mean_latency_seconds": round(ft_latency, 4),
        "improvement": {
            "bleu": round(ft_scores["bleu"] - base_scores["bleu"], 2),
            "chrf": round(ft_scores["chrf"] - base_scores["chrf"], 2),
        }
    }
    with open(ARTIFACTS_DIR / "eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n   💾 Results JSON: {ARTIFACTS_DIR / 'eval_results.json'}")

    print("\n✅ Evaluation complete. Next step: python ml/train_lang_detector.py\n")


if __name__ == "__main__":
    main()
