#!/usr/bin/env python3
"""
finetune_lora.py — LoRA fine-tuning of NLLB-200 on the Hinglish→English dataset.

Architecture:
  - Base model: facebook/nllb-200-distilled-600M (seq2seq)
  - Adapter:    LoRA rank=16, target_modules=["q_proj","v_proj"], alpha=32
  - Framework:  HuggingFace PEFT + Transformers Trainer
  - Output:     ml/artifacts/nllb_hinglish_adapter/

Prerequisites:
  Run ml/prepare_dataset.py first.

Usage:
    python ml/finetune_lora.py [--epochs N] [--batch-size N] [--lr LR] [--max-steps N]
"""
import argparse
import json
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
DATASET_DIR = ARTIFACTS_DIR / "hinglish_dataset"
ADAPTER_DIR = ARTIFACTS_DIR / "nllb_hinglish_adapter"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_hf_dataset(train_data, val_data, tokenizer, max_length: int = 128):
    """Tokenize and return HuggingFace datasets."""
    from datasets import Dataset

    def tokenize(batch):
        tokenizer.src_lang = "hin_Deva"
        tokenizer.tgt_lang = "eng_Latn"
        model_inputs = tokenizer(
            batch["hinglish"],
            text_target=batch["english"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        # Replace padding token ids in labels with -100 (ignored in loss)
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in lbl]
            for lbl in model_inputs["labels"]
        ]
        return model_inputs

    train_ds = Dataset.from_list([{"hinglish": r["hinglish"], "english": r["english"]} for r in train_data])
    val_ds   = Dataset.from_list([{"hinglish": r["hinglish"], "english": r["english"]} for r in val_data])

    print(f"   Tokenizing {len(train_ds)} train + {len(val_ds)} val examples …")
    train_ds = train_ds.map(tokenize, batched=True, batch_size=64, remove_columns=["hinglish", "english"])
    val_ds   = val_ds.map(tokenize, batched=True, batch_size=64, remove_columns=["hinglish", "english"])
    train_ds.set_format("torch")
    val_ds.set_format("torch")
    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune NLLB-200 on Hinglish→English")
    parser.add_argument("--epochs",     type=int,   default=3,    help="Number of training epochs")
    parser.add_argument("--batch-size", type=int,   default=4,    help="Per-device batch size")
    parser.add_argument("--lr",         type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max-steps",  type=int,   default=-1,   help="Override epochs with max steps (-1 = off)")
    parser.add_argument("--lora-rank",  type=int,   default=16,   help="LoRA rank r")
    parser.add_argument("--lora-alpha", type=int,   default=32,   help="LoRA alpha")
    args = parser.parse_args()

    # ── Check dataset exists ──────────────────────────────────────────────────
    if not (DATASET_DIR / "train.jsonl").exists():
        print("❌ Dataset not found. Run: python ml/prepare_dataset.py")
        sys.exit(1)

    print("\n🚀 Starting LoRA fine-tuning of NLLB-200 on Hinglish→English\n")
    print(f"   LoRA rank={args.lora_rank}, alpha={args.lora_alpha}, lr={args.lr}, epochs={args.epochs}")

    # ── Load data ─────────────────────────────────────────────────────────────
    train_data = load_jsonl(DATASET_DIR / "train.jsonl")
    val_data   = load_jsonl(DATASET_DIR / "val.jsonl")
    print(f"   Train: {len(train_data):,} | Val: {len(val_data):,}")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
    from peft import get_peft_model, LoraConfig, TaskType
    import torch

    MODEL_NAME = "facebook/nllb-200-distilled-600M"
    print(f"\n📦 Loading tokenizer: {MODEL_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.src_lang = "hin_Deva"
    tokenizer.tgt_lang = "eng_Latn"

    # ── Tokenize ─────────────────────────────────────────────────────────────
    print("🔢 Tokenizing dataset …")
    train_ds, val_ds = build_hf_dataset(train_data, val_data, tokenizer)

    # ── Load base model ───────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"\n📦 Loading base model on {device} …")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.to(device)

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    trainable_params, all_params = model.get_nb_trainable_parameters()
    trainable_ratio = 100.0 * trainable_params / all_params if all_params else 0.0

    # ── Training arguments ────────────────────────────────────────────────────
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(ADAPTER_DIR),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,          # effective batch = 16
        learning_rate=args.lr,
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        predict_with_generate=False,            # speed: disable during training
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,               # safe for macOS/CPU
        report_to="none",                       # no wandb
        save_total_limit=2,
        use_cpu=device == "cpu",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n🏋️  Training …")
    train_result = trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────────────
    print(f"\n💾 Saving LoRA adapter to {ADAPTER_DIR} …")
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    # ── Save training summary ─────────────────────────────────────────────────
    summary = {
        "model": MODEL_NAME,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.05,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": args.batch_size * 4,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "device": device,
        "trainable_params": int(trainable_params),
        "all_params": int(all_params),
        "trainable_ratio_pct": round(trainable_ratio, 4),
        "train_runtime_seconds": train_result.metrics.get("train_runtime", 0),
        "train_loss": train_result.metrics.get("train_loss", 0),
        "log_history": trainer.state.log_history,
    }

    with open(ARTIFACTS_DIR / "finetune_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Fine-tuning complete!")
    print(f"   Train loss:    {train_result.metrics.get('train_loss', 'N/A'):.4f}")
    print(f"   Training time: {train_result.metrics.get('train_runtime', 0)/60:.1f} min")
    print(f"   Adapter saved: {ADAPTER_DIR}")
    print("\n   Next step: python ml/evaluate.py\n")


if __name__ == "__main__":
    main()
