"""
train_qlora.py - QLoRA fine-tuning of Mistral 7B for Text-to-SQL.

TWO RECIPES, ONE SCRIPT
  v1 (default)   The notebook recipe (notebooks/LLM.ipynb), cleaned up into one file. The adapter in
                 models/adapter/ was trained by the NOTEBOOK, not by this script.
  v2 (--augment) The same recipe and hyper-parameters, with ONE change: training schemas get
                 distractor tables and columns added (see augment.py), so the model learns to ignore
                 tables and columns a question does not need. Only the training rows change. The
                 500 validation rows stay untouched.

By default the script writes to outputs/retrained_adapter/, so it never overwrites models/adapter/.

Settings shared by both recipes (copied from the notebook):
  data        b-mc2/sql-create-context, first 5,500 rows -> 5,000 train / 500 validation (seed 42)
  quant       4-bit NF4, double quant, float16 compute (QLoRA)
  LoRA        r=16, alpha=32, dropout=0.05, targets q_proj/k_proj/v_proj/o_proj
  training    1 epoch, batch 1 x grad-accum 8 (effective 8), lr 2e-4 cosine, 100 warmup steps,
              max_length 1024, gradient checkpointing, no packing, seed 42
  hardware    the notebook used a Colab T4: 625 steps in about 75 minutes (v2 prompts are longer,
              so expect more; this has not been measured)

Usage (training needs a GPU):
    python src/train_qlora.py --augment --dry-run          # build + inspect the training data, no GPU
    python src/train_qlora.py --augment --output-dir /content/drive/MyDrive/text_to_sql_mistral_v2 --save-steps 100
    python src/train_qlora.py --augment --output-dir <same folder> --save-steps 100 --resume
                                                           # continue after a Colab disconnect
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from augment import Augmenter, parse_context  # noqa: E402
from prompts import build_training_text  # noqa: E402

DATASET_NAME = "b-mc2/sql-create-context"
NUM_ROWS = 5500
VALIDATION_SIZE = 500
SEED = 42
MAX_LENGTH = 1024
BASE_MODEL = "mistralai/Mistral-7B-v0.1"   # same value as model.BASE_MODEL (checked before training)


def try_load_tokenizer():
    """Tokenizer for length checks. Returns None if transformers is not installed."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(BASE_MODEL)
    except Exception as exc:  # noqa: BLE001
        print(f"(token lengths not checked: {type(exc).__name__}: {exc})")
        return None


def n_tables(context):
    parsed = parse_context(context)
    return len(parsed) if parsed else context.count("CREATE TABLE")


def build_training_rows(train_part, augment, tokenizer):
    """Return (rows with a 'text' field, stats). With augment=False this is the v1 data."""
    contexts = list(train_part["context"])
    augmenter = Augmenter(contexts, seed=SEED) if augment else None
    rows, statuses = [], Counter()
    before = after = too_long = 0
    for i, example in enumerate(train_part):
        row = dict(example)
        before += n_tables(row["context"])
        status = "plain"
        if augmenter:
            result = augmenter.augment_random(row["context"], row["answer"], row_id=i)
            status = result.status
            if result.status == "augmented":
                candidate = dict(row, context=result.context)
                if tokenizer is not None and len(tokenizer(build_training_text(candidate)).input_ids) > MAX_LENGTH - 16:
                    status, too_long = "reverted_too_long", too_long + 1   # would be truncated: keep original
                else:
                    row = candidate
        statuses[status] += 1
        after += n_tables(row["context"])
        row["text"] = build_training_text(row)
        rows.append(row)
    stats = {"recipe": "v2_augmented" if augment else "v1_plain", "rows": len(rows),
             "status_counts": dict(statuses),
             "avg_tables_per_schema_before": round(before / len(rows), 3),
             "avg_tables_per_schema_after": round(after / len(rows), 3),
             "rows_reverted_because_too_long": too_long,
             "token_lengths_checked": tokenizer is not None}
    return rows, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "retrained_adapter"))
    parser.add_argument("--augment", action="store_true", help="v2 recipe: add distractor tables/columns to training schemas")
    parser.add_argument("--dry-run", action="store_true", help="build and inspect the training data, then stop (no GPU)")
    parser.add_argument("--resume", action="store_true", help="resume from the last checkpoint in --output-dir")
    parser.add_argument("--save-steps", type=int, default=500, help="checkpoint every N steps (notebook: 500; use 100 on Colab)")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    from datasets import Dataset, load_dataset

    # ---- data: same selection and split as the notebook ----
    dataset = load_dataset(DATASET_NAME)
    subset = dataset["train"].select(range(NUM_ROWS))
    split = subset.train_test_split(test_size=VALIDATION_SIZE, seed=SEED)
    train_part, eval_part = split["train"], split["test"]

    tokenizer = try_load_tokenizer()
    train_rows, stats = build_training_rows(train_part, args.augment, tokenizer)
    eval_rows = []
    for example in eval_part:                      # validation rows are NEVER augmented
        row = dict(example)
        row["text"] = build_training_text(row)
        eval_rows.append(row)

    print(f"Recipe: {stats['recipe']} | training rows: {len(train_rows)} | validation rows: {len(eval_rows)}")
    print("Stats:", json.dumps(stats, indent=2))
    for k in (0, 1, 2):
        print(f"\n--- sample training text #{k} ---\n{train_rows[k]['text']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "training_recipe.json", "w", encoding="utf-8") as f:
        json.dump({"augment": args.augment, "stats": stats, "dataset": DATASET_NAME, "rows_used": NUM_ROWS,
                   "validation_size": VALIDATION_SIZE, "seed": SEED, "max_length": MAX_LENGTH,
                   "save_steps": args.save_steps}, f, indent=2)
    if args.dry_run:
        print(f"\nDry run finished. Nothing was trained. Recipe file: {output_dir / 'training_recipe.json'}")
        return

    # ---- model: 4-bit base + LoRA adapter (= QLoRA) ----
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    import model as model_module
    assert model_module.BASE_MODEL == BASE_MODEL, "BASE_MODEL differs between train_qlora.py and model.py"

    tokenizer = model_module.load_tokenizer()
    model = model_module.load_base_model(for_training=True)
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- training config: same values as the notebook (only save_steps is adjustable) ----
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        bf16=True,            # as in the original notebook
        fp16=False,
        gradient_checkpointing=True,
        max_length=MAX_LENGTH,
        dataset_text_field="text",
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(eval_rows),
        processing_class=tokenizer,
    )

    checkpoint = get_last_checkpoint(str(output_dir)) if args.resume else None
    print(f"Resuming from {checkpoint}" if checkpoint else "Starting training from scratch.")
    trainer.train(resume_from_checkpoint=checkpoint)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Keep the training log: real numbers for the README.
    log_path = output_dir / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"Adapter saved to {output_dir}\nTraining log saved to {log_path}")


if __name__ == "__main__":
    main()
