"""
train_qlora.py - QLoRA fine-tuning of Mistral 7B for Text-to-SQL.

IMPORTANT - what this file is:
  The adapter committed in models/adapter/ was trained in the Colab notebook
  notebooks/LLM.ipynb. This script is that notebook's training code, cleaned up into one
  file with the same settings, so the training procedure is documented and repeatable.
  The committed adapter was NOT produced by running this script.

  By default this script writes to outputs/retrained_adapter/ so it can never
  overwrite models/adapter/.

Settings copied from the notebook:
  data        b-mc2/sql-create-context, first 5,500 rows -> 5,000 train / 500 validation (seed 42)
  quant       4-bit NF4, double quant, float16 compute (QLoRA)
  LoRA        r=16, alpha=32, dropout=0.05, targets q_proj/k_proj/v_proj/o_proj
  training    1 epoch, batch 1 x grad-accum 8 (effective 8), lr 2e-4 cosine, 100 warmup steps,
              max_length 1024, gradient checkpointing, no packing, seed 42
  hardware    Tesla T4 (Colab); the notebook recorded 625 steps in about 75 minutes

Usage (needs a GPU):
    python src/train_qlora.py
    python src/train_qlora.py --output-dir outputs/retrained_adapter
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompts import build_training_text  # noqa: E402

DATASET_NAME = "b-mc2/sql-create-context"
NUM_ROWS = 5500
VALIDATION_SIZE = 500
SEED = 42


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "retrained_adapter"))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    from model import load_base_model, load_tokenizer

    # ---- data: same selection and split as the notebook ----
    dataset = load_dataset(DATASET_NAME)
    subset = dataset["train"].select(range(NUM_ROWS))
    split = subset.train_test_split(test_size=VALIDATION_SIZE, seed=SEED)

    def add_text(example):
        example["text"] = build_training_text(example)
        return example

    train_data = split["train"].map(add_text)
    eval_data = split["test"].map(add_text)
    print(f"Training examples: {len(train_data)} | Validation examples: {len(eval_data)}")
    print("\nSample training text:\n" + train_data[0]["text"] + "\n")

    # ---- model: 4-bit base + LoRA adapter (= QLoRA) ----
    tokenizer = load_tokenizer()
    model = load_base_model(for_training=True)
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

    # ---- training config: same values as the notebook ----
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
        max_length=1024,
        dataset_text_field="text",
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Keep the training log: real numbers for the README.
    log_path = output_dir / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"Adapter saved to {output_dir}\nTraining log saved to {log_path}")


if __name__ == "__main__":
    main()
