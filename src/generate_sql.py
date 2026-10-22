"""
generate_sql.py - generate SQL for every test question with BOTH models and save the raw results.

Order of operations (this matters):
  1. Load the clean 4-bit base Mistral 7B and generate for every question.
  2. ONLY THEN attach the QLoRA adapter (PEFT modifies the base model in place)
     and generate again with the same prompts.

Both models get the exact training prompt (see prompts.py) and the same decoding
settings (greedy, same max_new_tokens). Nothing is hardcoded: every SQL string in the
output file was produced by a model during this run.

Usage (from the repo root):
    python src/generate_sql.py                       # 10 questions from data/evaluation_questions.py
    python src/generate_sql.py --limit 2             # quick smoke test
    python src/generate_sql.py --questions data/test_set.json --db data/company.db

Output: outputs/generated_sql.json
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                    # so `data.evaluation_questions` can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompts import build_prompt, schema_from_sqlite  # noqa: E402

DEFAULT_DB = ROOT / "data" / "company.db"
DEFAULT_OUT = ROOT / "outputs" / "generated_sql.json"
DEFAULT_MAX_NEW_TOKENS = 256  # the old notebook used 60, which cut some queries off mid-way


# --------------------------------------------------------------------------- helpers
def load_questions(path=None, limit=None):
    """Return a list of dicts with at least a 'question' key. Extra keys are passed through."""
    if path is None:
        from data.evaluation_questions import evaluation_questions
        items = [dict(q) for q in evaluation_questions]
    else:
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        if isinstance(items, dict) and "questions" in items:
            items = items["questions"]
    for i, item in enumerate(items, start=1):
        if "question" not in item:
            raise ValueError(f"Question #{i} has no 'question' field: {item}")
        item.setdefault("id", i)
    return items[:limit] if limit else items


def extract_sql(raw: str) -> str:
    """
    Turn raw model text into one SQL statement. The SAME rule is applied to both models:
      - cut at the first '###' (the base model tends to keep writing new prompts),
      - drop markdown code fences,
      - keep only the first statement (cut at the first ';').
    The untouched text is also saved as *_raw, so nothing is hidden.
    """
    text = raw.split("###")[0]
    text = re.sub(r"```(?:sql)?", "", text)
    text = text.strip()
    if ";" in text:
        text = text.split(";")[0]
    return text.strip()


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                       # greedy decoding -> reproducible
            pad_token_id=tokenizer.eos_token_id,
            stop_strings=["###"],                  # stop if the model starts a new prompt block
            tokenizer=tokenizer,                   # required by stop_strings
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_model(name, model, tokenizer, prompts, max_new_tokens):
    raws = []
    start = time.time()
    for i, prompt in enumerate(prompts, start=1):
        raws.append(generate_one(model, tokenizer, prompt, max_new_tokens))
        print(f"  [{name}] {i}/{len(prompts)} done ({time.time() - start:.0f}s elapsed)")
    return raws


def package_versions():
    versions = {}
    for pkg in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def save(out_path, metadata_dict, questions, base_raw, finetuned_raw):
    results = []
    for i, q in enumerate(questions):
        record = dict(q)
        if base_raw is not None:
            record["base_raw"] = base_raw[i]
            record["base_sql"] = extract_sql(base_raw[i])
        if finetuned_raw is not None:
            record["finetuned_raw"] = finetuned_raw[i]
            record["finetuned_sql"] = extract_sql(finetuned_raw[i])
        results.append(record)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata_dict, "results": results}, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default=None, help="JSON file with a list of {'question': ...}. Default: data/evaluation_questions.py")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite database used to build the schema in the prompt")
    parser.add_argument("--adapter", default=None, help="Adapter folder (default: models/adapter)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Where to write the JSON results")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions (smoke test)")
    args = parser.parse_args()

    # Heavy imports happen here so helpers above can be tested without a GPU.
    from model import (ADAPTER_PATH, BASE_MODEL, attach_adapter, count_lora_layers,
                       load_base_model, load_tokenizer)

    adapter_path = Path(args.adapter) if args.adapter else ADAPTER_PATH
    out_path = Path(args.out)

    questions = load_questions(args.questions, args.limit)
    schema = schema_from_sqlite(args.db)
    prompts = [build_prompt(schema, q["question"]) for q in questions]

    print(f"{len(questions)} questions | schema in prompt:\n  {schema}\n")
    print("Example prompt:\n" + "-" * 60 + f"\n{prompts[0]}" + "-" * 60 + "\n")

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_model": BASE_MODEL,
        "adapter_path": str(adapter_path),
        "quantization": "4-bit NF4, double quant, float16 compute",
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens, "stop_strings": ["###"]},
        "database": str(args.db),
        "schema_in_prompt": schema,
        "questions_source": args.questions or "data/evaluation_questions.py",
        "num_questions": len(questions),
        "package_versions": package_versions(),
    }

    tokenizer = load_tokenizer()

    # ---- Phase 1: clean base model (no adapter) ----
    print("PHASE 1: base Mistral 7B (no adapter)")
    base_model = load_base_model()
    assert count_lora_layers(base_model) == 0, "Base model unexpectedly contains LoRA layers!"
    base_raw = run_model("base", base_model, tokenizer, prompts, args.max_new_tokens)
    save(out_path, meta, questions, base_raw, None)  # checkpoint in case the runtime disconnects
    print(f"Base outputs saved to {out_path}\n")

    # ---- Phase 2: attach the adapter AFTER base generation is finished ----
    print("PHASE 2: base + QLoRA adapter")
    finetuned_model = attach_adapter(base_model)     # modifies base_model in place - fine, we are done with it
    n_lora = count_lora_layers(finetuned_model)
    assert n_lora > 0, "Adapter did not attach: no LoRA layers found!"
    print(f"Adapter attached ({n_lora} LoRA layers).")
    finetuned_raw = run_model("finetuned", finetuned_model, tokenizer, prompts, args.max_new_tokens)

    meta["lora_layers_attached"] = n_lora
    save(out_path, meta, questions, base_raw, finetuned_raw)

    differing = sum(1 for b, f in zip(base_raw, finetuned_raw) if extract_sql(b) != extract_sql(f))
    print(f"\nDone. Results written to {out_path}")
    print(f"Questions where base and fine-tuned SQL text differs: {differing}/{len(questions)}")
    print("Next: run evaluate.py (Step 3) to execute the SQL and compute accuracy.")


if __name__ == "__main__":
    main()
