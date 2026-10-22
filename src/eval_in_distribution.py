"""
eval_in_distribution.py - is the adapter healthy on the KIND of data it was trained on?

Why: on the 4-table company database the adapter scored far below the base model. Before
blaming (or retraining) the adapter we need to know whether it works on data like its
training data. This script runs BOTH models on b-mc2/sql-create-context rows that the adapter
never saw (index >= 5500; training used rows 0-5499), with the exact training prompt, split into:

    single-table rows   (1 CREATE TABLE in the schema)
    multi-table rows    (2+ CREATE TABLE statements)

What it measures (no data is needed, because these rows come without table contents):
  Valid SQL         the query compiles on EMPTY tables built from the row's own schema
                    (catches syntax errors and wrong table/column names).
  Exact match       normalised SQL text equals the gold SQL. WARNING: this favours the
                    fine-tuned model, because the gold SQL's style (T1/T2 aliases, quote style)
                    is what it was trained on. A zero-shot base model can be right and still
                    not match. Read it as "did the adapter learn the training style".
  Extra-JOIN rate   prediction has MORE joins than the gold query - the failure pattern seen
                    on the company database.

Rows whose question text also appears among the first 5,500 questions are removed
(use --allow-overlap to keep them), because identical questions would not be unseen.

Usage (GPU needed for generation, from the repo root):
    python src/eval_in_distribution.py --per-group 5          # smoke test
    python src/eval_in_distribution.py                        # 100 single + 100 multi-table rows
    python src/eval_in_distribution.py --score-only outputs/in_distribution/generated.json
                                                              # re-score a saved file (no GPU/dataset)
"""

import argparse
import csv
import json
import random
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import normalize_sql_text, wilson_interval  # noqa: E402
from generate_sql import extract_sql, generate_one, package_versions  # noqa: E402
from prompts import build_prompt  # noqa: E402

DATASET_NAME = "b-mc2/sql-create-context"
TRAIN_ROWS = 5500          # rows 0..5499 were used for training/validation in the notebook
SEED = 42
DEFAULT_OUT_DIR = ROOT / "outputs" / "in_distribution"
MODELS = ("base", "finetuned")
LABELS = {"base": "Base Mistral 7B", "finetuned": "Fine-tuned (QLoRA)"}


# --------------------------------------------------------------------------- helpers (CPU only)
def n_tables(context):
    return context.count("CREATE TABLE")


def count_joins(sql):
    return len(re.findall(r"\bjoin\b", sql or "", flags=re.IGNORECASE))


def norm_question(question):
    return re.sub(r"\s+", " ", question.lower()).strip()


def check_sql_valid(context, sql):
    """Compile `sql` on empty tables created from `context`. Returns (ok, error_message)."""
    if not sql or not sql.strip():
        return False, "empty SQL"
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(context.rstrip().rstrip(";") + ";")
        conn.execute("EXPLAIN " + sql)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()


def select_rows(questions, contexts, answers, per_group, seed=SEED,
                train_rows=TRAIN_ROWS, allow_overlap=False):
    """
    Pick `per_group` single-table and `per_group` multi-table rows from index >= train_rows.
    Returns (rows, info). Rows whose gold SQL does not compile on their own schema are dropped.
    """
    seen_questions = {norm_question(q) for q in questions[:train_rows]}
    single, multi = [], []
    removed_overlap = 0
    for i in range(train_rows, len(questions)):
        if not allow_overlap and norm_question(questions[i]) in seen_questions:
            removed_overlap += 1
            continue
        count = n_tables(contexts[i])
        if count == 0:
            continue
        row = {"index": i, "n_tables": count, "question": questions[i],
               "context": contexts[i], "gold_sql": answers[i]}
        (single if count == 1 else multi).append(row)

    rng = random.Random(seed)
    chosen, dropped_bad_gold = [], 0
    for group, pool in (("single_table", single), ("multi_table", multi)):
        sample = rng.sample(pool, min(per_group, len(pool)))
        for row in sample:
            ok, _ = check_sql_valid(row["context"], row["gold_sql"])
            if not ok:
                dropped_bad_gold += 1
                continue
            row["group"] = group
            chosen.append(row)
    info = {"candidates_single": len(single), "candidates_multi": len(multi),
            "removed_question_overlap_with_training_rows": removed_overlap,
            "dropped_gold_not_compiling": dropped_bad_gold,
            "seed": seed, "per_group_requested": per_group}
    return chosen, info


def score_row(row, model):
    sql = row.get(f"{model}_sql") or ""
    valid, error = check_sql_valid(row["context"], sql)
    exact = bool(sql) and normalize_sql_text(sql) == normalize_sql_text(row["gold_sql"])
    joins, gold_joins = count_joins(sql), count_joins(row["gold_sql"])
    return {"valid": valid, "error": error, "exact_match": exact,
            "joins": joins, "gold_joins": gold_joins, "more_joins_than_gold": joins > gold_joins}


def summarise(rows):
    """Per model and per group (plus 'all'): counts, rates and 95% Wilson intervals."""
    summary = {}
    for model in MODELS:
        summary[model] = {}
        scored = [(r, score_row(r, model)) for r in rows]
        for group in ("single_table", "multi_table", "all"):
            subset = [s for r, s in scored if group == "all" or r["group"] == group]
            n = len(subset)
            entry = {"n": n}
            for key, field in (("valid_sql", "valid"), ("exact_match", "exact_match"),
                               ("more_joins_than_gold", "more_joins_than_gold")):
                k = sum(1 for s in subset if s[field])
                entry[key] = {"k": k, "rate": k / n if n else None, "ci95": wilson_interval(k, n)}
            summary[model][group] = entry
    return summary


# --------------------------------------------------------------------------- printing / saving
def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def cell(entry):
    lo, hi = entry["ci95"]
    return f"{entry['k']}/{entry['n_rows']} ({pct(entry['rate'])}, CI {pct(lo)}-{pct(hi)})"


def print_summary(summary):
    print("\n" + "=" * 112)
    print("IN-DISTRIBUTION CHECK (rows the adapter never saw)")
    print("=" * 112)
    for group in ("single_table", "multi_table", "all"):
        print(f"\n[{group}]")
        print(f"  {'Model':<20} {'Valid SQL':<34} {'Exact match (favours fine-tuned)':<38} {'More JOINs than gold'}")
        for model in MODELS:
            e = summary[model][group]
            parts = []
            for key in ("valid_sql", "exact_match", "more_joins_than_gold"):
                item = dict(e[key], n_rows=e["n"])
                parts.append(cell(item))
            print(f"  {LABELS[model]:<20} {parts[0]:<34} {parts[1]:<38} {parts[2]}")
    print("\nHow to read this: if the adapter is clearly better than base on exact match AND has few")
    print("extra JOINs on multi-table rows, it learned its training data and the company-database")
    print("failure is a transfer problem. If it shows many extra JOINs even here, suspect the")
    print("training recipe. Exact match is style-dependent, so do not present it as 'accuracy'.")


def print_examples(rows, limit=5):
    bad = [r for r in rows if r["group"] == "multi_table" and score_row(r, "finetuned")["more_joins_than_gold"]]
    if not bad:
        return
    print("\n" + "=" * 112)
    print(f"EXAMPLES: fine-tuned multi-table outputs with MORE joins than gold (showing {min(limit, len(bad))} of {len(bad)})")
    print("=" * 112)
    for r in bad[:limit]:
        print(f"\nQ : {r['question']}\nSchema: {r['context']}\nGold  : {r['gold_sql']}\nBase  : {r.get('base_sql')}\nTuned : {r.get('finetuned_sql')}")


def save_scores(out_dir, meta, rows, summary):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "in_distribution_results.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": meta, "summary": summary}, f, indent=2, ensure_ascii=False)
    with open(out_dir / "in_distribution_per_row.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "group", "question", "gold_sql",
                    "base_sql", "base_valid", "base_exact", "base_joins",
                    "finetuned_sql", "finetuned_valid", "finetuned_exact", "finetuned_joins", "gold_joins"])
        for r in rows:
            b, t = score_row(r, "base"), score_row(r, "finetuned")
            w.writerow([r["index"], r["group"], r["question"], r["gold_sql"],
                        r.get("base_sql"), b["valid"], b["exact_match"], b["joins"],
                        r.get("finetuned_sql"), t["valid"], t["exact_match"], t["joins"], t["gold_joins"]])


# --------------------------------------------------------------------------- generation (GPU)
def generate_all(rows, meta, out_dir, max_new_tokens):
    from model import (ADAPTER_PATH, attach_adapter, count_lora_layers,
                       load_base_model, load_tokenizer)

    prompts = [build_prompt(r["context"], r["question"]) for r in rows]
    gen_path = out_dir / "generated.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint():
        with open(gen_path, "w", encoding="utf-8") as f:
            json.dump({"metadata": meta, "rows": rows}, f, indent=2, ensure_ascii=False)

    def run(name, model, tokenizer):
        start = time.time()
        for i, (row, prompt) in enumerate(zip(rows, prompts), start=1):
            raw = generate_one(model, tokenizer, prompt, max_new_tokens)
            row[f"{name}_raw"], row[f"{name}_sql"] = raw, extract_sql(raw)
            if i % 20 == 0 or i == len(rows):
                print(f"  [{name}] {i}/{len(rows)} ({time.time() - start:.0f}s)")

    tokenizer = load_tokenizer()
    print("PHASE 1: base model (no adapter)")
    base_model = load_base_model()
    assert count_lora_layers(base_model) == 0, "Base model unexpectedly contains LoRA layers!"
    run("base", base_model, tokenizer)
    checkpoint()

    print("PHASE 2: base + adapter (attached AFTER base generation is finished)")
    tuned = attach_adapter(base_model)
    n_lora = count_lora_layers(tuned)
    assert n_lora > 0, "Adapter did not attach!"
    meta["lora_layers_attached"] = n_lora
    meta["adapter_path"] = str(ADAPTER_PATH)
    run("finetuned", tuned, tokenizer)
    checkpoint()
    return gen_path


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-group", type=int, default=100, help="rows per group (single/multi table)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--allow-overlap", action="store_true",
                        help="keep rows whose question text also appears in the first 5,500 rows")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--score-only", default=None, help="score a saved generated.json (no GPU, no dataset)")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    if args.score_only:
        with open(args.score_only, "r", encoding="utf-8") as f:
            saved = json.load(f)
        rows, meta = saved["rows"], saved["metadata"]
        missing = [m for m in MODELS if not all(f"{m}_sql" in r for r in rows)]
        if missing:
            raise SystemExit(f"{args.score_only} has no outputs for: {missing}. Generation did not finish.")
    else:
        from datasets import load_dataset

        ds = load_dataset(DATASET_NAME)["train"]
        rows, info = select_rows(ds["question"], ds["context"], ds["answer"],
                                 args.per_group, args.seed, TRAIN_ROWS, args.allow_overlap)
        n_single = sum(1 for r in rows if r["group"] == "single_table")
        print(f"Selected {len(rows)} unseen rows ({n_single} single-table, {len(rows) - n_single} multi-table). {info}")
        meta = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "dataset": DATASET_NAME, "train_rows_excluded": TRAIN_ROWS, "selection": info,
                "max_new_tokens": args.max_new_tokens, "package_versions": package_versions()}
        generate_all(rows, meta, out_dir, args.max_new_tokens)

    summary = summarise(rows)
    print_summary(summary)
    print_examples(rows)
    save_scores(out_dir, meta, rows, summary)
    saved_files = "in_distribution_results.json, in_distribution_per_row.csv"
    if not args.score_only:
        saved_files = "generated.json, " + saved_files
    print(f"\nSaved to {out_dir}: {saved_files}")


if __name__ == "__main__":
    main()
