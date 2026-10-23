"""
eval_in_distribution.py - is the adapter healthy on the KIND of data it was trained on?

Why: on the 4-table company database the adapter scored far below the base model. Before
blaming (or retraining) the adapter we need to know whether it works on data like its
training data, with the exact training prompt, separately for single-table and multi-table schemas.

Two sources of rows the adapter never trained on (choose with --source):

  validation (default)  The 500 validation rows from the training notebook. The split is rebuilt with
                        the notebook's own code (first 5,500 rows, test_size=500, seed=42) and checked
                        against the notebook's first training example. These rows never produced a
                        gradient update. They were only used to log validation loss, and the final
                        adapter was saved without picking a "best" checkpoint. They contain BOTH
                        single-table and multi-table rows. Caveat: Spider-style datasets contain
                        similar questions, so this measures same-distribution performance and can be
                        optimistic.
  tail                  Rows with index >= 5,500. The first run showed these are ALL single-table:
                        every multi-table row of the dataset is inside the first 5,500 rows. Useful as
                        an extra single-table check only.

In both cases rows whose question text also appears among the TRAINING questions are removed
(use --allow-overlap to keep them).

What it measures (no table contents exist for these rows, so no result comparison is possible):
  Valid SQL         the query compiles on EMPTY tables built from the row's own schema
                    (catches syntax errors and wrong table/column names).
  Exact match       normalised SQL text equals the gold SQL. WARNING: this favours the
                    fine-tuned model, because the gold SQL's style (T1/T2 aliases, quote style)
                    is what it was trained on. A zero-shot base model can be right and still
                    not match. Read it as "did the adapter learn the training style".
  Extra-JOIN rate   prediction has MORE joins than the gold query - the failure pattern seen on the
                    company database. On single-table rows there is almost nothing to join, so
                    this number only means something on multi-table rows.

Distractor test (--distractor-tables N --distractor-cols M): before the models see a row, N unneeded
tables and M unneeded columns (borrowed from OTHER rows of the same pool) are added to its schema. The
gold SQL still holds, because it never uses them. The script then reports how often each model's SQL
mentions an added table or column. This reproduces the company-database failure ("uses tables it
does not need") on validation rows instead of on the test set. Use --adapter to test another adapter.

Usage (GPU needed for generation, from the repo root):
    python src/eval_in_distribution.py --per-group 5          # smoke test
    python src/eval_in_distribution.py                        # up to 100 single + 100 multi-table rows
    python src/eval_in_distribution.py --distractor-tables 2 --distractor-cols 2 --per-group 50
    python src/eval_in_distribution.py --score-only outputs/in_distribution/validation/generated.json
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

from augment import Augmenter  # noqa: E402
from evaluate import normalize_sql_text, wilson_interval  # noqa: E402
from generate_sql import extract_sql, generate_one, package_versions  # noqa: E402
from prompts import build_prompt  # noqa: E402

DATASET_NAME = "b-mc2/sql-create-context"
TRAIN_ROWS = 5500          # rows 0..5499 were used for training/validation in the notebook
SEED = 42                  # seed for sampling rows
SPLIT_SEED = 42            # seed the notebook used for train_test_split
VALIDATION_SIZE = 500
# First example of the notebook's training split (LLM.ipynb, cell 32/35). If the rebuilt split does
# not start with this question, it is not the split the adapter was trained on.
KNOWN_FIRST_TRAIN_QUESTION = ("How many million of windows phones were shipped during the quarter "
                              "that shipped 216.2 million total?")
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


def check_split_matches_training(first_train_question):
    """The rebuilt split must start with the notebook's first training example."""
    if first_train_question != KNOWN_FIRST_TRAIN_QUESTION:
        raise RuntimeError(
            "The rebuilt train/validation split does NOT match the notebook's split "
            f"(first training question is {first_train_question!r}). The 'validation' rows may "
            "include rows the adapter was trained on, so they cannot be called held-out. "
            "Stop and check the datasets version, or use --source tail for single-table rows.")


def select_from_pool(questions, contexts, answers, seen_questions, per_group, seed=SEED,
                     allow_overlap=False, index_offset=0, source="tail"):
    """
    Pick up to `per_group` single-table and `per_group` multi-table rows from a pool.
    Rows whose question also appears in `seen_questions` (the training questions) are removed.
    Rows whose gold SQL does not compile on their own schema are dropped.
    Returns (rows, info).
    """
    seen = {norm_question(q) for q in seen_questions}
    single, multi = [], []
    removed_overlap = 0
    for pos, (question, context, answer) in enumerate(zip(questions, contexts, answers)):
        if not allow_overlap and norm_question(question) in seen:
            removed_overlap += 1
            continue
        count = n_tables(context)
        if count == 0:
            continue
        row = {"index": index_offset + pos, "source": source, "n_tables": count,
               "question": question, "context": context, "gold_sql": answer}
        (single if count == 1 else multi).append(row)

    rng = random.Random(seed)
    chosen, dropped_bad_gold = [], 0
    for group, pool in (("single_table", single), ("multi_table", multi)):
        for row in rng.sample(pool, min(per_group, len(pool))):
            ok, _ = check_sql_valid(row["context"], row["gold_sql"])
            if not ok:
                dropped_bad_gold += 1
                continue
            row["group"] = group
            chosen.append(row)
    info = {"source": source, "candidates_single": len(single), "candidates_multi": len(multi),
            "removed_question_overlap_with_training_rows": removed_overlap,
            "dropped_gold_not_compiling": dropped_bad_gold,
            "seed": seed, "per_group_requested": per_group}
    return chosen, info


def select_rows(questions, contexts, answers, per_group, seed=SEED,
                train_rows=TRAIN_ROWS, allow_overlap=False):
    """'tail' source: rows with index >= train_rows, never used for training or validation."""
    return select_from_pool(questions[train_rows:], contexts[train_rows:], answers[train_rows:],
                            questions[:train_rows], per_group, seed, allow_overlap,
                            index_offset=train_rows, source="tail")


def score_row(row, model):
    sql = row.get(f"{model}_sql") or ""
    valid, error = check_sql_valid(row["context"], sql)
    exact = bool(sql) and normalize_sql_text(sql) == normalize_sql_text(row["gold_sql"])
    joins, gold_joins = count_joins(sql), count_joins(row["gold_sql"])
    uses_table = any(re.search(rf"\b{re.escape(t)}\b", sql, re.IGNORECASE) for t in row.get("added_tables") or [])
    uses_column = any(re.search(rf"\b{re.escape(c)}\b", sql, re.IGNORECASE) for c in row.get("added_columns") or [])
    return {"valid": valid, "error": error, "exact_match": exact,
            "joins": joins, "gold_joins": gold_joins, "more_joins_than_gold": joins > gold_joins,
            "uses_distractor_table": uses_table, "uses_distractor_column": uses_column}


def has_distractors(rows):
    return any(r.get("added_tables") or r.get("added_columns") for r in rows)


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
                               ("more_joins_than_gold", "more_joins_than_gold"),
                               ("uses_distractor_table", "uses_distractor_table"),
                               ("uses_distractor_column", "uses_distractor_column")):
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


def print_summary(summary, with_distractors=False):
    print("\n" + "=" * 112)
    print("IN-DISTRIBUTION CHECK (rows the adapter never saw)")
    print("=" * 112)
    for group in ("single_table", "multi_table", "all"):
        print(f"\n[{group}]")
        if summary[MODELS[0]][group]["n"] == 0:
            print("  (no rows in this group)")
            continue
        keys = ["valid_sql", "exact_match", "more_joins_than_gold"]
        titles = ["Valid SQL", "Exact match (favours fine-tuned)", "More JOINs than gold"]
        widths = [34, 38, 34]
        if with_distractors:
            keys += ["uses_distractor_table", "uses_distractor_column"]
            titles += ["Uses an added (unneeded) TABLE", "Uses an added (unneeded) COLUMN"]
            widths += [34, 34]
        print("  " + f"{'Model':<20} " + " ".join(f"{t:<{w}}" for t, w in zip(titles, widths)))
        for model in MODELS:
            e = summary[model][group]
            parts = [cell(dict(e[k], n_rows=e["n"])) for k in keys]
            print("  " + f"{LABELS[model]:<20} " + " ".join(f"{p:<{w}}" for p, w in zip(parts, widths)))
    print("\nHow to read this: if the adapter is clearly better than base on exact match AND has few")
    print("extra JOINs on multi-table rows, it learned its training data and the company-database")
    print("failure is a transfer problem. If it shows many extra JOINs even here, suspect the")
    print("training recipe. Exact match is style-dependent, so do not present it as 'accuracy'.")
    if with_distractors:
        print("\nDistractor test: unneeded tables/columns were ADDED to every schema. A model that ignores")
        print("them scores near 0% on the two 'Uses an added ...' columns. The base model is the reference.")


def print_examples(rows, limit=5, with_distractors=False):
    if with_distractors:
        used = [r for r in rows if score_row(r, "finetuned")["uses_distractor_table"]]
        print("\n" + "=" * 112)
        print(f"EXAMPLES: fine-tuned outputs that use an ADDED (unneeded) table (showing {min(3, len(used))} of {len(used)})")
        print("=" * 112)
        for r in used[:3]:
            print(f"\nQ : {r['question']}\nSchema: {r['context']}\nAdded tables: {r.get('added_tables')}\n"
                  f"Gold  : {r['gold_sql']}\nBase  : {r.get('base_sql')}\nTuned : {r.get('finetuned_sql')}")
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
                    "finetuned_sql", "finetuned_valid", "finetuned_exact", "finetuned_joins", "gold_joins",
                    "added_tables", "added_columns", "base_uses_added_table", "finetuned_uses_added_table"])
        for r in rows:
            b, t = score_row(r, "base"), score_row(r, "finetuned")
            w.writerow([r["index"], r["group"], r["question"], r["gold_sql"],
                        r.get("base_sql"), b["valid"], b["exact_match"], b["joins"],
                        r.get("finetuned_sql"), t["valid"], t["exact_match"], t["joins"], t["gold_joins"],
                        " ".join(r.get("added_tables") or []), " ".join(r.get("added_columns") or []),
                        b["uses_distractor_table"], t["uses_distractor_table"]])


# --------------------------------------------------------------------------- generation (GPU)
def generate_all(rows, meta, out_dir, max_new_tokens, adapter_path=None):
    from model import (ADAPTER_PATH, attach_adapter, count_lora_layers,
                       load_base_model, load_tokenizer)

    adapter_path = Path(adapter_path) if adapter_path else ADAPTER_PATH

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
    tuned = attach_adapter(base_model, adapter_path)
    n_lora = count_lora_layers(tuned)
    assert n_lora > 0, "Adapter did not attach!"
    meta["lora_layers_attached"] = n_lora
    meta["adapter_path"] = str(adapter_path)
    run("finetuned", tuned, tokenizer)
    checkpoint()
    return gen_path


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["validation", "tail"], default="validation",
                        help="where the unseen rows come from (default: the notebook's 500 validation rows)")
    parser.add_argument("--per-group", type=int, default=100, help="max rows per group (single/multi table)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--adapter", default=None, help="adapter folder to test (default: models/adapter)")
    parser.add_argument("--distractor-tables", type=int, default=0,
                        help="add this many UNNEEDED tables to every schema (distractor test)")
    parser.add_argument("--distractor-cols", type=int, default=0,
                        help="add this many UNNEEDED columns to every schema (distractor test)")
    parser.add_argument("--allow-overlap", action="store_true",
                        help="keep rows whose question text also appears among the training questions")
    parser.add_argument("--out-dir", default=None,
                        help="default: outputs/in_distribution/<source> (so runs never overwrite each other)")
    parser.add_argument("--score-only", default=None, help="score a saved generated.json (no GPU, no dataset)")
    args = parser.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR / args.source

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
        if args.source == "validation":
            split = ds.select(range(TRAIN_ROWS)).train_test_split(test_size=VALIDATION_SIZE, seed=SPLIT_SEED)
            train_part, val_part = split["train"], split["test"]
            check_split_matches_training(train_part[0]["question"])
            print("Split check passed: the rebuilt split starts with the notebook's first training example.")
            pool_contexts = list(val_part["context"])
            rows, info = select_from_pool(val_part["question"], val_part["context"], val_part["answer"],
                                          train_part["question"], args.per_group, args.seed,
                                          args.allow_overlap, index_offset=0, source="validation")
        else:
            pool_contexts = list(ds["context"][TRAIN_ROWS:])
            rows, info = select_rows(ds["question"], ds["context"], ds["answer"],
                                     args.per_group, args.seed, TRAIN_ROWS, args.allow_overlap)
        for group in ("single_table", "multi_table"):
            if not any(r["group"] == group for r in rows):
                message = f"No {group} rows available from source '{args.source}'. Selection info: {info}"
                if args.source == "validation":
                    raise SystemExit(message)
                print("WARNING:", message)
        if args.distractor_tables or args.distractor_cols:
            augmenter = Augmenter(pool_contexts, seed=args.seed)
            statuses = []
            for row in rows:
                result = augmenter.augment_fixed(row["context"], row["gold_sql"], row["index"],
                                                 args.distractor_tables, args.distractor_cols)
                statuses.append(result.status)
                if result.status == "augmented":
                    row["original_context"] = row["context"]
                    row["context"] = result.context
                    row["added_tables"], row["added_columns"] = result.added_tables, result.added_columns
            info["distractors"] = {"tables": args.distractor_tables, "columns": args.distractor_cols,
                                   "status_counts": {s: statuses.count(s) for s in sorted(set(statuses))}}
            print(f"Distractors added: {info['distractors']}")
        n_single = sum(1 for r in rows if r["group"] == "single_table")
        print(f"Selected {len(rows)} unseen rows ({n_single} single-table, {len(rows) - n_single} multi-table). {info}")
        meta = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "dataset": DATASET_NAME, "source": args.source, "train_rows_excluded": TRAIN_ROWS, "selection": info,
                "max_new_tokens": args.max_new_tokens, "package_versions": package_versions()}
        generate_all(rows, meta, out_dir, args.max_new_tokens, args.adapter)

    summary = summarise(rows)
    with_distractors = has_distractors(rows)
    print_summary(summary, with_distractors)
    print_examples(rows, with_distractors=with_distractors)
    save_scores(out_dir, meta, rows, summary)
    saved_files = "in_distribution_results.json, in_distribution_per_row.csv"
    if not args.score_only:
        saved_files = "generated.json, " + saved_files
    print(f"\nSaved to {out_dir}: {saved_files}")


if __name__ == "__main__":
    main()
