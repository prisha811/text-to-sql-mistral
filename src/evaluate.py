"""
evaluate.py - honest evaluation of base vs fine-tuned Text-to-SQL outputs.

Reads outputs/generated_sql.json (written by generate_sql.py) and, for every question and
for BOTH models:

  1. runs the generated SQL on the SQLite database (read-only, 5 s time limit),
  2. runs the gold query (or uses the stored expected_result) and compares result sets.

Metrics
  Execution accuracy  the generated query's result set equals the gold result set.
                      Row ORDER is ignored (rows are compared as multisets). Column order and
                      the number of columns DO matter. Floats are compared to 6 decimals.
  Valid SQL           the query executes without any error (syntax errors, unknown tables or
                      columns, several statements, timeouts all count as invalid).
                      Invalid SQL is always a failure.
  Exact match         the generated SQL text equals the gold SQL text after light
                      normalisation (case outside quotes, whitespace, quote style, trailing ';').
                      Table aliases are NOT normalised, so exact match is a strict, low number:
                      two different queries can both be correct. It needs gold_sql, so it is
                      reported as n/a when a question only has an expected_result.

Also printed: 95% Wilson confidence intervals, a paired comparison (both / only one / neither)
with a rough sign test, a breakdown by "category" if the questions have one, and a failure
analysis. With small test sets differences are often noise, and this script says so.

Usage (from the repo root, no GPU needed):
    python src/evaluate.py
    python src/evaluate.py --input outputs/generated_sql.json --db data/company.db
"""

import argparse
import csv
import json
import math
import re
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "outputs" / "generated_sql.json"
DEFAULT_DB = ROOT / "data" / "company.db"
DEFAULT_OUT_DIR = ROOT / "outputs"
QUERY_TIMEOUT_SECONDS = 5
MODELS = ("base", "finetuned")
MODEL_LABELS = {"base": "Base Mistral 7B", "finetuned": "Fine-tuned (QLoRA)"}


# --------------------------------------------------------------------------- SQL execution
def connect_readonly(db_path):
    """Open the database read-only so generated SQL can never modify it."""
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def run_sql(conn, sql, timeout=QUERY_TIMEOUT_SECONDS):
    """Execute one query. Never raises: returns {'ok', 'rows', 'n_cols', 'error'}."""
    if sql is None or not str(sql).strip():
        return {"ok": False, "rows": [], "n_cols": 0, "error": "empty SQL"}
    deadline = time.monotonic() + timeout
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        n_cols = len(cursor.description) if cursor.description else 0
        return {"ok": True, "rows": rows, "n_cols": n_cols, "error": None}
    except Exception as exc:  # any failure = invalid SQL
        return {"ok": False, "rows": [], "n_cols": 0, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        conn.set_progress_handler(None, 0)


# --------------------------------------------------------------------------- comparison
def normalize_value(value):
    return round(value, 6) if isinstance(value, float) else value


def result_multiset(rows):
    """Order-insensitive representation of a result set."""
    return Counter(tuple(normalize_value(v) for v in row) for row in rows)


def normalize_sql_text(sql):
    """Light normalisation for exact match. String literals keep their case."""
    text = sql.strip().rstrip(";").strip().replace('"', "'")
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"('[^']*')", text)
    parts = [p if p.startswith("'") else re.sub(r"\s*([(),=<>])\s*", r"\1", p.lower()) for p in parts]
    return "".join(parts)


def classify(pred, gold_rows, gold_cols):
    """Return (correct, failure_type)."""
    if not pred["ok"]:
        return False, "invalid_sql"
    if result_multiset(pred["rows"]) == result_multiset(gold_rows):
        return True, "ok"
    if gold_cols is not None and pred["n_cols"] != gold_cols:
        return False, "wrong_columns"
    return False, "wrong_rows"


def preview(rows, limit=5):
    shown = sorted((list(r) for r in rows), key=str)[:limit]
    more = f" ... (+{len(rows) - limit} more)" if len(rows) > limit else ""
    return f"{shown}{more}"


# --------------------------------------------------------------------------- statistics
def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval for a proportion. Returns (low, high) as fractions."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def sign_test_p(only_a, only_b):
    """Exact two-sided sign test on the questions where exactly one model was correct."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


# --------------------------------------------------------------------------- evaluation
def evaluate_record(conn, record):
    """Evaluate one question for both models. Returns a dict (JSON-friendly)."""
    out = {
        "id": record.get("id"),
        "question": record["question"],
        "category": record.get("category"),
        "gold_sql": record.get("gold_sql"),
        "gold_ok": True,
        "gold_error": None,
        "gold_empty": False,
    }

    # Gold result: execute gold_sql if we have it, else use the stored expected_result.
    if record.get("gold_sql"):
        gold = run_sql(conn, record["gold_sql"])
        if not gold["ok"]:
            out.update(gold_ok=False, gold_error=gold["error"])
            return out
        gold_rows, gold_cols = gold["rows"], gold["n_cols"]
    elif "expected_result" in record:
        gold_rows = [tuple(r) for r in record["expected_result"]]
        gold_cols = len(gold_rows[0]) if gold_rows else None
    else:
        out.update(gold_ok=False, gold_error="record has neither gold_sql nor expected_result")
        return out

    out["gold_empty"] = len(gold_rows) == 0
    out["gold_rows_preview"] = preview(gold_rows)

    for model in MODELS:
        sql = record.get(f"{model}_sql")
        pred = run_sql(conn, sql)
        correct, failure = classify(pred, gold_rows, gold_cols)
        exact = None
        if record.get("gold_sql"):
            exact = bool(sql) and normalize_sql_text(sql) == normalize_sql_text(record["gold_sql"])
        out[model] = {
            "sql": sql,
            "valid": pred["ok"],
            "correct": correct,
            "exact_match": exact,
            "failure_type": failure,
            "error": pred["error"],
            "rows_preview": preview(pred["rows"]) if pred["ok"] else None,
        }
    return out


def summarise(evaluated):
    """Aggregate metrics over the questions whose gold result is usable."""
    usable = [e for e in evaluated if e["gold_ok"]]
    n = len(usable)
    summary = {"num_questions": len(evaluated), "num_scored": n,
               "num_bad_gold": len(evaluated) - n,
               "num_empty_gold": sum(1 for e in usable if e["gold_empty"]), "models": {}}
    for model in MODELS:
        correct = sum(e[model]["correct"] for e in usable)
        valid = sum(e[model]["valid"] for e in usable)
        with_gold_sql = [e for e in usable if e[model]["exact_match"] is not None]
        exact = sum(e[model]["exact_match"] for e in with_gold_sql)
        failures = Counter(e[model]["failure_type"] for e in usable if not e[model]["correct"])
        summary["models"][model] = {
            "execution_correct": correct,
            "execution_accuracy": correct / n if n else None,
            "execution_ci95": wilson_interval(correct, n),
            "valid_sql": valid,
            "valid_sql_rate": valid / n if n else None,
            "exact_match": exact if with_gold_sql else None,
            "exact_match_n": len(with_gold_sql),
            "exact_match_rate": exact / len(with_gold_sql) if with_gold_sql else None,
            "failure_types": dict(failures),
        }
    both = sum(1 for e in usable if e["base"]["correct"] and e["finetuned"]["correct"])
    only_base = sum(1 for e in usable if e["base"]["correct"] and not e["finetuned"]["correct"])
    only_ft = sum(1 for e in usable if e["finetuned"]["correct"] and not e["base"]["correct"])
    neither = n - both - only_base - only_ft
    summary["paired"] = {"both_correct": both, "only_base_correct": only_base,
                         "only_finetuned_correct": only_ft, "neither_correct": neither,
                         "sign_test_p": sign_test_p(only_base, only_ft)}

    categories = sorted({e["category"] for e in usable if e["category"]})
    summary["by_category"] = {}
    for cat in categories:
        subset = [e for e in usable if e["category"] == cat]
        summary["by_category"][cat] = {
            "n": len(subset),
            **{m: sum(e[m]["correct"] for e in subset) for m in MODELS},
        }
    return summary


# --------------------------------------------------------------------------- printing
def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def mark(ok):
    return "PASS" if ok else "FAIL"


def format_p(p):
    """Show tiny p-values in scientific notation instead of rounding them to 0.000."""
    return f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def print_per_question_table(evaluated):
    print("\n" + "=" * 100)
    print("PER-QUESTION RESULTS (execution correctness)")
    print("=" * 100)
    header = f"{'ID':>4}  {'Question':<52} {'Base':<6} {'Fine-tuned':<10} {'Failure type (base / fine-tuned)'}"
    print(header)
    print("-" * len(header))
    for e in evaluated:
        question = e["question"] if len(e["question"]) <= 50 else e["question"][:47] + "..."
        if not e["gold_ok"]:
            print(f"{str(e['id']):>4}  {question:<52} SKIPPED (bad gold: {e['gold_error']})")
            continue
        b, f = e["base"], e["finetuned"]
        failure = f"{b['failure_type']} / {f['failure_type']}"
        print(f"{str(e['id']):>4}  {question:<52} {mark(b['correct']):<6} {mark(f['correct']):<10} {failure}")


def print_summary(summary):
    n = summary["num_scored"]
    print("\n" + "=" * 100)
    print(f"SUMMARY  ({n} scored questions)")
    print("=" * 100)
    if summary["num_bad_gold"]:
        print(f"WARNING: {summary['num_bad_gold']} question(s) skipped because the gold query failed or is missing.")
    if summary["num_empty_gold"]:
        print(f"WARNING: {summary['num_empty_gold']} question(s) have an EMPTY gold result; "
              "an empty answer would count as correct there. Fix those test questions.")
    header = f"{'Model':<22} {'Execution accuracy':<26} {'95% CI':<18} {'Valid SQL':<14} {'Exact match'}"
    print(header)
    print("-" * len(header))
    for model in MODELS:
        m = summary["models"][model]
        lo, hi = m["execution_ci95"]
        acc = f"{m['execution_correct']}/{n} = {pct(m['execution_accuracy'])}"
        valid = f"{m['valid_sql']}/{n} ({pct(m['valid_sql_rate'])})"
        exact = ("n/a (no gold SQL)" if m["exact_match"] is None
                 else f"{m['exact_match']}/{m['exact_match_n']} ({pct(m['exact_match_rate'])})")
        ci = f"{pct(lo)}-{pct(hi)}"
        print(f"{MODEL_LABELS[model]:<22} {acc:<26} {ci:<18} {valid:<14} {exact}")

    p = summary["paired"]
    print("\nPaired comparison (same questions, both models):")
    print(f"  both correct: {p['both_correct']} | only base correct: {p['only_base_correct']} | "
          f"only fine-tuned correct: {p['only_finetuned_correct']} | neither: {p['neither_correct']}")
    print(f"  rough sign test on the 'only one correct' questions: p = {format_p(p['sign_test_p'])} "
          "(small p = difference unlikely to be chance; with few questions it is rarely small)")

    print("\nFailure types among wrong answers:")
    for model in MODELS:
        print(f"  {MODEL_LABELS[model]:<22} {summary['models'][model]['failure_types'] or 'none'}")

    if summary["by_category"]:
        print("\nBy category (correct / total):")
        print(f"  {'Category':<24} {'n':>3}  {'Base':>6}  {'Fine-tuned':>10}")
        for cat, c in summary["by_category"].items():
            print(f"  {cat:<24} {c['n']:>3}  {c['base']:>6}  {c['finetuned']:>10}")


def print_failure_details(evaluated):
    failed = [e for e in evaluated if e["gold_ok"] and not (e["base"]["correct"] and e["finetuned"]["correct"])]
    if not failed:
        return
    print("\n" + "=" * 100)
    print("FAILURE DETAILS (use these to write your error analysis)")
    print("=" * 100)
    for e in failed:
        print(f"\n[{e['id']}] {e['question']}")
        print(f"  gold result : {e['gold_rows_preview']}")
        for model in MODELS:
            r = e[model]
            if r["correct"]:
                continue
            print(f"  {MODEL_LABELS[model]} -> {r['failure_type']}")
            print(f"    SQL    : {r['sql']}")
            print(f"    {'error ' if r['error'] else 'result'} : {r['error'] or r['rows_preview']}")


def markdown_table(summary):
    n = summary["num_scored"]
    lines = ["| Model | Execution accuracy | Valid SQL | Exact match |", "|---|---|---|---|"]
    for model in MODELS:
        m = summary["models"][model]
        lo, hi = m["execution_ci95"]
        exact = "n/a" if m["exact_match"] is None else f"{m['exact_match']}/{m['exact_match_n']} ({pct(m['exact_match_rate'])})"
        lines.append(
            f"| {MODEL_LABELS[model]} | {m['execution_correct']}/{n} ({pct(m['execution_accuracy'])}, "
            f"95% CI {pct(lo)}-{pct(hi)}) | {m['valid_sql']}/{n} ({pct(m['valid_sql_rate'])}) | {exact} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- saving
def save_outputs(out_dir, source, db_path, generation_meta, evaluated, summary):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "input_file": str(source),
            "database": str(db_path),
            "query_timeout_seconds": QUERY_TIMEOUT_SECONDS,
            "generation_metadata": generation_meta,
        },
        "summary": summary,
        "per_question": evaluated,
    }
    with open(out_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    with open(out_dir / "evaluation_per_question.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "category", "question", "base_correct", "base_failure", "base_sql",
                         "finetuned_correct", "finetuned_failure", "finetuned_sql"])
        for e in evaluated:
            if not e["gold_ok"]:
                writer.writerow([e["id"], e["category"], e["question"], "SKIPPED"] + [""] * 5)
                continue
            writer.writerow([e["id"], e["category"], e["question"],
                             e["base"]["correct"], e["base"]["failure_type"], e["base"]["sql"],
                             e["finetuned"]["correct"], e["finetuned"]["failure_type"], e["finetuned"]["sql"]])


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="JSON written by generate_sql.py")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite database to execute queries on")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data["results"]
    for model in MODELS:
        if not all(f"{model}_sql" in r for r in records):
            raise SystemExit(f"Input file is missing '{model}_sql' for some questions. "
                             "Run generate_sql.py to completion first.")

    conn = connect_readonly(args.db)
    try:
        evaluated = [evaluate_record(conn, r) for r in records]
    finally:
        conn.close()

    summary = summarise(evaluated)
    print_per_question_table(evaluated)
    print_summary(summary)
    print_failure_details(evaluated)
    print("\nMarkdown table (review before copying into the README):\n")
    print(markdown_table(summary))

    save_outputs(Path(args.out_dir), args.input, args.db, data.get("metadata"), evaluated, summary)
    print(f"\nSaved: {Path(args.out_dir) / 'evaluation_results.json'} and evaluation_per_question.csv")


if __name__ == "__main__":
    main()
