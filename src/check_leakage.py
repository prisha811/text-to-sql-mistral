"""
check_leakage.py - checks that the test questions do not appear in the QLoRA training data.

Training data = the first 5,500 rows of b-mc2/sql-create-context (5,000 train + 500 validation),
exactly what the notebook used. All 5,500 are checked, because the validation rows were also
seen by the training script.

Three checks per test question:
  1. exact match   - same question text after lower-casing and removing punctuation
  2. near match    - word-overlap (Jaccard) with the closest training question; >= 0.70 is flagged
  3. SQL match     - the gold SQL text equals a training answer after normalisation

It also reports how many training rows mention tables called "employees" / "departments" etc.,
which tells you how close the training distribution is to your test schema (informational).

Usage (Colab/Kaggle needs internet for the dataset):
    python src/check_leakage.py
    python src/check_leakage.py --local some_rows.jsonl     # offline: lines with question/answer/context
Writes outputs/leakage_report.json. Exit code 1 if anything is flagged.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NUM_ROWS = 5500
NEAR_THRESHOLD = 0.70


def norm_q(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def norm_sql(text):
    return re.sub(r"\s+", " ", text.lower().replace('"', "'")).strip().rstrip(";").strip()


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def load_rows(local):
    if local:
        return [json.loads(line) for line in Path(local).read_text(encoding="utf-8").splitlines() if line.strip()]
    from datasets import load_dataset
    ds = load_dataset("b-mc2/sql-create-context")["train"].select(range(NUM_ROWS))
    return [dict(r) for r in ds]


def check(test_items, rows):
    tq = [(norm_q(r["question"]), set(norm_q(r["question"]).split())) for r in rows]
    ts = {norm_sql(r["answer"]) for r in rows}
    flagged = []
    for t in test_items:
        n = norm_q(t["question"]); words = set(n.split())
        best, best_row = 0.0, None
        for (rn, rw), row in zip(tq, rows):
            score = 1.0 if rn == n else jaccard(words, rw)
            if score > best:
                best, best_row = score, row
        sql_hit = norm_sql(t["gold_sql"]) in ts
        if best >= NEAR_THRESHOLD or sql_hit:
            flagged.append({"id": t["id"], "question": t["question"], "similarity": round(best, 3),
                            "closest_training_question": best_row["question"] if best_row else None,
                            "gold_sql_in_training": sql_hit})
    return flagged


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-set", default=str(ROOT / "data" / "test_set.json"))
    p.add_argument("--local", default=None)
    p.add_argument("--out", default=str(ROOT / "outputs" / "leakage_report.json"))
    args = p.parse_args()

    items = json.loads(Path(args.test_set).read_text(encoding="utf-8"))
    rows = load_rows(args.local)
    flagged = check(items, rows)

    schema_hits = {name: sum(1 for r in rows if re.search(rf"CREATE TABLE {name}\b", r["context"], re.I))
                   for name in ("employees", "employee", "departments", "department", "projects")}
    report = {"training_rows_checked": len(rows), "test_questions": len(items),
              "near_match_threshold": NEAR_THRESHOLD, "flagged": flagged,
              "training_rows_with_table_name": schema_hits}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Checked {len(items)} test questions against {len(rows)} training rows.")
    print(f"Training rows whose schema has a table named ...: {schema_hits}")
    for f in flagged:
        print(f"[FLAGGED] #{f['id']} sim={f['similarity']} sql_in_training={f['gold_sql_in_training']}\n"
              f"   test : {f['question']}\n   train: {f['closest_training_question']}")
    print(f"\n{len(flagged)}/{len(items)} flagged. "
          + ("Rewrite or remove flagged questions, rebuild test_set.json and re-run." if flagged
             else "No overlap found by these checks (this does not prove the base model never saw similar data)."))
    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
