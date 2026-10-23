"""
CPU-only checks for the training-data side of train_qlora.py (no GPU, no download). From the repo root:

    python tests/test_train_data.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from augment import sql_compiles  # noqa: E402
from prompts import build_prompt, build_training_text  # noqa: E402
from train_qlora import build_training_rows  # noqa: E402

CONTEXTS = [
    "CREATE TABLE department (creation VARCHAR, department_id VARCHAR); CREATE TABLE management (department_id VARCHAR, head_id VARCHAR); CREATE TABLE head (head_id VARCHAR, born_state VARCHAR)",
    "CREATE TABLE singer (name VARCHAR, age VARCHAR, country VARCHAR)",
    "CREATE TABLE table_name_77 (player VARCHAR, score VARCHAR, year INTEGER)",
    "CREATE TABLE table_name_12 (team VARCHAR, wins INTEGER, losses INTEGER)",
    "CREATE TABLE concert (concert_id VARCHAR, year VARCHAR, stadium_id VARCHAR); CREATE TABLE stadium (stadium_id VARCHAR, capacity INTEGER)",
]
ANSWERS = [
    "SELECT T1.creation FROM department AS T1 JOIN management AS T2 ON T1.department_id = T2.department_id",
    "SELECT name FROM singer WHERE age > 30",
    "SELECT score FROM table_name_77 WHERE player = 'x'",
    "SELECT team FROM table_name_12 WHERE wins > 3",
    "SELECT T1.year FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id = T2.stadium_id",
]


class FakePart:
    """Mimics the bits of a Hugging Face Dataset that build_training_rows uses."""

    def __init__(self, n):
        self.rows = [{"question": f"question {i}", "context": CONTEXTS[i % 5], "answer": ANSWERS[i % 5]}
                     for i in range(n)]

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, key):
        return [r[key] for r in self.rows]


class LongTokenizer:
    """Pretends every text is far too long."""
    def __call__(self, text):
        class Out:
            input_ids = list(range(5000))
        return Out()


def test_v1_rows_are_the_plain_training_text():
    rows, stats = build_training_rows(FakePart(50), augment=False, tokenizer=None)
    assert stats["recipe"] == "v1_plain" and stats["status_counts"] == {"plain": 50}
    for row in rows:
        assert row["text"] == build_training_text({"context": row["context"], "question": row["question"],
                                                   "answer": row["answer"]})
        assert row["context"] in CONTEXTS                       # nothing was changed


def test_v2_adds_tables_but_never_touches_question_or_answer():
    rows, stats = build_training_rows(FakePart(400), augment=True, tokenizer=None)
    assert stats["recipe"] == "v2_augmented"
    assert stats["status_counts"].get("augmented", 0) > 100
    assert stats["avg_tables_per_schema_after"] > stats["avg_tables_per_schema_before"]
    originals = FakePart(400).rows
    for row, original in zip(rows, originals):
        assert row["question"] == original["question"] and row["answer"] == original["answer"]
        assert sql_compiles(row["context"], row["answer"])      # the SQL is still valid on the new schema
        assert row["text"].startswith(build_prompt(row["context"], row["question"]))
        assert row["text"].endswith(row["answer"])


def test_v2_is_reproducible():
    a, _ = build_training_rows(FakePart(120), augment=True, tokenizer=None)
    b, _ = build_training_rows(FakePart(120), augment=True, tokenizer=None)
    assert [r["text"] for r in a] == [r["text"] for r in b]


def test_rows_that_would_be_truncated_fall_back_to_the_original_schema():
    rows, stats = build_training_rows(FakePart(200), augment=True, tokenizer=LongTokenizer())
    assert stats["rows_reverted_because_too_long"] > 0
    assert stats["status_counts"].get("augmented", 0) == 0
    originals = FakePart(200).rows
    assert all(r["context"] == o["context"] for r, o in zip(rows, originals))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} checks passed.")
