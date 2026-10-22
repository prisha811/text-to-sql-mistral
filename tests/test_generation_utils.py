"""
CPU-only checks (no GPU, no model download). Run from the repo root:

    python tests/test_generation_utils.py
"""

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from generate_sql import extract_sql, load_questions  # noqa: E402
from prompts import build_prompt, build_training_text, schema_from_sqlite  # noqa: E402


def test_prompt_matches_training_format():
    example = {
        "context": "CREATE TABLE t (a VARCHAR)",
        "question": "How many rows?",
        "answer": "SELECT COUNT(*) FROM t",
    }
    prompt = build_prompt(example["context"], example["question"])
    # The inference prompt must be an exact prefix of the training text.
    assert build_training_text(example).startswith(prompt)
    assert prompt.endswith("### SQL:\n")
    assert "generate the correct SQL query." in prompt  # not "exactly ONE SQL query"


def test_schema_uses_training_style():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "t.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE departments (department_id INTEGER PRIMARY KEY, department_name TEXT)")
        conn.execute(
            "CREATE TABLE employees (employee_id INTEGER PRIMARY KEY, name TEXT, department_id INTEGER, "
            "salary INTEGER, FOREIGN KEY (department_id) REFERENCES departments(department_id))"
        )
        conn.commit()
        conn.close()
        schema = schema_from_sqlite(db)
    assert schema == (
        "CREATE TABLE departments (department_id INTEGER, department_name VARCHAR); "
        "CREATE TABLE employees (employee_id INTEGER, name VARCHAR, department_id INTEGER, salary INTEGER)"
    )


def test_extract_sql_handles_rambling_base_output():
    raw = "SELECT name\nFROM employees\nWHERE salary > 80000;\n\n### Question:\nSomething else\n\n### SQL:\nSELECT 1"
    assert extract_sql(raw) == "SELECT name\nFROM employees\nWHERE salary > 80000"


def test_extract_sql_plain_and_fenced():
    assert extract_sql("  SELECT COUNT(*) FROM employees\n") == "SELECT COUNT(*) FROM employees"
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("") == ""


def test_load_questions_json_and_limit():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "q.json"
        path.write_text(json.dumps([{"question": "a"}, {"question": "b", "id": "x"}, {"question": "c"}]))
        items = load_questions(str(path), limit=2)
    assert [i["question"] for i in items] == ["a", "b"]
    assert items[0]["id"] == 1 and items[1]["id"] == "x"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} checks passed.")
