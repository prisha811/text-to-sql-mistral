"""
CPU-only checks for evaluate.py. Run from the repo root:

    python tests/test_evaluate.py

Windows note: SQLite keeps the database file open until the connection is closed, and Windows
refuses to delete an open file. Every test therefore opens the DB through open_db(), which
closes the connection before the temporary folder is removed.
"""

import sqlite3
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evaluate import (classify, connect_readonly, evaluate_record, normalize_sql_text,  # noqa: E402
                      run_sql, sign_test_p, summarise, wilson_interval)


def make_db(tmp):
    path = Path(tmp) / "t.db"
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE employees (employee_id INTEGER PRIMARY KEY, name TEXT, salary INTEGER)")
        conn.executemany("INSERT INTO employees VALUES (?, ?, ?)", [(1, "A", 100), (2, "B", 200), (3, "C", 300)])
        conn.commit()
    finally:
        conn.close()
    return path


@contextmanager
def open_db():
    """Temp database + read-only connection, always closed before the folder is deleted."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # Python 3.10+
        conn = connect_readonly(make_db(tmp))
        try:
            yield conn
        finally:
            conn.close()


def record(gold, base, finetuned, **extra):
    return {"id": 1, "question": "q", "gold_sql": gold, "base_sql": base, "finetuned_sql": finetuned, **extra}


def test_row_order_is_ignored_but_columns_matter():
    with open_db() as conn:
        gold = "SELECT name FROM employees ORDER BY salary ASC"
        reordered = "SELECT name FROM employees ORDER BY salary DESC"
        extra_col = "SELECT name, salary FROM employees"
        e = evaluate_record(conn, record(gold, reordered, extra_col))
        assert e["base"]["correct"] is True
        assert e["finetuned"]["correct"] is False
        assert e["finetuned"]["failure_type"] == "wrong_columns"


def test_invalid_sql_is_a_failure_and_not_valid():
    with open_db() as conn:
        e = evaluate_record(conn, record("SELECT COUNT(*) FROM employees", "SELEC oops", "SELECT * FROM nope"))
        for model in ("base", "finetuned"):
            assert e[model]["valid"] is False and e[model]["correct"] is False
            assert e[model]["failure_type"] == "invalid_sql"
        assert run_sql(conn, "SELECT 1; SELECT 2")["ok"] is False  # two statements
        assert run_sql(conn, "")["ok"] is False


def test_database_is_read_only():
    with open_db() as conn:
        assert run_sql(conn, "DELETE FROM employees")["ok"] is False
        assert run_sql(conn, "SELECT COUNT(*) FROM employees")["rows"] == [(3,)]


def test_timeout_stops_runaway_query():
    with open_db() as conn:
        runaway = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT COUNT(*) FROM c"
        result = run_sql(conn, runaway, timeout=0.5)
        assert result["ok"] is False and "interrupted" in result["error"].lower()


def test_expected_result_fallback_and_float_tolerance():
    with open_db() as conn:
        rec = {"id": 1, "question": "avg", "expected_result": [[200.0]],
               "base_sql": "SELECT AVG(salary) FROM employees", "finetuned_sql": "SELECT MAX(salary) FROM employees"}
        e = evaluate_record(conn, rec)
        assert e["base"]["correct"] is True and e["finetuned"]["correct"] is False
        assert e["base"]["exact_match"] is None  # no gold SQL -> exact match not available


def test_exact_match_normalisation_keeps_literal_case():
    a = normalize_sql_text('SELECT  name FROM employees WHERE name = "Bob";')
    b = normalize_sql_text("select name from employees where name='Bob'")
    c = normalize_sql_text("select name from employees where name='bob'")
    assert a == b and a != c


def test_empty_gold_is_flagged():
    with open_db() as conn:
        e = evaluate_record(conn, record("SELECT name FROM employees WHERE salary > 9999",
                                         "SELECT name FROM employees WHERE salary > 9999", "SELECT 1"))
        assert e["gold_empty"] is True
        assert summarise([e])["num_empty_gold"] == 1


def test_statistics():
    lo, hi = wilson_interval(5, 10)
    assert 0.23 < lo < 0.25 and 0.75 < hi < 0.77
    assert sign_test_p(0, 0) == 1.0
    assert abs(sign_test_p(0, 5) - 0.0625) < 1e-9
    assert sign_test_p(3, 3) == 1.0
    assert classify({"ok": True, "rows": [(1,)], "n_cols": 1}, [(1,)], 1) == (True, "ok")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} checks passed.")
