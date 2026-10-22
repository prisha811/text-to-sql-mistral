"""
CPU-only checks for eval_in_distribution.py (no GPU, no dataset download). From the repo root:

    python tests/test_eval_in_distribution.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eval_in_distribution import (KNOWN_FIRST_TRAIN_QUESTION, check_sql_valid,  # noqa: E402
                                  check_split_matches_training, count_joins, norm_question,
                                  score_row, select_from_pool, select_rows, summarise)

# Taken from the schema-check output (multi-table row of sql-create-context).
MULTI_CONTEXT = ("CREATE TABLE department (creation VARCHAR, department_id VARCHAR); "
                 "CREATE TABLE management (department_id VARCHAR, head_id VARCHAR); "
                 "CREATE TABLE head (head_id VARCHAR, born_state VARCHAR)")
SINGLE_CONTEXT = "CREATE TABLE table_14260687_3 (windows_phone VARCHAR, total VARCHAR)"


def test_validity_uses_the_rows_own_schema():
    ok, _ = check_sql_valid(SINGLE_CONTEXT, 'SELECT windows_phone FROM table_14260687_3 WHERE total = "216.2"')
    assert ok  # double-quoted literal is accepted by SQLite, like in the gold data
    bad_col, err = check_sql_valid(SINGLE_CONTEXT, "SELECT nope FROM table_14260687_3")
    assert not bad_col and "no such column" in err
    bad_table, _ = check_sql_valid(SINGLE_CONTEXT, "SELECT * FROM employees")
    assert not bad_table
    assert not check_sql_valid(SINGLE_CONTEXT, "")[0]
    assert not check_sql_valid(SINGLE_CONTEXT, "SELEC x")[0]


def test_multi_table_validity_and_join_counting():
    sql = ("SELECT T1.creation FROM department AS T1 JOIN management AS T2 ON T1.department_id = T2.department_id "
           "JOIN head AS T3 ON T2.head_id = T3.head_id WHERE T3.born_state = 'Alabama'")
    assert check_sql_valid(MULTI_CONTEXT, sql)[0]
    assert count_joins(sql) == 2
    assert count_joins("select 1") == 0
    assert count_joins("SELECT * FROM a join b ON 1=1 Join c ON 1=1") == 2


def test_scoring_flags_extra_joins_and_exact_match_ignores_style():
    row = {"context": SINGLE_CONTEXT, "gold_sql": 'SELECT windows_phone FROM table_14260687_3 WHERE total = "216.2"',
           "group": "single_table",
           "base_sql": "SELECT windows_phone FROM table_14260687_3 WHERE total = '216.2'",
           "finetuned_sql": "SELECT windows_phone FROM table_14260687_3 AS T1 JOIN table_14260687_3 AS T2 ON 1=1"}
    base, tuned = score_row(row, "base"), score_row(row, "finetuned")
    assert base["exact_match"] and base["valid"] and not base["more_joins_than_gold"]
    assert not tuned["exact_match"] and tuned["more_joins_than_gold"] and tuned["joins"] == 1


def test_selection_excludes_training_rows_and_repeated_questions():
    n_train = 4
    questions = ["q0", "q1", "q2", "q3",            # "training" rows
                 "q1", "new a", "new b", "new c"]    # row 4 repeats a training question
    contexts = [SINGLE_CONTEXT] * 4 + [SINGLE_CONTEXT, SINGLE_CONTEXT, MULTI_CONTEXT, MULTI_CONTEXT]
    answers = ["SELECT total FROM table_14260687_3"] * 6 + \
              ["SELECT T1.creation FROM department AS T1", "SELECT T1.creation FROM department AS T1"]
    rows, info = select_rows(questions, contexts, answers, per_group=5, train_rows=n_train)
    indices = {r["index"] for r in rows}
    assert indices == {5, 6, 7}            # nothing < 4, and repeated question (row 4) removed
    assert info["removed_question_overlap_with_training_rows"] == 1
    assert {r["group"] for r in rows} == {"single_table", "multi_table"}

    rows2, _ = select_rows(questions, contexts, answers, per_group=5, train_rows=n_train, allow_overlap=True)
    assert 4 in {r["index"] for r in rows2}


def test_selection_is_reproducible_and_drops_bad_gold():
    n_train = 1
    questions = ["train"] + [f"question {i}" for i in range(30)]
    contexts = [SINGLE_CONTEXT] * 31
    answers = ["SELECT total FROM table_14260687_3"] * 30 + ["SELECT nope FROM table_14260687_3"]
    a, _ = select_rows(questions, contexts, answers, per_group=10, seed=7, train_rows=n_train)
    b, _ = select_rows(questions, contexts, answers, per_group=10, seed=7, train_rows=n_train)
    assert [r["index"] for r in a] == [r["index"] for r in b]
    assert all(r["gold_sql"] != "SELECT nope FROM table_14260687_3" for r in a)


def test_summarise_counts():
    gold = "SELECT total FROM table_14260687_3"
    rows = [
        {"context": SINGLE_CONTEXT, "gold_sql": gold, "group": "single_table",
         "base_sql": gold, "finetuned_sql": gold},
        {"context": SINGLE_CONTEXT, "gold_sql": gold, "group": "single_table",
         "base_sql": "SELECT nope FROM table_14260687_3", "finetuned_sql": gold + " AS a JOIN table_14260687_3 AS b ON 1=1"},
        {"context": MULTI_CONTEXT, "gold_sql": "SELECT creation FROM department", "group": "multi_table",
         "base_sql": "SELECT creation FROM department", "finetuned_sql": "SELECT creation FROM department"},
    ]
    s = summarise(rows)
    assert s["base"]["all"]["n"] == 3 and s["base"]["single_table"]["n"] == 2
    assert s["base"]["all"]["valid_sql"]["k"] == 2
    assert s["finetuned"]["single_table"]["more_joins_than_gold"]["k"] == 1
    assert s["finetuned"]["multi_table"]["exact_match"]["k"] == 1
    assert norm_question("  How   MANY? ") == "how many?"


def test_split_check_accepts_the_notebook_split_and_rejects_others():
    check_split_matches_training(KNOWN_FIRST_TRAIN_QUESTION)  # must not raise
    try:
        check_split_matches_training("Some other first question?")
    except RuntimeError as exc:
        assert "does NOT match" in str(exc)
    else:
        raise AssertionError("a different split must be rejected")


def test_validation_style_pool_has_both_groups_and_drops_trained_questions():
    train_questions = ["How many heads are there?", "List all departments."]
    questions = ["How many heads are there?", "Which heads were born in Alabama?", "What is the total?"]
    contexts = [MULTI_CONTEXT, MULTI_CONTEXT, SINGLE_CONTEXT]
    answers = ["SELECT COUNT(*) FROM head", "SELECT born_state FROM head", "SELECT total FROM table_14260687_3"]
    rows, info = select_from_pool(questions, contexts, answers, train_questions, per_group=10,
                                  source="validation")
    assert {r["group"] for r in rows} == {"single_table", "multi_table"}
    assert all(r["source"] == "validation" for r in rows)
    assert info["removed_question_overlap_with_training_rows"] == 1
    assert {r["index"] for r in rows} == {1, 2}   # position in the validation split


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} checks passed.")
