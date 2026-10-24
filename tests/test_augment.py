"""
CPU-only checks for augment.py. Run from the repo root:

    python tests/test_augment.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import augment  # noqa: E402
from augment import Augmenter, parse_context, render_context, sql_compiles  # noqa: E402

# Schemas copied from real sql-create-context rows seen in the Colab outputs.
POOL = [
    "CREATE TABLE department (creation VARCHAR, department_id VARCHAR); CREATE TABLE management (department_id VARCHAR, head_id VARCHAR); CREATE TABLE head (head_id VARCHAR, born_state VARCHAR)",
    "CREATE TABLE professor (emp_num VARCHAR, dept_code VARCHAR); CREATE TABLE department (dept_code VARCHAR, dept_name VARCHAR); CREATE TABLE CLASS (prof_num VARCHAR); CREATE TABLE employee (emp_fname VARCHAR, emp_num VARCHAR)",
    "CREATE TABLE city (city VARCHAR, city_id VARCHAR); CREATE TABLE hosting_city (host_city VARCHAR); CREATE TABLE temperature (city_id VARCHAR, Mar INTEGER, Dec VARCHAR)",
    "CREATE TABLE table_14260687_3 (windows_phone VARCHAR, total VARCHAR)",
    "CREATE TABLE table_name_77 (player VARCHAR, score VARCHAR, year INTEGER)",
    "CREATE TABLE table_name_12 (team VARCHAR, wins INTEGER, losses INTEGER)",
    "CREATE TABLE singer (name VARCHAR, age VARCHAR, country VARCHAR)",
    "CREATE TABLE concert (concert_id VARCHAR, year VARCHAR, stadium_id VARCHAR); CREATE TABLE stadium (stadium_id VARCHAR, capacity INTEGER)",
]
SINGLE = "CREATE TABLE singer (name VARCHAR, age VARCHAR, country VARCHAR)"
SINGLE_SQL = "SELECT name FROM singer WHERE age > 30"
MULTI = "CREATE TABLE department (creation VARCHAR, department_id VARCHAR); CREATE TABLE management (department_id VARCHAR, head_id VARCHAR)"
MULTI_SQL = ("SELECT T1.creation FROM department AS T1 JOIN management AS T2 "
             "ON T1.department_id = T2.department_id")


def test_parse_and_render_roundtrip_on_real_schemas():
    for context in POOL:
        parsed = parse_context(context)
        assert parsed is not None
        assert render_context(parsed) == context
    assert render_context(parse_context("CREATE TABLE t (a VARCHAR(20))")) == "CREATE TABLE t (a VARCHAR(20))"
    assert parse_context("CREATE TABLE t (a INTEGER PRIMARY KEY)") is None   # not a plain 'name TYPE' column
    assert parse_context("CREATE TABLE t (a DECIMAL(10,2))") is None
    assert parse_context("DROP TABLE x") is None
    assert parse_context("") is None


def test_augmentation_only_adds_and_keeps_the_sql_valid():
    aug = Augmenter(POOL, seed=1)
    for row_id in range(200):
        for context, sql in ((SINGLE, SINGLE_SQL), (MULTI, MULTI_SQL)):
            result = aug.augment_fixed(context, sql, row_id, n_tables=2, n_cols=2)
            assert result.status in ("augmented", "kept")
            assert sql_compiles(result.context, sql)               # gold SQL still compiles
            new = dict(parse_context(result.context))
            for name, columns in parse_context(context):           # nothing original was removed
                assert name in new
                assert set(columns) <= set(new[name])
            names = [n.lower() for n in new]
            assert len(names) == len(set(names))                   # no duplicate table names
            for added in result.added_tables:
                assert added in new
            own_names = [name for name, _ in parse_context(context)]
            own_columns = [c.lower() for name in own_names for c, _ in new[name]]
            for added in result.added_columns:                     # never duplicates a column of the row's own tables
                assert own_columns.count(added.lower()) == 1


def test_single_table_rows_get_distractor_tables_and_columns():
    aug = Augmenter(POOL, seed=2)
    result = aug.augment_fixed(SINGLE, SINGLE_SQL, 7, n_tables=3, n_cols=2)
    assert result.status == "augmented"
    assert len(result.added_tables) == 3 and len(result.added_columns) == 2
    assert result.context.count("CREATE TABLE") == 4


def test_deterministic_for_same_seed_and_row_id():
    a = Augmenter(POOL, seed=5).augment_random(MULTI, MULTI_SQL, 11)
    b = Augmenter(POOL, seed=5).augment_random(MULTI, MULTI_SQL, 11)
    c = Augmenter(POOL, seed=6).augment_random(MULTI, MULTI_SQL, 11)
    assert (a.context, a.added_tables, a.added_columns) == (b.context, b.added_tables, b.added_columns)
    # a different seed will normally give a different result somewhere across rows
    differs = any(Augmenter(POOL, seed=5).augment_random(MULTI, MULTI_SQL, i).context !=
                  Augmenter(POOL, seed=6).augment_random(MULTI, MULTI_SQL, i).context for i in range(30))
    assert differs and c is not None


def test_training_policy_augments_roughly_three_quarters_of_rows():
    aug = Augmenter(POOL, seed=3)
    statuses = [aug.augment_random(SINGLE, SINGLE_SQL, i).status for i in range(2000)]
    share = statuses.count("augmented") / len(statuses)
    # P_AUGMENT is 0.75, but rows that draw "0 tables and 0 columns" stay unchanged -> a bit lower
    assert 0.55 < share < 0.80, share
    assert set(statuses) <= {"augmented", "kept"}


def test_unparsable_rows_are_left_alone_and_invalid_results_fall_back():
    aug = Augmenter(POOL, seed=4)
    weird = "CREATE TABLE t (a INTEGER PRIMARY KEY)"
    result = aug.augment_random(weird, "SELECT a FROM t", 1)
    assert result.status == "unparsed" and result.context == weird

    original = augment.sql_compiles
    try:
        augment.sql_compiles = lambda context, sql: False          # force the safety net
        fallback = aug.augment_fixed(SINGLE, SINGLE_SQL, 3, n_tables=2, n_cols=1)
    finally:
        augment.sql_compiles = original
    assert fallback.status == "invalid_fallback"
    assert fallback.context == SINGLE and not fallback.added_tables and not fallback.added_columns


def test_pool_borrowing_excludes_the_rows_own_table_names():
    aug = Augmenter(POOL, seed=8)
    for row_id in range(100):
        result = aug.augment_fixed(MULTI, MULTI_SQL, row_id, n_tables=3, n_cols=0)
        assert "department" not in [t.lower() for t in result.added_tables]
        assert "management" not in [t.lower() for t in result.added_tables]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} checks passed.")
