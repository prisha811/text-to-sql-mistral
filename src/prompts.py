"""
prompts.py - the ONE place where the prompt format is defined.

Both train_qlora.py (training) and generate_sql.py (inference) import from here,
so the model is always prompted exactly the way it was trained.

The template below is copied from the training notebook (LLM.ipynb, the second
`format_example`, which is the one that was actually used for training).

Run `python src/prompts.py` to check the template still matches the notebook output.
"""

import sqlite3

# Separator between CREATE TABLE statements when a schema has several tables.
# TO VERIFY ONCE: print a multi-table row of b-mc2/sql-create-context (see the
# check snippet in the Step 2 instructions) and change this if the dataset uses
# something different.
SCHEMA_SEPARATOR = "; "


def build_prompt(schema: str, question: str) -> str:
    """Prompt used at inference time. Ends right where the model should start writing SQL."""
    return (
        "You are a SQL generation assistant.\n\n"
        "Given a database schema and a natural language question, generate the correct SQL query.\n\n"
        f"### Database Schema:\n{schema}\n\n"
        f"### Question:\n{question}\n\n"
        "### SQL:\n"
    )


def build_training_text(example: dict) -> str:
    """Full training text = prompt + gold SQL. `example` has keys context/question/answer."""
    return build_prompt(example["context"], example["question"]) + example["answer"]


def _map_type(sqlite_type: str) -> str:
    """Map SQLite column types to the style used in the training data (INTEGER / VARCHAR)."""
    t = (sqlite_type or "").upper()
    if "INT" in t:
        return "INTEGER"
    if t == "" or "CHAR" in t or "TEXT" in t or "CLOB" in t:
        return "VARCHAR"
    return t  # REAL, FLOAT, NUMERIC, ... left as they are


def schema_from_sqlite(db_path) -> str:
    """
    Read the schema from a SQLite file and write it in the training-data style:
    single line per table, e.g. 'CREATE TABLE employees (employee_id INTEGER, name VARCHAR)'.
    Foreign keys are not included, because the training schemas did not contain them.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
            )
        ]
        statements = []
        for table in tables:
            columns = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
            column_sql = ", ".join(f"{col[1]} {_map_type(col[2])}" for col in columns)
            statements.append(f"CREATE TABLE {table} ({column_sql})")
        return SCHEMA_SEPARATOR.join(statements)
    finally:
        conn.close()


if __name__ == "__main__":
    # Expected text copied from the notebook output (LLM.ipynb, cell 35).
    sample = {
        "context": "CREATE TABLE table_14260687_3 (windows_phone VARCHAR, total VARCHAR)",
        "question": "How many million of windows phones were shipped during the quarter that shipped 216.2 million total?",
        "answer": 'SELECT windows_phone FROM table_14260687_3 WHERE total = "216.2"',
    }
    expected = (
        "You are a SQL generation assistant.\n\n"
        "Given a database schema and a natural language question, generate the correct SQL query.\n\n"
        "### Database Schema:\n"
        "CREATE TABLE table_14260687_3 (windows_phone VARCHAR, total VARCHAR)\n\n"
        "### Question:\n"
        "How many million of windows phones were shipped during the quarter that shipped 216.2 million total?\n\n"
        "### SQL:\n"
        'SELECT windows_phone FROM table_14260687_3 WHERE total = "216.2"'
    )
    assert build_training_text(sample) == expected, "Prompt template drifted from the training format!"
    print("OK: prompt template matches the training notebook output exactly.")
