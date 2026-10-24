"""
augment.py - add DISTRACTOR tables and columns to a schema, leaving the question and SQL unchanged.

Why this exists
---------------
In the training data (b-mc2/sql-create-context) every table listed in a multi-table schema is used
by the SQL answer (checked: 2001 of 2001 rows), and the schemas contain only the columns the query
needs. The adapter therefore learned "use every table you are given". On the 4-table company
database it joined all tables and added irrelevant filters, even for "List the names of all
departments".

The fix tested here: show the model schemas that contain tables and columns the query does NOT
use, so it has to decide what is relevant.

How it works
------------
  * Distractor tables are copied from OTHER rows' schemas. Half of the time a table that shares a
    column name with the row's own tables is preferred (so it looks joinable, like the unused
    tables in a real database); otherwise a random table is used.
  * Extra columns are borrowed from other tables and added to the row's own tables at random
    positions.
  * Names never collide with names already in the schema, and table order is shuffled.
  * After augmenting, the gold SQL must still compile on the new schema; if it does not, the
    original schema is kept (status "invalid_fallback").
  * Everything is deterministic: the random choices depend only on (seed, row id).

The numbers below (75% of rows augmented, up to 3 extra tables and 4 extra columns) are design
choices, not tuned values.
"""

import random
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field

P_AUGMENT = 0.75                                        # share of training rows that get distractors
TABLE_COUNT_WEIGHTS = {0: 2.0, 1: 4.0, 2: 3.0, 3: 1.5}  # how many distractor tables to add
COLUMN_COUNT_WEIGHTS = {0: 2.0, 1: 2.0, 2: 2.0, 3: 1.5, 4: 1.0}  # how many extra columns to add
P_RELATED = 0.5                                         # prefer tables that share a column name
MAX_TRIES = 25

_STATEMENT = re.compile(r"CREATE TABLE\s+(\w+)\s*\((.*)\)", re.S | re.I)


# --------------------------------------------------------------------------- parsing / rendering
def parse_context(context):
    """
    'CREATE TABLE a (x VARCHAR, y INTEGER); CREATE TABLE b (z VARCHAR)'
    -> [('a', [('x', 'VARCHAR'), ('y', 'INTEGER')]), ('b', [('z', 'VARCHAR')])]
    Returns None if anything is not a plain 'name TYPE' column list (such rows are never augmented).
    """
    tables = []
    for statement in context.split(";"):
        statement = statement.strip()
        if not statement:
            continue
        match = _STATEMENT.fullmatch(statement)
        if not match:
            return None
        columns = []
        for part in match.group(2).split(","):
            bits = part.strip().split()
            if len(bits) != 2:
                return None
            columns.append((bits[0], bits[1]))
        tables.append((match.group(1), columns))
    return tables or None


def render_context(tables):
    return "; ".join(
        f"CREATE TABLE {name} ({', '.join(f'{col} {typ}' for col, typ in columns)})"
        for name, columns in tables
    )


def sql_compiles(context, sql):
    """True if `sql` compiles on EMPTY tables created from `context`."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(context.rstrip().rstrip(";") + ";")
        conn.execute("EXPLAIN " + sql)
        return True
    except Exception:
        return False
    finally:
        conn.close()


# --------------------------------------------------------------------------- augmentation
@dataclass
class AugmentResult:
    context: str
    added_tables: list = field(default_factory=list)    # names of distractor tables
    added_columns: list = field(default_factory=list)   # names of extra columns
    status: str = "kept"   # augmented | kept | unparsed | invalid_fallback


def _weighted(rng, weights):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


class Augmenter:
    def __init__(self, pool_contexts, seed=42):
        """`pool_contexts`: schemas that distractor tables and columns are borrowed from."""
        self.seed = seed
        self.pool = []                     # (table_name, [(col, type), ...])
        seen = set()
        for context in pool_contexts:
            parsed = parse_context(context)
            if not parsed:
                continue
            for name, columns in parsed:
                key = (name.lower(), tuple(col.lower() for col, _ in columns))
                if key not in seen:
                    seen.add(key)
                    self.pool.append((name, columns))
        self.by_column = defaultdict(list)
        for i, (_, columns) in enumerate(self.pool):
            for col, _ in columns:
                self.by_column[col.lower()].append(i)
        self.pool_columns = [(col, typ) for _, columns in self.pool for col, typ in columns]

    def _rng(self, row_id):
        return random.Random(f"{self.seed}:{row_id}")   # string seeds are stable across runs

    # ---- public API
    def augment_random(self, context, answer, row_id):
        """Training-time policy: only some rows are augmented, with random amounts."""
        parsed = parse_context(context)
        if parsed is None:
            return AugmentResult(context, status="unparsed")
        rng = self._rng(row_id)
        if rng.random() >= P_AUGMENT:
            return AugmentResult(context, status="kept")
        n_tables = _weighted(rng, TABLE_COUNT_WEIGHTS)
        n_cols = _weighted(rng, COLUMN_COUNT_WEIGHTS)
        return self._finish(context, answer, parsed, rng, n_tables, n_cols)

    def augment_fixed(self, context, answer, row_id, n_tables, n_cols):
        """Evaluation-time policy: always add exactly this many (if enough names are available)."""
        parsed = parse_context(context)
        if parsed is None:
            return AugmentResult(context, status="unparsed")
        return self._finish(context, answer, parsed, self._rng(row_id), n_tables, n_cols)

    # ---- internals
    def _finish(self, context, answer, parsed, rng, n_tables, n_cols):
        tables = [(name, list(columns)) for name, columns in parsed]
        added_columns = self._add_columns(rng, tables, n_cols)
        added_tables = self._add_tables(rng, tables, n_tables)
        if not added_columns and not added_tables:
            return AugmentResult(context, status="kept")
        rng.shuffle(tables)
        new_context = render_context(tables)
        if not sql_compiles(new_context, answer):
            return AugmentResult(context, status="invalid_fallback")
        return AugmentResult(new_context, added_tables, added_columns, "augmented")

    def _add_columns(self, rng, tables, k):
        """Add k borrowed columns to the row's OWN tables. Mutates `tables`; returns added names."""
        if k <= 0 or not self.pool_columns:
            return []
        taken = {col.lower() for _, columns in tables for col, _ in columns}
        added = []
        for _ in range(k):
            for _try in range(MAX_TRIES):
                col, typ = self.pool_columns[rng.randrange(len(self.pool_columns))]
                if col.lower() not in taken:
                    taken.add(col.lower())
                    columns = tables[rng.randrange(len(tables))][1]
                    columns.insert(rng.randint(0, len(columns)), (col, typ))
                    added.append(col)
                    break
        return added

    def _add_tables(self, rng, tables, n):
        """Append n borrowed tables. Mutates `tables`; returns added table names."""
        if n <= 0 or not self.pool:
            return []
        taken = {name.lower() for name, _ in tables}
        own_columns = {col.lower() for _, columns in tables for col, _ in columns}
        related = sorted({i for col in own_columns for i in self.by_column.get(col, [])})
        added = []
        for _ in range(n):
            for _try in range(MAX_TRIES):
                if related and rng.random() < P_RELATED:
                    index = related[rng.randrange(len(related))]
                else:
                    index = rng.randrange(len(self.pool))
                name, columns = self.pool[index]
                if name.lower() not in taken:
                    taken.add(name.lower())
                    tables.append((name, list(columns)))
                    added.append(name)
                    break
        return added
