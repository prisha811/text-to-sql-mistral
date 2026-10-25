"""
Regression test for a real bug: `generate_sql.py --adapter X` used to record X in the output metadata
but silently attach the DEFAULT adapter. This test runs generate_sql.main() with a fake `model` module
(no GPU needed) and checks that the adapter that is attached is the one that was asked for.

Run from the repo root:
    python tests/test_generate_sql_adapter.py
"""

import json
import sqlite3
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import generate_sql  # noqa: E402

DEFAULT_ADAPTER = Path("/default/adapter")


class FakeTuned:
    is_tuned = True


def make_fake_model_module(calls):
    fake = types.ModuleType("model")
    fake.ADAPTER_PATH = DEFAULT_ADAPTER
    fake.BASE_MODEL = "fake/base-model"
    fake.load_tokenizer = lambda: object()
    fake.load_base_model = lambda: object()
    fake.count_lora_layers = lambda m: 128 if getattr(m, "is_tuned", False) else 0

    def attach_adapter(base, path=DEFAULT_ADAPTER):
        calls["attached"] = Path(path)
        return FakeTuned()

    fake.attach_adapter = attach_adapter
    fake.adapter_fingerprint = lambda path: "abc123abc123"
    return fake


def run_main(extra_args):
    """Run generate_sql.main() with fakes. Returns (adapter path that was attached, metadata written)."""
    calls = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        db = tmp / "t.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE employees (employee_id INTEGER, name TEXT)")
        conn.commit()
        conn.close()
        questions = tmp / "q.json"
        questions.write_text(json.dumps([{"question": "How many employees?"}, {"question": "List names."}]))
        out = tmp / "out.json"

        saved = (sys.modules.get("model"), generate_sql.generate_one, sys.argv)
        sys.modules["model"] = make_fake_model_module(calls)
        generate_sql.generate_one = lambda model, tokenizer, prompt, max_new_tokens: "SELECT 1"
        sys.argv = ["generate_sql.py", "--questions", str(questions), "--db", str(db),
                    "--out", str(out)] + extra_args
        try:
            generate_sql.main()
        finally:
            if saved[0] is None:
                sys.modules.pop("model", None)
            else:
                sys.modules["model"] = saved[0]
            generate_sql.generate_one, sys.argv = saved[1], saved[2]
        meta = json.loads(out.read_text())["metadata"]
    return calls["attached"], meta


def test_the_adapter_given_by_the_flag_is_the_one_attached():
    attached, meta = run_main(["--adapter", "/some/other/v2"])
    assert attached == Path("/some/other/v2"), f"attached {attached} instead of the requested adapter"
    assert meta["adapter_path"] == str(Path("/some/other/v2"))
    assert meta["adapter_sha256_12"] == "abc123abc123"       # provenance is recorded


def test_without_the_flag_the_default_adapter_is_attached():
    attached, meta = run_main([])
    assert attached == DEFAULT_ADAPTER
    assert meta["adapter_path"] == str(DEFAULT_ADAPTER)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} checks passed.")
