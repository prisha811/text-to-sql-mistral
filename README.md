# Text-to-SQL with QLoRA-fine-tuned Mistral 7B: an honest evaluation

**Prisha Singh** · B.Tech Computer Science (Data Science)

**Question:** does QLoRA fine-tuning improve Mistral 7B at turning a natural-language question plus a database schema into one correct, executable SQL query, compared with the base model?

**Short answer for the adapter I trained (v1):** on data like its training data, yes. On a 4-table database it had never seen, no: it scored far *below* the base model. This repository measures that with execution-based evaluation, traces the failure to a specific cause, and documents a fix (implemented, not yet trained).

---

## Key findings

| | Base Mistral 7B | Fine-tuned (QLoRA, v1) |
|---|---|---|
| **62-question test on my 4-table company database** (execution accuracy) | **55/62 = 88.7%** | **17/62 = 27.4%** |
| In-distribution check, 182 unseen dataset rows (valid SQL) | 169/182 = 92.9% | 180/182 = 98.9% |
| Same rows with 2 unneeded tables + 2 unneeded columns added: uses an unneeded table | 6/91 = 6.6% | 73/91 = 80.2% |

1. **The adapter did learn its training task.** On held-out rows of its own dataset it writes valid SQL more often than base and matches the dataset's SQL style far more often (exact match 64.3% vs 24.7%, which favours the adapter by construction, see Limitations).
2. **It fails on my company database.** It used a table the correct query does not need in 47 of 62 questions, and all four tables in 37. Base did so in 2 and 0.
3. **Likely cause (evidence, not proof):** in the training data, every table listed in a multi-table schema is used by the SQL answer (2001 of 2001 rows). The adapter appears to have learned "use every table you are given". When I add unneeded tables and columns to dataset rows it never trained on, it uses them 80% of the time; base does so 6.6% of the time.
4. **Fix implemented, not trained yet:** training on schemas with added distractor tables and columns (`src/augment.py`, `src/train_qlora.py --augment`). No results are claimed for it (see "Status of v2").

---

## How it works

```
question + schema  ->  prompt (fixed template)  ->  Mistral 7B (4-bit)  [+ QLoRA adapter]
                                                        |
                                                 generated SQL
                                                        |
        run on SQLite (read-only, 5 s limit)  ->  result rows
                                                        |
        compare with the result rows of the gold SQL  ->  correct / incorrect
```

The prompt template is defined once in `src/prompts.py` and used for both training and inference. `python src/prompts.py` checks that it reproduces the training text from the original notebook character for character.

## Training (v1 adapter)

Trained in `notebooks/LLM.ipynb` on a Colab Tesla T4.

| | |
|---|---|
| Base model | `mistralai/Mistral-7B-v0.1` |
| Data | `b-mc2/sql-create-context`, first 5,500 rows: 5,000 train / 500 validation (seed 42) |
| Quantization | 4-bit NF4, double quantization, float16 compute (QLoRA) |
| LoRA | r = 16, alpha = 32, dropout 0.05, on `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Trainable parameters | 13,631,488 (0.19% of the model) |
| Schedule | 1 epoch (625 steps), batch 1 x gradient accumulation 8, lr 2e-4 cosine, 100 warmup steps, max length 1024 |
| Result of training | average training loss about 0.49, validation loss about 0.40 (token-level losses, not SQL accuracy); about 75 minutes |

The training prompt schemas are small: 90.1% of the 5,500 rows have 1-2 tables, and only 71 rows (1.3%) have 4 or more.

## Evaluation method

* **Execution accuracy (main metric):** the generated query is executed on SQLite and its result set must equal the result set of the gold query. Row order is ignored; column count and order matter; floats are compared to 6 decimals. Invalid SQL is always a failure.
* **Valid SQL rate:** the query runs without any error.
* **Exact match (secondary):** normalised SQL text equals the gold SQL. Strict and style-dependent.
* **Statistics:** 95% Wilson confidence intervals and an exact sign test on the questions where only one model is correct.
* **Safety:** queries run on a read-only connection with a 5-second time limit.
* **Fair comparison:** both models get the same prompt, greedy decoding and the same output-cleaning rule (cut at `###`, keep the first statement). Raw outputs are saved next to the cleaned SQL. The base model is generated first, and the adapter is attached afterwards, because attaching it modifies the base model in place.

## Test database and questions

`data/company_v2.db`: 4 tables, `departments` (7 rows), `employees` (30), `projects` (10), `assignments` (56), with unequal department sizes, an empty department, employees without assignments, and no ties in "highest/lowest" questions.

`data/test_set.json`: 62 hand-written questions, each with gold SQL that is executed (answers are never typed by hand). Categories: simple 8, aggregation 8, join 8, subquery 10, group_by 5, having 4, multi_join 6, order_limit 4, self_join 2, misc 7.

**Leakage check** (`src/check_leakage.py`, against all 5,500 training rows): one question was flagged, "How many employees are there in total?" (similar to the training question "How many countries are there in total?"; the same SQL pattern also appears in training on another schema). I kept it and report it. Without it the scores are base 54/61 (88.5%) and adapter 16/61 (26.2%), so the conclusion does not change.

## Results

### 1. 62-question test, company database

| Model | Execution accuracy | 95% CI | Valid SQL | Exact match |
|---|---|---|---|---|
| Base Mistral 7B | 55/62 (88.7%) | 78.5%-94.4% | 61/62 (98.4%) | 26/62 (41.9%) |
| Fine-tuned (QLoRA, v1) | 17/62 (27.4%) | 17.9%-39.6% | 53/62 (85.5%) | 3/62 (4.8%) |

Paired: both correct 15, only base correct 40, only adapter correct 2, neither 5 (sign test p about 4e-10).

By category (correct / total):

| Category | n | Base | Adapter |
|---|---|---|---|
| simple | 8 | 7 | 0 |
| aggregation | 8 | 6 | 2 |
| join | 8 | 6 | 4 |
| subquery | 10 | 9 | 5 |
| group_by | 5 | 5 | 1 |
| having | 4 | 4 | 1 |
| multi_join | 6 | 6 | 2 |
| order_limit | 4 | 3 | 0 |
| self_join | 2 | 2 | 0 |
| misc | 7 | 7 | 2 |

Wrong answers by type: base 4 wrong rows, 2 wrong columns, 1 invalid SQL; adapter 36 wrong rows, 9 invalid SQL.

**Join behaviour** (counted from the raw outputs): no gold query uses more than 2 JOINs. The adapter used 3 or more JOINs in 38 queries, more JOINs than the gold query in 48, a table the gold query does not use in 47, and all four tables in 37. Base: 0, 0, 2 and 0.

Example: for "List the names of all departments." the correct query is `SELECT department_name FROM departments`. The adapter produced a 4-table join with `WHERE T4.hours > 100`.

Speed on a T4, one run, generation only: 142 s for base and 574 s for the adapter on 62 questions (the adapter's outputs are longer and the adapter is not merged into the base weights).

### 2. In-distribution check (rows the adapter never trained on)

Rows come from the 500 validation rows of the training notebook; the split was rebuilt and verified against the notebook's first training example. 182 rows (100 single-table, 82 multi-table).

| Group | Model | Valid SQL | Exact match* | More JOINs than gold |
|---|---|---|---|---|
| single-table (100) | Base | 99.0% | 45.0% | 0.0% |
| | Adapter | 100.0% | 87.0% | 0.0% |
| multi-table (82) | Base | 85.4% | 0.0% | 0.0% |
| | Adapter | 97.6% | 36.6% | 2.4% (2/82) |
| all (182) | Base | 92.9% | 24.7% | 0.0% |
| | Adapter | 98.9% | 64.3% | 1.1% |

\*These rows have no table contents, so results cannot be compared. Exact match favours the adapter because the gold SQL is in the style it was trained on. The adapter shows no over-joining here, where every listed table is needed.

### 3. Distractor test (diagnosis)

The same kind of rows, but each schema gets 2 unneeded tables and 2 unneeded columns borrowed from other rows (91 rows: 50 single-table, 41 multi-table). The gold SQL is still correct because it never uses them.

| Model | Valid SQL | Exact match* | Uses an added table | Uses an added column |
|---|---|---|---|---|
| Base | 83.5% | 18.7% | 6/91 (6.6%) | 10/91 (11.0%) |
| Adapter | 72.5% | 8.8% | 73/91 (80.2%) | 73/91 (80.2%) |

Console output of runs 2 and 3 is in `docs/run_logs.md`.

## Status of v2 (not trained)

`python src/train_qlora.py --augment` trains the same recipe with one change: 75% of training schemas get up to 3 distractor tables and up to 4 distractor columns (`src/augment.py`; gold SQL still compiles on the new schema; validation rows are untouched). The data build was checked on the real dataset (3,351 augmented, 1,370 unchanged, 279 kept as original because the SQL did not compile on the new schema). Training was **not completed**: the first attempt crashed because the current TRL release needs a GPU with compute capability 8.0+ (the T4 is 7.5), so `train_qlora.py` was rewritten to use the plain `transformers` Trainer, and GPU quota ran out before the retry. **No v2 results are reported.**

## Limitations

* One synthetic schema and 62 questions written by me. Results are specific to this data and schema.
* The fix was designed after I had read the adapter's failures on the test set, so the test set is no longer unseen for that decision. A v2 result should be read with that in mind.
* Execution accuracy can give false positives (a wrong query can return the right rows by luck) and some of base's passes may be lucky guesses (for example `department_id = 1` for Engineering).
* Exact match is style-dependent and favours the fine-tuned model on dataset rows.
* The in-distribution rows were used to log validation loss during training (no checkpoint was selected on them), and the dataset contains similar questions across rows, so those numbers may be optimistic.
* One run per configuration, greedy decoding, no repeated seeds. Timings are from a single run.
* The per-row files of the in-distribution and distractor runs are not in the repository; the tables above are copied from the console output.

## Project history

An earlier version of this repository reported 100% for both models. Those numbers came from hand-written SQL stored in a file, not from the models, so that file was removed. The first real test (10 questions on the original 8-employee database) gave base 9/10 and adapter 6/10, which led to the larger database and test set above.

## Repository layout

```
data/
  build_company_v2.py   builds company_v2.db (4 tables)
  make_test_set.py      writes + validates test_set.json (62 questions with gold SQL)
  company_v2.db, test_set.json
  company.db, database.py, evaluation_questions.py   original 8-employee toy database (v1), kept for history
models/adapter/         the trained QLoRA adapter (v1)
notebooks/LLM.ipynb     the Colab notebook the adapter was trained in
src/
  prompts.py            the one prompt template (training + inference)
  model.py              loads base model and adapter separately
  generate_sql.py       base then adapter -> outputs JSON
  evaluate.py           execution accuracy, valid SQL, exact match, CIs, paired test
  check_leakage.py      test questions vs the 5,500 training rows
  eval_in_distribution.py   in-distribution + distractor tests
  augment.py            distractor tables/columns for v2 training
  train_qlora.py        training script (v1 recipe, or --augment for v2)
tests/                  CPU-only tests
outputs/                result files of the 62-question run and the leakage report
docs/run_logs.md        console output of the in-distribution and distractor runs
```

## Reproduce

Tests (CPU, seconds): `python tests/test_generation_utils.py`, `test_evaluate.py`, `test_eval_in_distribution.py`, `test_augment.py`, `test_train_data.py`.

Generation needs a CUDA GPU (a 16 GB Tesla T4 was enough). Tested with torch 2.11, transformers 5.20, peft 0.21, bitsandbytes 0.50, accelerate 1.15.

```bash
pip install -r requirements.txt
python data/build_company_v2.py
python data/make_test_set.py
python src/check_leakage.py                                   # needs internet for the dataset
python src/generate_sql.py --questions data/test_set.json --db data/company_v2.db --out outputs/generated_sql_v2.json   # GPU
python src/evaluate.py --input outputs/generated_sql_v2.json --db data/company_v2.db --out-dir outputs/test62           # CPU
python src/eval_in_distribution.py                                                                                      # GPU
python src/eval_in_distribution.py --distractor-tables 2 --distractor-cols 2 --per-group 50 --out-dir outputs/distractor_v1   # GPU
```

To try v2: `python src/train_qlora.py --augment --max-steps 5 --output-dir /content/smoke` (smoke test), then `python src/train_qlora.py --augment --output-dir <folder> --save-steps 100`, then rerun the commands above with `--adapter <folder>`.

## Next steps

Train v2 and report it next to v1 and base; try giving the model only the tables a question needs (schema pruning) at inference time; test on a second schema; merge the adapter into the weights for faster inference.
