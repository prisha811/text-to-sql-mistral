# Console output of the diagnostic runs

These tables are copied from the console output of the runs (Colab, Tesla T4). The per-row files of these runs are not in the repository. Re-running the commands in the README regenerates them.

## In-distribution check on validation rows

Command: `python src/eval_in_distribution.py`

```
Split check passed: the rebuilt split starts with the notebook's first training example.
Selected 182 unseen rows (100 single-table, 82 multi-table). {'source': 'validation', 'candidates_single': 337, 'candidates_multi': 161, 'removed_question_overlap_with_training_rows': 2, 'dropped_gold_not_compiling': 18, 'seed': 42, 'per_group_requested': 100}

[single_table]
  Base Mistral 7B      Valid 99/100 (99.0%)    Exact match 45/100 (45.0%)    More JOINs than gold 0/100 (0.0%)
  Fine-tuned (QLoRA)   Valid 100/100 (100.0%)  Exact match 87/100 (87.0%)    More JOINs than gold 0/100 (0.0%)

[multi_table]
  Base Mistral 7B      Valid 70/82 (85.4%)     Exact match 0/82 (0.0%)       More JOINs than gold 0/82 (0.0%)
  Fine-tuned (QLoRA)   Valid 80/82 (97.6%)     Exact match 30/82 (36.6%)     More JOINs than gold 2/82 (2.4%)

[all]
  Base Mistral 7B      Valid 169/182 (92.9%)   Exact match 45/182 (24.7%)    More JOINs than gold 0/182 (0.0%)
  Fine-tuned (QLoRA)   Valid 180/182 (98.9%)   Exact match 117/182 (64.3%)   More JOINs than gold 2/182 (1.1%)
```

Generation time: base 449 s, adapter 789 s for 182 rows.

## Distractor test (current adapter, v1)

Command: `python src/eval_in_distribution.py --distractor-tables 2 --distractor-cols 2 --per-group 50 --out-dir outputs/distractor_v1`

```
Distractors added: {'tables': 2, 'columns': 2, 'status_counts': {'augmented': 91}}
Selected 91 unseen rows (50 single-table, 41 multi-table).

[single_table]
  Base Mistral 7B      Valid 46/50 (92.0%)  Exact 17/50 (34.0%)  More JOINs 0/50   Uses added TABLE 5/50 (10.0%)   Uses added COLUMN 5/50 (10.0%)
  Fine-tuned (QLoRA)   Valid 37/50 (74.0%)  Exact 3/50 (6.0%)    More JOINs 43/50  Uses added TABLE 43/50 (86.0%)  Uses added COLUMN 44/50 (88.0%)

[multi_table]
  Base Mistral 7B      Valid 30/41 (73.2%)  Exact 0/41 (0.0%)    More JOINs 0/41   Uses added TABLE 1/41 (2.4%)    Uses added COLUMN 5/41 (12.2%)
  Fine-tuned (QLoRA)   Valid 29/41 (70.7%)  Exact 5/41 (12.2%)   More JOINs 30/41  Uses added TABLE 30/41 (73.2%)  Uses added COLUMN 29/41 (70.7%)

[all]
  Base Mistral 7B      Valid 76/91 (83.5%)  Exact 17/91 (18.7%)  More JOINs 0/91   Uses added TABLE 6/91 (6.6%)    Uses added COLUMN 10/91 (11.0%)
  Fine-tuned (QLoRA)   Valid 66/91 (72.5%)  Exact 8/91 (8.8%)    More JOINs 73/91  Uses added TABLE 73/91 (80.2%)  Uses added COLUMN 73/91 (80.2%)
```

Generation time: base 222 s, adapter 790 s for 91 rows.

## v2 training data build (dry run)

Command: `python src/train_qlora.py --augment --dry-run`

```
recipe: v2_augmented | training rows: 5000 | validation rows: 500
status_counts: augmented 3351, kept 1370, invalid_fallback 279
avg_tables_per_schema: 1.48 before, 2.433 after
rows_reverted_because_too_long: 0
```

## v2 training attempts

1. The first full-training attempt (TRL) stopped at step 0 with `RuntimeError: gemm input type at::BFloat16 and output type float is only supported for CUDA devices with compute capability 8.0 or higher`, raised inside TRL's `chunked_logprob` kernel. The Colab T4 has compute capability 7.5. `train_qlora.py` was changed to use the plain `transformers` Trainer.
2. Smoke test of the new path (`--augment --max-steps 5`): completed and saved an adapter (not usable, only 5 steps). 5 steps took 2 min 16 s (about 27 s per step).
3. Full run (`--augment --save-steps 100`): reached step 91 of 625 after 42 min 58 s (about 28 s per step on average; the progress bar's smoothed estimate was 30.6 s per step with 4 h 32 min remaining, so a full run is roughly 5 hours) when the free Colab GPU quota ended. Training loss logged every 10 steps fell from 1.747 (step 10) to 0.754 (step 90). Checkpoints were saved every 100 steps, so none existed and the run could not be resumed.

## Pilot: 10 questions on the original 8-employee database

Console output of `python src/evaluate.py` on the first real generation run (original toy database, 10 hand-written questions with expected results; these files were removed from the repository afterwards, and the database was too small to be reliable, for example a wrong `salary > 10000` filter returned the right count).

```
Model                  Execution accuracy   95% CI        Valid SQL
Base Mistral 7B        9/10 = 90.0%         59.6%-98.2%   10/10
Fine-tuned (QLoRA)     6/10 = 60.0%         31.3%-83.2%    9/10
Paired: both correct 6, only base correct 3, only fine-tuned correct 0, neither 1 (sign test p = 0.25)
Fine-tuned failures: Q2 wrong rows, Q3 invalid SQL, Q5 wrong columns, Q9 wrong columns
Base failure: Q9 wrong rows (the expected columns of Q9 were ambiguous)
```
