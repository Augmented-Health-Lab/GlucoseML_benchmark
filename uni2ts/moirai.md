# Moirai (Uni2TS) - GlucoseML Benchmark

Only 3 runnable scripts are kept under `uni2ts/`:

- Zero-shot: `uni2ts/predict_glucose_multiwindow_uni2ts_zeroshot.py`
- Full-shot: `uni2ts/predict_glucose_multiwindow_uni2ts_fullshot.py`
- Few-shot: `uni2ts/predict_glucose_multiwindow_uni2ts_fewshot.py` (same as full-shot, except `--train-stride-steps` defaults to 240)

## Data Layout

From the project root (`GlucoseML_benchmark/`), the recommended directory structure is:

```
hf_cache/
  train/
    mixed/
      <dataset>__<subject_id>.csv
  test/
    <dataset_name>/
      <subject_id>.csv
```

- `hf_cache/train/mixed/`: training pool for full-shot / few-shot.
- `hf_cache/test/<dataset_name>/`: per-dataset test folders (the scripts evaluate each dataset separately).

You can also skip CSV export and load from HuggingFace directly by passing `--data-source hf` (requires `datasets`).

CSV column requirements (auto-detected):
- time column: `timestamp` / `time` / `datetime` / `date_time` / `date`
- value column: `bgvalue` / `glucose` / `glucose_value` / `sensor_glucose` / `value`

## Commands

Run from the project root so the default relative paths work.

### Zero-shot (no training; evaluation only)

Evaluate test only:

```
python uni2ts/predict_glucose_multiwindow_uni2ts_zeroshot.py --splits test --data-root-test hf_cache/test
```

Evaluate train + test:

```
python uni2ts/predict_glucose_multiwindow_uni2ts_zeroshot.py --splits train test --data-root-train hf_cache/train --data-root-test hf_cache/test
```

### Full-shot (train on mixed; test on each dataset)

```
python uni2ts/predict_glucose_multiwindow_uni2ts_fullshot.py --data-root-train hf_cache/train/mixed --data-root-test hf_cache/test
```

### Few-shot

```
python uni2ts/predict_glucose_multiwindow_uni2ts_fewshot.py --data-root-train hf_cache/train/mixed --data-root-test hf_cache/test
```

### Defaults (Full-shot / Few-shot)

Full-shot defaults:
- `--context-hours`: `12`
- `--horizons-minutes`: `30`
- `--eval-stride-steps`: `1`
- `--train-epochs`: `10`
- `--train-stride-steps`: `10`

Few-shot defaults:
- same as full-shot, except `--train-stride-steps`: `240`

### Common Optional Args

- `--datasets <name1> <name2> ...`: run only selected datasets (dataset names from the HF split, or folder names under `hf_cache/test/` in CSV mode)
- `--context-hours ...` / `--horizons-minutes ...`: select context windows / prediction horizons
- `--eval-stride-steps N`: evaluation sliding-window stride (`0` means `context_steps`; `1` means 5 minutes)
- full-shot/few-shot training: `--train-epochs` / `--train-batch-size` / `--train-stride-steps` / `--max-train-windows` / `--max-train-steps`

## Outputs

After running, results are written under `uni2ts/`:

- Zero-shot: `uni2ts/multi_horizon_results_uni2ts_zeroshot/`
- Full-shot: `uni2ts/multi_horizon_results_uni2ts_fullshot/`
- Few-shot: `uni2ts/multi_horizon_results_uni2ts_fewshot/`
