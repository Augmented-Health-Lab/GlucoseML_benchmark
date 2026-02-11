# Timer - GlucoseML Benchmark

Only 3 runnable scripts are kept under `timer-model/`:

- Zero-shot: `timer-model/predict_glucose_multiwindow_timer_zeroshot.py`
- Full-shot: `timer-model/predict_glucose_multiwindow_timer_fullshot.py`
- Few-shot: `timer-model/predict_glucose_multiwindow_timer_fewshot.py` (same as full-shot, except `--train-stride-steps` defaults to 240)

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
python timer-model/predict_glucose_multiwindow_timer_zeroshot.py --splits test --data-root-test hf_cache/test
```

### Full-shot (train on mixed; test on each dataset)

```
python timer-model/predict_glucose_multiwindow_timer_fullshot.py --data-root-train hf_cache/train/mixed --data-root-test hf_cache/test
```

### Few-shot

```
python timer-model/predict_glucose_multiwindow_timer_fewshot.py --data-root-train hf_cache/train/mixed --data-root-test hf_cache/test
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

## Outputs

After running, results are written under `timer-model/`:

- Zero-shot: `timer-model/multi_horizon_results_timer_zeroshot/`
- Full-shot: `timer-model/multi_horizon_results_timer_fullshot/`
- Few-shot: `timer-model/multi_horizon_results_timer_fewshot/`

