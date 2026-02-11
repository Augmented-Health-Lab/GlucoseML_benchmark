# GPFormer - GlucoseML Benchmark

Runnable scripts under `GPFormer/`:

- Full-shot: `GPFormer/predict_glucose_multiwindow_gpformer_fullshot.py`
- Few-shot: `GPFormer/predict_glucose_multiwindow_gpformer_fewshot.py`

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
- `hf_cache/test/<dataset_name>/`: per-dataset test folders.

You can also skip CSV export and load from HuggingFace directly by passing `--data-source hf` (requires `datasets`).

CSV column requirements (auto-detected):
- time column: `timestamp` / `time` / `datetime` / `date_time` / `date`
- value column: `bgvalue` / `glucose` / `glucose_value` / `sensor_glucose` / `value`

## Commands

Run from the project root so the default relative paths work.

### Full-shot (train on mixed; test on each dataset)

```
python GPFormer/predict_glucose_multiwindow_gpformer_fullshot.py --data-root-train hf_cache/train/mixed --data-root-test hf_cache/test
```

### Few-shot (same training pool; fewer training windows via larger stride)

```
python GPFormer/predict_glucose_multiwindow_gpformer_fewshot.py --data-root-train hf_cache/train/mixed --data-root-test hf_cache/test
```

### Defaults

- Shared:
  - `--context-hours`: `12`
  - `--horizons-minutes`: `30`
  - `--eval-stride-steps`: `1`
  - `--train-epochs`: `10`
- Full-shot:
  - `--train-stride-steps`: `12`
- Few-shot:
  - `--train-stride-steps`: `240`

### Common Optional Args

- `--datasets <name1> <name2> ...`: run only selected datasets (dataset names from the HF split, or folder names under `hf_cache/test/` in CSV mode)
- `--context-hours ...` / `--horizons-minutes ...`: select context windows / prediction horizons
- `--eval-stride-steps N`: evaluation sliding-window stride (`0` means `context_steps`; `1` means 5 minutes)
- training: `--train-epochs` / `--train-batch-size` / `--train-stride-steps` / `--max-train-windows` / `--max-train-steps`

## Outputs

After running, results are written under `GPFormer/`:

- Full-shot: `GPFormer/multi_horizon_results_gpformer_fullshot/`
- Few-shot: `GPFormer/multi_horizon_results_gpformer_fewshot/`

Trained checkpoints are saved under:
- `GPFormer/multi_horizon_results_gpformer_fullshot/saved_models/`
- `GPFormer/multi_horizon_results_gpformer_fewshot/saved_models/`
