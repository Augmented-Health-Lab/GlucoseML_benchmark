# GlucoFM-Bench - TiRex-2 Evaluation

Run zero-shot TiRex-2 on the HuggingFace CGM benchmark dataset.

Dataset used:
byluuu/gluco-tsfm-benchmark

## Model

| Property | Value |
|---|---|
| Checkpoint | `NX-AI/TiRex-2` (HuggingFace), loaded with `tirex2.load_model` |
| Size | 82.5 M parameters, fp32 (`model.safetensors`, 330 MB) |
| Backbone | xLSTM: 12 blocks alternating bi-mLSTM / bi-sLSTM time mixers, each followed by an attention variate mixer (so one model handles univariate and multivariate inputs) |
| Context | up to 2048 steps (~170 h at 5 min); shorter contexts are left-padded with NaN |
| Output | 9 quantiles, 0.1 to 0.9; up to 320 future steps |
| Scaling | arcsinh squashing after normalization  |
| Test-time augmentation | Differencing off, sign flip on (the checkpoint defaults are differencing on, sign flip off) |

---

## 1. Setup

Requires Python >= 3.11.

Create environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

Install packages:
```bash
pip install -r TiRex-2/requirements.txt
```

This installs `tirex-2==0.4.0` (with `torch`, `numpy`, `flashrnn`, `mlstm_kernels`, `huggingface-hub`, `safetensors`), `datasets`, `pandas` and `matplotlib`.

- On an NVIDIA GPU (`--device cuda`), the sLSTM CUDA kernel is compiled on the first run. This needs a CUDA toolkit with `nvcc`, and the first run is slower.
- On Apple silicon (`--device mps`) or CPU, no compiler is needed.
- The checkpoint `NX-AI/TiRex-2` is downloaded from HuggingFace on the first run.

## 2. Zero-shot

Run from the repository root. First export the dataset once to the local CSV cache `hf_cache/` (see "Prepare Dataset" in the main README):
```bash
python prepare_dataset.py --hf-name byluuu/gluco-tsfm-benchmark --output-dir ./hf_cache
```

Then run the evaluation, which reads `hf_cache/test/`:
```bash
python TiRex-2/predict_glucose_multiwindow_tirex-2_zeroshot.py
```

To load the dataset directly from HuggingFace instead of the local cache, add `--data-source hf`.

To recompute only the metrics from saved raw predictions (no inference):
```bash
python TiRex-2/predict_glucose_multiwindow_tirex-2_zeroshot.py --data-source hf --metrics-only
```

## 3. Default attributes

| Argument | Default | Meaning |
|---|---|---|
| `--data-source` | `csv` | `csv` reads `hf_cache/test/`; `hf` loads the dataset from HuggingFace |
| `--hf-name` | `byluuu/gluco-tsfm-benchmark` | HuggingFace dataset (with `--data-source hf`) |
| `--data-root-test` | `hf_cache/test` | CSV folder (with `--data-source csv`) |
| `--datasets` | all except `mixed` | Datasets to run, e.g. `--datasets AZT1D D1NAMO` |
| `--context-hours` | `1 4 8 12 16 24` | Context windows, in hours |
| `--horizons-minutes` | `15 30 60 90` | Forecast horizons, in minutes |
| `--device` | `mps`, else `cuda`, else `cpu` | Device for TiRex-2 |
| `--batch-size` | `256` | Windows per forecast call |
| `--workers` | `4` | Datasets scored in parallel in the metrics stage |
| `--overwrite` | off | Recompute forecasts instead of reusing cached rows. Each dataset's files are rewritten with only this run's contexts and horizons, so use it with the full grid |
| `--metrics-only` | off | Skip inference; score the saved raw predictions |

Fixed settings (not exposed as arguments):
- test split, sliding windows with stride 1 step (5 min);
- each window is scored at the last step of the horizon (`metric_mode final`);
- point forecast = median quantile (0.5); prediction intervals PI80 (q0.1–q0.9) and PI60 (q0.2–q0.8);
- test-time differencing off, test-time sign flip on.

## 4. Output

Saved to:
```css
TiRex-2/multi_horizon_results_tirex2_zeroshot/
  <dataset_name>_test_metrics.csv
  raw_predictions/
    <dataset_name>_test_raw_predictions.csv
  quantile_predictions/
    <dataset_name>_test_quantile_predictions.csv
  full_metrics/
    tirex2_zeroshot_per_participant.csv
    tirex2_zeroshot_summary.csv
    tirex2_zeroshot_overall.csv
  predict_glucose_multiwindow_tirex2_zeroshot_<timestamp>.log
```

- `quantile_predictions`: the same scored points as `raw_predictions`, with one extra column for each of the 9 quantiles of the model (`q10`, `q20`, ..., `q90`). The column `q50` equals `prediction`.
- `per_participant`: all metrics for each participant, context and horizon.
- `summary`: mean and std over participants, for each dataset, "Overall (Open)" and "Overall (Total)".
- `overall`: compact "Overall (Total)" table, `mean (std)` for each context and horizon.
