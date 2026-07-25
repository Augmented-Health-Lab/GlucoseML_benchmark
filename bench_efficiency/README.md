# Runtime efficiency benchmark

Training time and inference time for all 8 methods across the zero-/few-/full-shot
protocols, at context 144 steps (12 h) and horizon 30 min.

Nothing here re-implements a method. Each method's *current* script is imported
and its own functions are wrapped with timers, so the numbers come from the exact
code that produced the paper results.

## The idea

Measuring end-to-end wall clock for every method x protocol would take days.
Instead we measure **throughput** (cheap, seconds) and multiply by **workload**
(counted on CPU, no model loaded):

```
train_time = (train_windows / batch) x epochs x sec_per_step
infer_time = test_windows x sec_per_window
```

Throughput does not depend on the protocol — few-shot and full-shot differ only
in the training stride, i.e. in workload. So **each method is benched once** and
both protocols are derived in `aggregate.py`.

Two systematic errors are removed:

* **Warm-up / setup.** Every training measurement runs twice with a different
  step cap and takes the slope. Model loading, dataset construction and CUDA
  autotune are identical in both runs and cancel out. Inference drops the first
  subject for the same reason.
* **Async CUDA.** Every measurement is bracketed by `torch.cuda.synchronize()`.

## Prerequisites

```bash
# One canonical data cache for all methods (the per-method hf_cache copies are stale)
python prepare_dataset.py --hf_name byluuu/gluco-tsfm-benchmark \
    --output_dir hf_cache --create_mixed

# The LSTM baseline imports tensorflow.keras.optimizers.legacy -> needs Keras 2
pip install tf_keras
```

## Run

```bash
bash bench_efficiency/run_all.sh          # ~40-70 min on one GPU
```

Or one method at a time:

```bash
python bench_efficiency/count_windows.py --data-root hf_cache

python bench_efficiency/bench_lstm.py --mode throughput
python bench_efficiency/bench_lstm.py --mode fewshot-real --train-csv-dir hf_cache/train/mixed

python bench_efficiency/bench_multiwindow.py --method timesfm            # also gpformer/timer/moirai
python bench_efficiency/bench_multiwindow.py --method timesfm --zeroshot

python bench_efficiency/bench_chronos.py --protocol fullshot
python bench_efficiency/bench_informer.py --method timellm --test-windows 4321 9876

python bench_efficiency/aggregate.py      # -> results/efficiency_table.{md,json}
```

## Files

| File | Role |
|---|---|
| `methods.py` | Registry of each method's current script + its real hyperparameters. **Update here when a script changes.** |
| `timing.py` | CUDA-synced phase timers, param counts, peak memory. |
| `count_windows.py` | Workload: forecast-window counts per split/stride, both window conventions. |
| `bench_multiwindow.py` | GPFormer / TimesFM / Timer / Moirai (shared harness). |
| `bench_lstm.py` | LSTM (Keras); `throughput` and `fewshot-real` modes. |
| `bench_chronos.py` | Chronos-2 LoRA fine-tune + rolling forecast. |
| `bench_informer.py` | Time-LLM / CALF, via their own `speed: X s/iter` output. |
| `aggregate.py` | throughput x workload -> final table. |

## Two reported columns

* **normalized** — every method at `epochs=10, batch=16, stride 12/240`.
  Comparable across methods. Use this for the head-to-head claim.
* **as_configured** — each method's own settings (`methods.py:AS_CONFIGURED`).
  What the paper's runs actually cost. Rows are *not* comparable to each other.

Both are needed because the methods are not currently configured alike:

| | epochs | batch | train stride (full/few) | pred_len |
|---|---|---|---|---|
| LSTM | 10000 + EarlyStop(50) | 1024 | 12 / 240 | 6 |
| GPFormer | 10 | 16 | 12 / 240 | 6 |
| TimesFM / Timer / Moirai | 10 | 16 | **10** / 240 | 6 |
| Chronos-2 | 16000 steps (no epochs) | 32 | 12 / 240 | 18 |
| Time-LLM | 40 (cap 30k win/epoch) | 16 | 12 / 240 | 18 |
| CALF | 30 + EarlyStop(8) (cap 30k) | 8 | 12 / 240 | 18 |

## Measured workload (HF train split, 529 subjects, ctx 144 + hor 6)

| stride | use | windows |
|---|---|---|
| 1 | evaluation | 2,143,754 |
| 12 | full-shot training | 178,789 |
| 240 | few-shot training | 9,208 |

The two window conventions agree to within 0.3 % (the LSTM restarts its stride
inside each gap-free segment; the others stride over the whole series).

## Known issues found while building this

* **`2019Martinsson_et_al_LSTM/hf_cache/test/` is a byte-for-byte copy of
  `train/`.** [`fullshot_lstm.py:38`](../2019Martinsson_et_al_LSTM/fullshot_lstm.py#L38)
  hardcodes `split="train"` and ignores its own `split` argument; all 529 files
  match in size. This affects the LSTM's *evaluation results*, not only timing.
  The root [`prepare_dataset.py`](../prepare_dataset.py) does it correctly — use
  that cache.
* **TimesFM / Timer / Moirai default to `--train-stride-steps 10`**, while
  GPFormer, the LSTM and Chronos use 12. `run_all.sh` passes 12 explicitly for
  the normalized column; the discrepancy is recorded in `AS_CONFIGURED`.
* **Chronos has no epoch concept** — it fine-tunes for a fixed `--num_steps
  16000`, so its as-configured training time is a step budget, not
  epochs x dataset size.
