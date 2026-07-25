"""Registry of the *current* entry point for every method x protocol.

Verified against `git log` + file mtimes on 2026-07-25.  If a newer script
lands, change it here and nowhere else -- every bench runner reads this table.

`train_fn` / `eval_fn` are patched by name at run time, so the registry never
has to know a function's signature.  `base_module` is the sibling module that
the `_with_raw*` wrappers import as `fullshot`; the training loop lives there,
so it has to be patched too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class MultiWindowMethod:
    """A method built on the shared `predict_glucose_multiwindow_*` harness."""

    name: str
    script: Path
    base_module: Optional[str] = None
    train_fns: tuple = ("train_fullshot", "train_finetune")
    eval_fns: tuple = (
        "evaluate_subject",
        "evaluate_subject_with_raw",
        "evaluate_subject_with_raw_quantile",
    )
    smallest_dataset: str = "D1NAMO"
    extra_args: List[str] = field(default_factory=list)


MULTIWINDOW: Dict[str, MultiWindowMethod] = {
    "gpformer": MultiWindowMethod(
        name="GPFormer",
        script=REPO_ROOT / "GPFormer" / "gpformer_fullshot_raw.py",
    ),
    "timesfm": MultiWindowMethod(
        name="TimesFM",
        script=REPO_ROOT
        / "timesfm"
        / "predict_glucose_multiwindow_timesfm_fullshot_with_raw_quantile.py",
        base_module="predict_glucose_multiwindow_timesfm_fullshot",
    ),
    "timer": MultiWindowMethod(
        name="Timer",
        script=REPO_ROOT
        / "timer-model"
        / "predict_glucose_multiwindow_timer_fullshot_with_raw.py",
        base_module="predict_glucose_multiwindow_timer_fullshot",
    ),
    "moirai": MultiWindowMethod(
        name="Moirai",
        script=REPO_ROOT
        / "uni2ts"
        / "predict_glucose_multiwindow_uni2ts_fullshot_with_raw_quantile.py",
        base_module="predict_glucose_multiwindow_uni2ts_fullshot",
    ),
}

# Zero-shot inference only (no training loop to time).
MULTIWINDOW_ZEROSHOT: Dict[str, MultiWindowMethod] = {
    "timesfm": MultiWindowMethod(
        name="TimesFM",
        script=REPO_ROOT
        / "timesfm"
        / "predict_glucose_multiwindow_timesfm_zeroshot_with_raw_quantile.py",
        base_module="predict_glucose_multiwindow_timesfm_zeroshot",
    ),
    "timer": MultiWindowMethod(
        name="Timer",
        script=REPO_ROOT
        / "timer-model"
        / "predict_glucose_multiwindow_timer_zeroshot_with_raw.py",
        base_module="predict_glucose_multiwindow_timer_zeroshot",
    ),
    "moirai": MultiWindowMethod(
        name="Moirai",
        script=REPO_ROOT
        / "uni2ts"
        / "predict_glucose_multiwindow_uni2ts_zeroshot_with_raw_quantile.py",
        base_module="predict_glucose_multiwindow_uni2ts_zeroshot",
    ),
}


# ---------------------------------------------------------------------------
# Protocol configuration actually used by each method in the paper.
# `aggregate.py` multiplies measured throughput by these to get the
# "as-configured" column; the "normalized" column uses NORMALIZED instead.
# ---------------------------------------------------------------------------
NORMALIZED = {
    "train_epochs": 10,
    "train_batch_size": 16,
    "train_stride_fullshot": 12,
    "train_stride_fewshot": 240,
    "eval_stride": 1,
}

AS_CONFIGURED = {
    # method: epochs, batch, full/few train stride, notes
    "lstm": {
        "train_epochs": None,  # early stopping: measured, see bench_lstm.py
        "train_batch_size": 1024,
        "train_stride_fullshot": 12,
        "train_stride_fewshot": 240,
        "window_convention": "lstm",
        "note": "epochs=10000 with EarlyStopping(patience=50); real epoch count measured",
    },
    "gpformer": {
        "train_epochs": 10,
        "train_batch_size": 16,
        "train_stride_fullshot": 12,
        "train_stride_fewshot": 240,
        "window_convention": "multiwindow",
    },
    "timesfm": {
        "train_epochs": 10,
        "train_batch_size": 16,
        "train_stride_fullshot": 10,
        "train_stride_fewshot": 240,
        "window_convention": "multiwindow",
        "note": "DEFAULT_TRAIN_STRIDE_STEPS=10, not 12",
    },
    "timer": {
        "train_epochs": 10,
        "train_batch_size": 16,
        "train_stride_fullshot": 10,
        "train_stride_fewshot": 240,
        "window_convention": "multiwindow",
        "note": "DEFAULT_TRAIN_STRIDE_STEPS=10, not 12",
    },
    "moirai": {
        "train_epochs": 10,
        "train_batch_size": 16,
        "train_stride_fullshot": 10,
        "train_stride_fewshot": 240,
        "window_convention": "multiwindow",
        "note": "DEFAULT_TRAIN_STRIDE_STEPS=10, not 12",
    },
    "chronos": {
        "train_steps": 16000,  # fixed step budget, not epoch based
        "train_batch_size": 32,
        "train_stride_fullshot": 12,
        "train_stride_fewshot": 240,
        "window_convention": "multiwindow",
        "note": "LoRA fine-tune with a fixed --num_steps budget; epochs undefined",
    },
    "timellm": {
        "train_epochs": 40,
        "train_batch_size": 16,
        "train_stride_fullshot": 12,
        "train_stride_fewshot": 240,
        "max_windows_per_epoch": 30000,
        "window_convention": "multiwindow",
    },
    "calf": {
        "train_epochs": 30,
        "train_batch_size": 8,
        "train_stride_fullshot": 12,
        "train_stride_fewshot": 240,
        "max_windows_per_epoch": 30000,
        "window_convention": "multiwindow",
        "note": "EarlyStopping(patience=8) may cut this short",
    },
}
