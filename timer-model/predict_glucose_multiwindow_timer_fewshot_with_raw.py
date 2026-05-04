from __future__ import annotations

from pathlib import Path

import predict_glucose_multiwindow_timer_fullshot_with_raw as fullshot_r


if __name__ == "__main__":
    fullshot_r.main(
        results_root=Path(__file__).resolve().parent / "multi_horizon_results_timer_fewshot_with_raw",
        log_stem="predict_glucose_multiwindow_timer_fewshot_with_raw",
        shot_label="few-shot",
        default_train_stride_steps=240,
    )
