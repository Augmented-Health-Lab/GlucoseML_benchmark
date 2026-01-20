from __future__ import annotations

from pathlib import Path

import predict_glucose_multiwindow_timesfm_fullshot as fullshot


if __name__ == "__main__":
    fullshot.main(
        results_root=Path(__file__).resolve().parent / "multi_horizon_results_timesfm_fewshot",
        log_stem="predict_glucose_multiwindow_timesfm_fewshot",
        shot_label="few-shot",
        default_train_stride_steps=240,
    )

