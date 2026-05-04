from __future__ import annotations

from pathlib import Path

import predict_glucose_multiwindow_timesfm_fullshot_with_raw_quantile as fullshot_rq


if __name__ == "__main__":
    fullshot_rq.main(
        results_root=Path(__file__).resolve().parent / "multi_horizon_results_timesfm_fewshot_with_raw_quantile",
        log_stem="predict_glucose_multiwindow_timesfm_fewshot_with_raw_quantile",
        shot_label="few-shot",
        default_train_stride_steps=240,
    )
