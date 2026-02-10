import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class MultiSubjectGlucoseDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        seq_len=24,
        pred_len=6,
        stride=1,
        split_ratio=(0.875, 0.125, 0.0),
        min_len=None,
        scale_value=100.0,   # <<< NEW
    ):
        assert split in ["train", "val", "test"]

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.split = split
        self.scale_value = scale_value

        self.subject_series = {}   # subject_id -> np.array (scaled)
        self.windows = []

        if min_len is None:
            min_len = seq_len + pred_len

        subject_files = sorted(os.listdir(data_dir))

        for fname in subject_files:
            subject_id = fname.replace(".csv", "")
            df = pd.read_csv(os.path.join(data_dir, fname))

            values = df["BGvalue"].values.astype(np.float32)

            if len(values) < min_len:
                continue

            # ---------- scaling (NOT z-score) ----------
            values = values / self.scale_value

            # ---------- chronological split ----------
            n = len(values)
            train_end = int(n * split_ratio[0])
            val_end = int(n * (split_ratio[0] + split_ratio[1]))

            if split == "train":
                segment = values[:train_end]
            elif split == "val":
                segment = values[train_end:val_end]
            else:
                segment = values

            if len(segment) < min_len:
                continue

            self.subject_series[subject_id] = segment

            max_start = len(segment) - seq_len - pred_len
            for start in range(0, max_start + 1, stride):
                self.windows.append((subject_id, start))

        # subject -> window indices
        self.subject_to_indices = {}
        for idx, (sid, _) in enumerate(self.windows):
            self.subject_to_indices.setdefault(sid, []).append(idx)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        subject_id, start = self.windows[idx]
        series = self.subject_series[subject_id]

        x = series[start : start + self.seq_len]
        y = series[start + self.seq_len : start + self.seq_len + self.pred_len]

        # -------- dummy time features (for Time-LLM compatibility) --------
        # encoder time mark
        x_mark = np.zeros((self.seq_len, 1), dtype=np.float32)
        # decoder time mark (only pred_len is enough for Time-LLM)
        y_mark = np.zeros((self.pred_len, 1), dtype=np.float32)

        return (
            torch.tensor(x, dtype=torch.float32).unsqueeze(-1),      # [seq_len, 1]
            torch.tensor(y, dtype=torch.float32).unsqueeze(-1),      # [pred_len, 1]
            torch.tensor(x_mark, dtype=torch.float32),               # [seq_len, 1]
            torch.tensor(y_mark, dtype=torch.float32),               # [pred_len, 1]
            subject_id
        )
