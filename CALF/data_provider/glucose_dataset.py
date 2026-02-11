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
        scale_value=100.0,
        max_gap_minutes=60,  # Maximum allowed time gap (default: 1 hour)
    ):
        assert split in ["train", "val", "test"]

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.split = split
        self.scale_value = scale_value
        self.max_gap_minutes = max_gap_minutes

        self.subject_series = {}   # subject_id -> list of continuous segments
        self.windows = []

        if min_len is None:
            min_len = seq_len + pred_len

        subject_files = sorted(os.listdir(data_dir))

        for fname in subject_files:
            subject_id = fname.replace(".csv", "")
            df = pd.read_csv(os.path.join(data_dir, fname))

            timestamps = df["timestamp"].values
            values = df["BGvalue"].values.astype(np.float32)

            if len(values) < min_len:
                continue

            # ---------- detect gaps and split into continuous sequences ----------
            continuous_sequences = self._split_by_gaps(timestamps, values)

            # ---------- chronological split ----------
            all_split_sequences = []
            for seq_timestamps, seq_values in continuous_sequences:
                n = len(seq_values)
                train_end = int(n * split_ratio[0])
                val_end = int(n * (split_ratio[0] + split_ratio[1]))

                if split == "train":
                    segment_values = seq_values[:train_end]
                elif split == "val":
                    segment_values = seq_values[train_end:val_end]
                else:  # test - use all data from continuous sequences
                    segment_values = seq_values

                if len(segment_values) >= min_len:
                    # ---------- scaling (NOT z-score) ----------
                    segment_values = segment_values / self.scale_value
                    all_split_sequences.append(segment_values)

            if not all_split_sequences:
                continue

            # Store sequences for this subject
            self.subject_series[subject_id] = all_split_sequences

            # Create windows within each continuous sequence
            for seq_idx, segment in enumerate(all_split_sequences):
                max_start = len(segment) - seq_len - pred_len
                for start in range(0, max_start + 1, stride):
                    self.windows.append((subject_id, seq_idx, start))

        # subject -> window indices
        self.subject_to_indices = {}
        for idx, (sid, _, _) in enumerate(self.windows):
            self.subject_to_indices.setdefault(sid, []).append(idx)

    def _split_by_gaps(self, timestamps, values):
        """
        Split data into continuous sequences based on time gaps.
        
        Args:
            timestamps: Array of Unix timestamps
            values: Array of glucose values
            
        Returns:
            List of (timestamp_sequence, value_sequence) tuples
        """
        if len(timestamps) == 0:
            return []

        sequences = []
        current_timestamps = [timestamps[0]]
        current_values = [values[0]]

        max_gap_seconds = self.max_gap_minutes * 60

        for i in range(1, len(timestamps)):
            time_diff = timestamps[i] - timestamps[i-1]
            
            # If gap is too large, start a new sequence
            if time_diff > max_gap_seconds:
                # Save current sequence if it's long enough
                if len(current_values) > 0:
                    sequences.append((
                        np.array(current_timestamps),
                        np.array(current_values, dtype=np.float32)
                    ))
                # Start new sequence
                current_timestamps = [timestamps[i]]
                current_values = [values[i]]
            else:
                # Continue current sequence
                current_timestamps.append(timestamps[i])
                current_values.append(values[i])

        # Don't forget the last sequence
        if len(current_values) > 0:
            sequences.append((
                np.array(current_timestamps),
                np.array(current_values, dtype=np.float32)
            ))

        return sequences

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        subject_id, seq_idx, start = self.windows[idx]
        series = self.subject_series[subject_id][seq_idx]

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
