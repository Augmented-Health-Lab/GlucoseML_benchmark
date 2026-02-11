"""
Prepare the GlucoFM Benchmark dataset from HuggingFace.

This script mirrors `Time-LLM/prepare_dataset.py` but is placed at the repo root
so all models can share the same preparation workflow.

It exports the HuggingFace dataset to per-subject CSV files and optionally
creates a `mixed/` folder that concatenates all sub-datasets.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd


def _require_datasets():  # pragma: no cover
    try:
        from datasets import load_dataset  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'datasets'. Install it first:\n"
            "  pip install datasets\n"
        ) from exc
    return load_dataset


def export_hf_to_csv(hf_name: str, split: str, out_root: str) -> None:
    """Export the HF dataset split into per-subject CSV files."""

    print(f"\n{'=' * 70}")
    print(f"Exporting {split} split from {hf_name}")
    print(f"{'=' * 70}")

    load_dataset = _require_datasets()
    ds = load_dataset(hf_name, split=split)

    subject_count = 0
    for row in ds:
        dataset = row["dataset"]
        subject_id = row["subject_id"]

        out_dir = os.path.join(out_root, split, str(dataset))
        os.makedirs(out_dir, exist_ok=True)

        csv_path = os.path.join(out_dir, f"{subject_id}.csv")

        # Keep raw timestamps (epoch seconds) as provided by HF. Downstream
        # loaders infer and convert to datetime as needed.
        df = pd.DataFrame({"timestamp": row["timestamp"], "BGvalue": row["BGvalue"]})
        df.to_csv(csv_path, index=False)
        subject_count += 1

    print(f"Exported {subject_count} subjects to: {os.path.join(out_root, split)}")

    # Create "all" markers for the training split (some legacy loaders use it).
    if split == "train":
        datasets_found = set(ds["dataset"])
        for dataset in datasets_found:
            marker_path = os.path.join(out_root, "train", str(dataset), "all")
            Path(marker_path).touch()
        print(f"Created 'all' markers for {len(datasets_found)} datasets")


def copy_hf_csvs_to_mixed(root_dir: str = "hf_cache") -> None:
    """Copy all CSV files from subdirectories to a split-level `mixed/` folder.

    Name collisions are avoided by prefixing the filename with the relative
    dataset path (joined using `__`).
    """

    print(f"\n{'=' * 70}")
    print(f"Creating mixed datasets from {root_dir}")
    print(f"{'=' * 70}")

    root = Path(root_dir)

    for split in ["train", "test"]:
        split_dir = root / split
        dst_dir = split_dir / "mixed"
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not split_dir.exists():
            print(f"[{split}] Skip: {split_dir} not found")
            continue

        csv_paths = [p for p in split_dir.rglob("*.csv") if "mixed" not in p.parts]
        print(f"[{split}] Found {len(csv_paths)} CSV files")

        copied = 0
        for src_path in csv_paths:
            rel = src_path.relative_to(split_dir)
            safe_name = "__".join(rel.parts)
            dst_path = dst_dir / safe_name
            shutil.copy2(src_path, dst_path)
            copied += 1

        if split == "train":
            (dst_dir / "all").touch()

        print(f"[{split}] Copied {copied} files to: {dst_dir}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare GlucoFM Benchmark dataset from HuggingFace")
    parser.add_argument(
        "--hf_name",
        "--hf-name",
        type=str,
        default="byluuu/gluco-tsfm-benchmark",
        help="HuggingFace dataset name (default: byluuu/gluco-tsfm-benchmark).",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=str,
        default="./hf_cache",
        help="Output directory for exported CSV files (default: ./hf_cache).",
    )
    parser.add_argument(
        "--create_mixed",
        "--create-mixed",
        action="store_true",
        help="Create split-level `mixed/` folders that combine all subdatasets.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GLUCOFM DATASET PREPARATION")
    print("=" * 70)

    export_hf_to_csv(hf_name=args.hf_name, split="train", out_root=args.output_dir)
    export_hf_to_csv(hf_name=args.hf_name, split="test", out_root=args.output_dir)

    if args.create_mixed:
        copy_hf_csvs_to_mixed(root_dir=args.output_dir)

    print(f"\n{'=' * 70}")
    print("DATASET PREPARATION COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Train data: {args.output_dir}/train/")
    print(f"Test data:  {args.output_dir}/test/")
    if args.create_mixed:
        print(f"Mixed train: {args.output_dir}/train/mixed/")
        print(f"Mixed test:  {args.output_dir}/test/mixed/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

