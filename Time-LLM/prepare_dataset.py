"""
Dataset preparation script for Time-LLM glucose forecasting.
Exports HuggingFace dataset to CSV format and organizes into train/test splits.
"""

import os
import shutil
from pathlib import Path
import pandas as pd
from datasets import load_dataset


def export_hf_to_csv(hf_name, split, out_root):
    """
    Export HF dataset to per-subject CSV files compatible with Time-LLM.
    
    Args:
        hf_name: HuggingFace dataset name
        split: 'train' or 'test'
        out_root: Output root directory
    """
    print(f"\n{'='*70}")
    print(f"Exporting {split} split from {hf_name}")
    print(f"{'='*70}")
    
    ds = load_dataset(hf_name, split=split)
    
    subject_count = 0
    for row in ds:
        dataset = row["dataset"]
        subject_id = row["subject_id"]
        
        out_dir = os.path.join(out_root, split, dataset)
        os.makedirs(out_dir, exist_ok=True)
        
        csv_path = os.path.join(out_dir, f"{subject_id}.csv")
        
        df = pd.DataFrame({
            "timestamp": row["timestamp"],
            "BGvalue": row["BGvalue"],
        })
        
        df.to_csv(csv_path, index=False)
        subject_count += 1
    
    print(f"Exported {subject_count} subjects to: {os.path.join(out_root, split)}")
    
    # Create "all" marker for training split
    if split == "train":
        datasets_found = set(ds["dataset"])
        for dataset in datasets_found:
            marker_path = os.path.join(out_root, "train", dataset, "all")
            Path(marker_path).touch()
        print(f"Created 'all' markers for {len(datasets_found)} datasets")


def copy_hf_csvs_to_mixed(root_dir="hf_cache"):
    """
    Copy all CSV files from subdirectories to a 'mixed' folder.
    Handles name collisions by prefixing with relative path.
    
    Args:
        root_dir: Root directory containing train/test splits
    """
    print(f"\n{'='*70}")
    print(f"Creating mixed datasets from {root_dir}")
    print(f"{'='*70}")
    
    root = Path(root_dir)
    
    for split in ["train", "test"]:
        split_dir = root / split
        dst_dir = split_dir / "mixed"
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        if not split_dir.exists():
            print(f"[{split}] Skip: {split_dir} not found")
            continue
        
        # Recursively find CSVs (excluding mixed folder)
        csv_paths = [p for p in split_dir.rglob("*.csv") if "mixed" not in p.parts]
        
        print(f"[{split}] Found {len(csv_paths)} CSV files")
        
        copied = 0
        for src_path in csv_paths:
            # Build collision-safe filename
            rel = src_path.relative_to(split_dir)
            safe_name = "__".join(rel.parts)
            dst_path = dst_dir / safe_name
            
            shutil.copy2(src_path, dst_path)
            copied += 1
        
        # Create "all" marker for training
        if split == "train":
            (dst_dir / "all").touch()
        
        print(f"[{split}] Copied {copied} files to: {dst_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Time-LLM glucose dataset")
    parser.add_argument("--hf_name", type=str, default="byluuu/gluco-tsfm-benchmark",
                        help="HuggingFace dataset name")
    parser.add_argument("--output_dir", type=str, default="./hf_cache",
                        help="Output directory for CSV files")
    parser.add_argument("--create_mixed", action="store_true",
                        help="Create mixed dataset combining all subdatasets")
    
    args = parser.parse_args()
    
    print("="*70)
    print("TIME-LLM DATASET PREPARATION")
    print("="*70)
    
    # Export train and test splits
    export_hf_to_csv(
        hf_name=args.hf_name,
        split="train",
        out_root=args.output_dir
    )
    
    export_hf_to_csv(
        hf_name=args.hf_name,
        split="test",
        out_root=args.output_dir
    )
    
    # Optionally create mixed datasets
    if args.create_mixed:
        copy_hf_csvs_to_mixed(root_dir=args.output_dir)
    
    print(f"\n{'='*70}")
    print("DATASET PREPARATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Train data: {args.output_dir}/train/")
    print(f"Test data: {args.output_dir}/test/")
    if args.create_mixed:
        print(f"Mixed train: {args.output_dir}/train/mixed/")
        print(f"Mixed test: {args.output_dir}/test/mixed/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
