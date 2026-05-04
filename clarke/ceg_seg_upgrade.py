"""
clarke/ceg_seg_upgrade.py

Upgraded SEG implementation that uses the **Klonoff et al. 2014
clinician-consensus risk table** (600x600 absolute-risk lookup) instead
of the ISO-15197 envelope approximation in clarke/ceg_seg.py.

Reference
---------
Klonoff DC, Lias C, Vigersky R, Clarke W, Parkes JL, Sacks DB, Kirkman MS,
Kovatchev B, Error Grid Panel.
"The Surveillance Error Grid." J Diabetes Sci Technol. 2014;8(4):658-672.
DOI: 10.1177/1932296814539589

Per Appendix A of that paper, the SEG risk surface is empirical: it is the
(monotonicity-cleaned) average of 206 clinicians' risk ratings. There is no
closed-form. We pull the table from the methcomp package, which ships a
600x600 lookup faithful to the paper:

    https://github.com/wptmdoorn/methcomp/blob/master/methcomp/static/seg.csv

Sanity-check of that file (validated):
  - shape = (600, 600), diagonal = 0 exactly
  - asymmetric (e.g. table[60, 200] = 2.80 vs table[200, 60] = 2.14)
  - non-negative; the file stores |risk score|, range = [0, 3.70]

Setup (one-time)
----------------
Easiest:
    pip install methcomp

That ships ``seg.csv`` inside ``methcomp/static/`` and this module finds
it automatically. As fallbacks the loader will also look for one of:
    clarke/seg_risk_table.npy
    clarke/seg_risk_table.csv
    clarke/seg_risk_table.xlsx

To freeze a copy locally (e.g. for environments where methcomp isn't
installed), run once:
    python -m clarke.ceg_seg_upgrade --freeze

Loader convention
-----------------
- Table is square, indexed [reference_mg_dl, measured_mg_dl].
- Values are absolute risk scores (>= 0); ``methcomp/seg.csv`` already
  stores |risk|. If you supply a signed table, the loader auto-detects
  and takes abs.
- Diagonal must be 0 (sanity-checked on load).
- "Zone A / no clinical risk" = |risk| <= 0.5 (paper Table 5 / 11).

Usage
-----
    from clarke.ceg_seg_upgrade import seg_scores, plot_seg

    mean_abs_risk, zone_a_pct = seg_scores(ref, pred)
    plot_seg(ref, pred)

CEG utilities (Clarke Error Grid) are re-exported unchanged from
clarke.ceg_seg, so this module is a drop-in replacement.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Re-export the CEG-side helpers untouched -- only SEG was wrong.
from clarke.ceg_seg import (
    classify_ceg_zone,
    plot_ceg,
    ceg_zone_pcts,
    load_dataset_participants,
    patient_to_dataset,
)


# ─── SEG lookup table loading ─────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent

LOCAL_TABLE_CANDIDATES = [
    _HERE / "seg_risk_table.npy",
    _HERE / "seg_risk_table.csv",
    _HERE / "seg_risk_table.xlsx",
]


def _validate_seg_table(table: np.ndarray, source: str) -> None:
    if table.ndim != 2 or table.shape[0] != table.shape[1]:
        raise ValueError(
            f"SEG table must be square; got shape {table.shape} from {source}"
        )
    n = table.shape[0]
    if n not in (581, 600, 601):
        print(f"[ceg_seg_upgrade] warning: SEG table is {n}x{n}; "
              f"paper distributes a 581x581 grid; methcomp ships 600x600 "
              f"(source={source})")
    diag = np.diag(table)
    diag_max = float(np.max(np.abs(diag)))
    if diag_max > 1e-3:
        raise ValueError(
            f"SEG table diagonal is not all-zero (max |diag| = {diag_max:.4f}). "
            f"The grid in {source} appears corrupted or has a header row/column "
            f"that wasn't stripped."
        )
    rng = (float(np.nanmin(table)), float(np.nanmax(table)))
    if rng[0] < -4.5 or rng[1] > 4.5:
        print(f"[ceg_seg_upgrade] warning: values outside expected [-4, 4]: "
              f"{rng} (source={source})")


def _load_from_methcomp() -> tuple[np.ndarray, str] | None:
    """Try to load methcomp's bundled seg.csv."""
    try:
        import importlib.resources as ir
        from methcomp import static  # type: ignore
    except Exception:
        return None
    try:
        with ir.files(static).joinpath("seg.csv").open("r") as fh:
            table = np.loadtxt(fh).astype(np.float32)
        return table, "methcomp/static/seg.csv"
    except Exception:
        try:
            import pkg_resources  # type: ignore
            with pkg_resources.open_text(static, "seg.csv") as fh:
                table = np.loadtxt(fh).astype(np.float32)
            return table, "methcomp/static/seg.csv (pkg_resources)"
        except Exception:
            return None


def load_seg_table(path: str | Path | None = None) -> np.ndarray:
    """Load the SEG risk table.

    Resolution order:
      1. Explicit ``path`` if given.
      2. Local file in clarke/ matching LOCAL_TABLE_CANDIDATES.
      3. methcomp's bundled ``seg.csv`` (pip install methcomp).

    Always returns absolute risk values (>= 0). If a signed table is
    supplied, abs is taken automatically.
    """
    table = None
    source = ""

    if path is not None:
        path = Path(path)
        suf = path.suffix.lower()
        if suf == ".npy":
            table = np.load(path).astype(np.float32)
        elif suf in {".csv", ".tsv"}:
            # methcomp's seg.csv is whitespace-separated; np.loadtxt handles
            # both whitespace and any custom delimiter robustly.
            try:
                table = np.loadtxt(path, dtype=np.float32)
            except ValueError:
                sep = "\t" if suf == ".tsv" else ","
                table = pd.read_csv(path, header=None, sep=sep).to_numpy(dtype=np.float32)
        elif suf in {".xlsx", ".xls"}:
            table = pd.read_excel(path, header=None).to_numpy(dtype=np.float32)
        else:
            raise ValueError(f"Unsupported SEG table extension: {suf} ({path})")
        source = str(path)

    if table is None:
        for candidate in LOCAL_TABLE_CANDIDATES:
            if candidate.exists():
                return load_seg_table(candidate)

    if table is None:
        loaded = _load_from_methcomp()
        if loaded is not None:
            table, source = loaded

    if table is None:
        raise FileNotFoundError(
            "No SEG risk table available. Either:\n"
            "  pip install methcomp\n"
            "  (ships a 600x600 lookup compatible with this loader), or\n"
            "place a table at one of:\n  - "
            + "\n  - ".join(str(p) for p in LOCAL_TABLE_CANDIDATES)
        )

    if (table < 0).any():
        print(f"[ceg_seg_upgrade] note: signed table loaded from {source}; "
              f"taking |risk| for zone classification.")
        table = np.abs(table)

    _validate_seg_table(table, source=source)
    return table


SEG_TABLE: np.ndarray = load_seg_table()
_N = SEG_TABLE.shape[0]


# ─── SEG scoring (paper-faithful) ─────────────────────────────────────────────
def seg_risk_scores(ref, pred) -> np.ndarray:
    """SEG |risk| score per (reference, predicted) pair, in [0, ~4].

    The bundled methcomp table stores absolute risk; this function returns
    those values directly (no sign).
    """
    ref  = np.asarray(ref,  dtype=float)
    pred = np.asarray(pred, dtype=float)
    ref_idx  = np.clip(np.rint(ref ).astype(int), 0, _N - 1)
    pred_idx = np.clip(np.rint(pred).astype(int), 0, _N - 1)
    return SEG_TABLE[ref_idx, pred_idx].astype(float)


def seg_scores(ref, pred) -> tuple[float, float]:
    """Return (mean |risk|, zone_a_pct) per Klonoff 2014.

    Zone A = "no clinical risk" = |risk| <= 0.5 (Table 5 / Table 11).
    """
    risk = seg_risk_scores(ref, pred)
    return float(risk.mean()), float((risk <= 0.5).mean() * 100)


def seg_zone_a_pct(ref, pred) -> float:
    """Convenience: % of points with |risk| <= 0.5 (no clinical risk)."""
    risk = seg_risk_scores(ref, pred)
    return float((risk <= 0.5).mean() * 100)


_ZONE_BINS   = [0.0, 0.5, 1.5, 2.5, 3.5, 4.0 + 1e-6]
_ZONE_LABELS = ["None", "Slight", "Moderate", "Great", "Extreme"]
_ZONE_COLORS = ["#00b050", "#92d050", "#ffff00", "#ff9900", "#ff0000"]


def seg_zone_distribution(ref, pred) -> dict[str, float]:
    """Return % of points in each of the 5 SEG risk zones (Table 11)."""
    risk = seg_risk_scores(ref, pred)
    counts, _ = np.histogram(risk, bins=_ZONE_BINS)
    n = max(int(risk.size), 1)
    return {label: float(c / n * 100) for label, c in zip(_ZONE_LABELS, counts)}


# ─── Plot ─────────────────────────────────────────────────────────────────────
def plot_seg(ref_values, pred_values, title="Surveillance Error Grid",
             ax=None, s=2, alpha=0.3, view_max: int = 400):
    """Plot SEG using the official lookup; return (mean |risk|, zone_a_pct)."""
    ref_values  = np.asarray(ref_values,  dtype=float)
    pred_values = np.asarray(pred_values, dtype=float)

    risk_abs  = seg_risk_scores(ref_values, pred_values)
    mean_risk = float(risk_abs.mean())
    zone_a    = float((risk_abs <= 0.5).mean() * 100)
    color_idx = np.clip(np.digitize(risk_abs, _ZONE_BINS[1:]), 0, 4)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    n = min(view_max + 1, _N)
    bg_abs  = SEG_TABLE[:n, :n]
    bg_zone = np.clip(np.digitize(bg_abs, _ZONE_BINS[1:]), 0, 4)
    seg_cmap = ListedColormap(_ZONE_COLORS)
    ax.pcolormesh(np.arange(n), np.arange(n), bg_zone.T,
                  cmap=seg_cmap, vmin=0, vmax=4, alpha=0.25, zorder=0)

    for i, (label, color) in enumerate(zip(_ZONE_LABELS, _ZONE_COLORS)):
        mask = color_idx == i
        if mask.any():
            ax.scatter(ref_values[mask], pred_values[mask],
                       c=color, s=s, alpha=alpha, rasterized=True,
                       label=f"{label} ({mask.sum() / len(mask) * 100:.1f}%)")

    ax.plot([0, view_max], [0, view_max], "k--", lw=1.0, zorder=3)
    ax.set_xlim(0, view_max); ax.set_ylim(0, view_max)
    ax.set_xlabel("Reference Glucose (mg/dL)", fontsize=12)
    ax.set_ylabel("Predicted Glucose (mg/dL)",  fontsize=12)
    ax.set_title(f"{title}\nMean |Risk| = {mean_risk:.4f}  |  "
                 f"Zone A (no risk) = {zone_a:.1f}%", fontsize=13)
    ax.legend(loc="upper left", fontsize=8, markerscale=4)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.2)
    return mean_risk, zone_a


# ─── Method-level summary (parallel to clarke.ceg_seg.summarize_method) ───────
def summarize_method(results: Mapping, default_dataset: str | None = None) -> pd.DataFrame:
    """Tidy a per-method results dict into a DataFrame.

    Each value should now contain ``seg_mean_abs_risk`` (computed via
    ``seg_scores``) plus the existing CEG zone percentages. For backward
    compatibility, keys ``seg_mean_risk`` and ``seg_zone_a_pct`` from the old
    module are also accepted.
    """
    rows = []
    for key, r in results.items():
        if isinstance(key, tuple):
            ds_name, horizon = key
        elif isinstance(key, (int, np.integer)):
            ds_name, horizon = (default_dataset or "Method"), f"{int(key)}min"
        else:
            ds_name, horizon = key, "30min"
        row = {"Dataset": ds_name, "Horizon": str(horizon)}
        for z in "ABCDE":
            row[f"CEG Zone {z} (%)"] = round(r["ceg"][z], 2)
        mean_risk = r.get("seg_mean_abs_risk", r.get("seg_mean_risk"))
        row["SEG Mean |Risk|"]  = round(float(mean_risk), 4)
        row["SEG Zone A (%)"]   = round(float(r["seg_zone_a_pct"]), 2)
        rows.append(row)
    return pd.DataFrame(rows).set_index(["Dataset", "Horizon"])


# ─── Per-test-dataset SEG-aware Zone A summary ────────────────────────────────
def _default_participants_path() -> Path:
    return _HERE.parent / "test_dataset" / "dataset_participants.json"


def summarize_zone_a_per_dataset(
    df: pd.DataFrame,
    *,
    patient_col: str | None = None,
    dataset_col: str | None = None,
    pred_col: str = "Prediction",
    truth_col: str = "GroundTruth",
    horizon_col: str | None = "Horizon",
    participants_path: str | Path | None = None,
    method_name: str = "Method",
    warn_unmapped: bool = True,
) -> pd.DataFrame:
    """Per (test_dataset, horizon) CEG/SEG Zone A summary using the official table."""
    if dataset_col is None and patient_col is None:
        raise ValueError("Provide either dataset_col or patient_col.")

    needed = [c for c in (patient_col, dataset_col, pred_col,
                          truth_col, horizon_col) if c is not None]
    work = df.loc[:, needed].copy()

    if dataset_col is None:
        path = participants_path or _default_participants_path()
        rev  = patient_to_dataset(load_dataset_participants(path))
        work["Dataset"] = work[patient_col].map(rev)
        unmapped = work.loc[work["Dataset"].isna(), patient_col].unique()
        if len(unmapped):
            if warn_unmapped:
                print(f"[summarize_zone_a_per_dataset] {len(unmapped)} patient "
                      f"id(s) not in {Path(path).name}; dropping. "
                      f"Examples: {list(unmapped)[:5]}")
            work = work.dropna(subset=["Dataset"])
        ds_col_used = "Dataset"
    else:
        ds_col_used = dataset_col

    group_keys = [ds_col_used] + ([horizon_col] if horizon_col is not None else [])
    rows = []
    for keys, sub in work.groupby(group_keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        ds_name = keys[0]
        horizon = keys[1] if horizon_col is not None else "-"
        ref  = sub[truth_col].to_numpy(dtype=float)
        pred = sub[pred_col].to_numpy(dtype=float)
        ceg = ceg_zone_pcts(ref, pred)
        seg_mean, seg_a = seg_scores(ref, pred)
        rows.append({
            "Method": method_name,
            "Dataset": ds_name,
            "Horizon": str(horizon),
            "N": int(len(sub)),
            "CEG Zone A (%)":  round(ceg["A"], 2),
            "SEG Zone A (%)":  round(seg_a, 2),
            "SEG Mean |Risk|": round(seg_mean, 4),
        })
    return (pd.DataFrame(rows)
              .sort_values(["Method", "Dataset", "Horizon"])
              .reset_index(drop=True))


def zone_a_pivot(per_method_dfs: Iterable[pd.DataFrame], *,
                 value_col: str = "CEG Zone A (%)",
                 horizon: str | None = None) -> pd.DataFrame:
    combined = pd.concat(list(per_method_dfs), ignore_index=True)
    if horizon is not None:
        combined = combined[combined["Horizon"].astype(str) == str(horizon)]
    return combined.pivot_table(index="Dataset",
                                columns="Method",
                                values=value_col)


def summarize_zone_a_per_participant(
    df: pd.DataFrame,
    *,
    patient_col: str = "participant_id",
    pred_col: str = "prediction",
    truth_col: str = "ground_truth",
    dataset_col: str = "dataset",
    horizon_col: str | None = "horizon_minutes",
    method_name: str = "Method",
) -> pd.DataFrame:
    """Per (dataset, [horizon,] participant) CEG/SEG Zone A summary."""
    group_cols = [dataset_col]
    if horizon_col is not None:
        group_cols.append(horizon_col)
    group_cols.append(patient_col)

    rows = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        ds_name = keys[0]
        if horizon_col is not None:
            horizon, participant = keys[1], keys[2]
        else:
            horizon, participant = None, keys[1]
        ref  = sub[truth_col].to_numpy(dtype=float)
        pred = sub[pred_col].to_numpy(dtype=float)
        ceg = ceg_zone_pcts(ref, pred)
        seg_mean, seg_a = seg_scores(ref, pred)
        row = {
            "Method": method_name,
            "Dataset": ds_name,
            "Participant": participant,
            "N": int(len(sub)),
            "CEG Zone A (%)":  float(ceg["A"]),
            "SEG Zone A (%)":  float(seg_a),
            "SEG Mean |Risk|": float(seg_mean),
        }
        if horizon_col is not None:
            row["Horizon"] = horizon
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_zone_a_across_participants(
    per_participant_df: pd.DataFrame,
    *,
    open_excluded: Iterable[str] = ("5_T1DEXI", "OhioT1DM", "8_DiaTrend"),
    horizon_col: str | None = "Horizon",
    digits: int = 2,
) -> pd.DataFrame:
    """Aggregate per-participant Zone A to mean/std per (Method, Dataset, [Horizon])
    plus Overall (Open) and Overall (Total). Mirrors clarke.ceg_seg but on the
    SEG-correct columns produced by ``summarize_zone_a_per_participant`` here.
    """
    open_set = set(open_excluded)
    has_horizon = horizon_col is not None and horizon_col in per_participant_df.columns

    def _row(sub, method, dataset, horizon=None):
        row = {"Method": method, "Dataset": dataset}
        if has_horizon:
            row["Horizon"] = horizon
        row["N_Participants"]      = int(len(sub))
        row["CEG Zone A mean (%)"] = round(float(sub["CEG Zone A (%)"].mean()), digits)
        row["CEG Zone A std (%)"]  = round(float(sub["CEG Zone A (%)"].std()),  digits)
        row["SEG Zone A mean (%)"] = round(float(sub["SEG Zone A (%)"].mean()), digits)
        row["SEG Zone A std (%)"]  = round(float(sub["SEG Zone A (%)"].std()),  digits)
        row["SEG |Risk| mean"]     = round(float(sub["SEG Mean |Risk|"].mean()), 4)
        row["SEG |Risk| std"]      = round(float(sub["SEG Mean |Risk|"].std()),  4)
        return row

    rows = []
    per_ds_keys = ["Method", "Dataset"] + (["Horizon"] if has_horizon else [])
    for keys, sub in per_participant_df.groupby(per_ds_keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        method, dataset = keys[0], keys[1]
        horizon = keys[2] if has_horizon else None
        rows.append(_row(sub, method, dataset, horizon))

    overall_keys = ["Method"] + (["Horizon"] if has_horizon else [])
    open_df = per_participant_df[~per_participant_df["Dataset"].isin(open_set)]
    for keys, sub in open_df.groupby(overall_keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        method = keys[0]
        horizon = keys[1] if has_horizon else None
        rows.append(_row(sub, method, "Overall (Open)", horizon))
    for keys, sub in per_participant_df.groupby(overall_keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        method = keys[0]
        horizon = keys[1] if has_horizon else None
        rows.append(_row(sub, method, "Overall (Total)", horizon))

    out = pd.DataFrame(rows)
    sort_keys = ["Method", "Dataset"] + (["Horizon"] if has_horizon else [])
    return out.sort_values(sort_keys).reset_index(drop=True)


# ─── CLI ──────────────────────────────────────────────────────────────────────
def _cli() -> None:
    p = argparse.ArgumentParser(
        description="SEG table utilities."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--freeze", action="store_true",
                   help="Snapshot the currently-loaded SEG table to "
                        "clarke/seg_risk_table.npy for offline use.")
    g.add_argument("--convert",
                   help="Convert an external SEG table (.xlsx/.csv/.tsv) "
                        "into clarke/seg_risk_table.npy.")
    p.add_argument("--out", default=str(_HERE / "seg_risk_table.npy"),
                   help="Output .npy path (default: clarke/seg_risk_table.npy).")
    args = p.parse_args()

    if args.freeze:
        out = Path(args.out)
        np.save(out, SEG_TABLE)
        print(f"Froze {SEG_TABLE.shape[0]}x{SEG_TABLE.shape[1]} SEG table -> {out}")
        return

    src = Path(args.convert)
    if src.suffix.lower() in {".xlsx", ".xls"}:
        table = pd.read_excel(src, header=None).to_numpy(dtype=np.float32)
    elif src.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if src.suffix.lower() == ".tsv" else ","
        table = pd.read_csv(src, header=None, sep=sep).to_numpy(dtype=np.float32)
    else:
        raise SystemExit(f"Unsupported source extension: {src.suffix}")

    if (table < 0).any():
        table = np.abs(table)
    _validate_seg_table(table, source=str(src))
    out = Path(args.out)
    np.save(out, table)
    print(f"Saved {table.shape[0]}x{table.shape[1]} SEG table -> {out}")


if __name__ == "__main__":
    _cli()
