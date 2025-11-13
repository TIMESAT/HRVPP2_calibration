#!/usr/bin/env python3
"""
Summarize EVI2_GPP_VPP-style CSVs with accuracy metrics per setting.

High-core, large-RAM optimized:
- Vectorized reducers (NumPy)
- Large setting batches (4096+)
- Parallel across batches with fixed-schema block outputs
- One concat + groupby to stitch results (no repeated outer merges)

Outputs:
  1) Overall (all sites pooled)
  2) By land cover (lc_id)

Rules:
- For {SOSD, EOSD, MAXD, LENGTH}: MAE, RMSE, Bias
- Else: R2

Assumptions:
- Columns: site, lc_id, id, GPP, and ≥1 "settings *" columns
- id formatted like "YYYY_sX_VPP"; VPP is last token
- Any YYDOY→sequential day conversion already applied upstream

Usage:
  python summarize_evi2_gpp_vpp.py \
      --input EVI2_GPP_VPP.csv \
      --out-all EVI2_SUMMARY__ALL.csv \
      --out-by-lc EVI2_SUMMARY__BY_LC.csv \
      --workers 90 --batch-size 4096 --engine pyarrow
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

DIRECT_VPPS = {"SOSD", "EOSD", "MAXD", "LENGTH"}
VIname = "LAI"

# -------------------------------
# Column helpers
# -------------------------------

def _detect_settings_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("settings ")]
    if not cols:
        raise ValueError("No 'settings ' columns found in input.")
    return cols

def _ensure_vpp_column(df: pd.DataFrame) -> pd.DataFrame:
    if "vpp" in df.columns:
        return df
    out = df.copy()
    out["vpp"] = out["id"].astype(str).str.split("_").str[-1]
    return out

def _metric_names(all_vpps: Iterable[str]) -> List[str]:
    all_vpps = list(all_vpps)
    names: List[str] = []
    for v in sorted(DIRECT_VPPS):
        names.extend([f"{v}_MAE", f"{v}_RMSE", f"{v}_Bias"])
    others = sorted([v for v in all_vpps if v not in DIRECT_VPPS])
    names.extend([f"{v}_R2" for v in others])
    # single N for each setting (per lc group / overall)
    names.append("N")
    return names

# -------------------------------
# Vectorized reducers
# -------------------------------

def _masked_stats_direct(ref: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = pd.to_numeric(pd.Series(ref), errors="coerce").to_numpy(dtype=float)
    vals = np.asarray(vals, dtype=float)
    m = np.isfinite(ref)[:, None] & np.isfinite(vals)
    if not m.any():
        m_cols = vals.shape[1]
        nan = np.full(m_cols, np.nan, dtype=float)
        return nan, nan, nan

    diffs = np.where(m, vals - ref[:, None], 0.0)
    cnt = m.sum(axis=0).astype(float)
    cnt[cnt == 0] = np.nan

    mae = np.sum(np.abs(diffs), axis=0) / cnt
    rmse = np.sqrt(np.sum(diffs * diffs, axis=0) / cnt)
    bias = np.sum(diffs, axis=0) / cnt
    return mae, rmse, bias

def _masked_r2(ref: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """
    Pearson's r^2 between ref (GPP-based VPP) and each column in vals (VI-based VPP),
    computed with per-column masks (ignore NaNs independently per column).

    Returns
    -------
    r2 : np.ndarray, shape (n_cols,)
        Pearson correlation squared for each settings column; NaN where undefined.
    """
    # coerce & broadcast
    ref = pd.to_numeric(pd.Series(ref), errors="coerce").to_numpy(dtype=float)
    X = np.asarray(vals, dtype=float)  # (n_rows, n_cols)

    # mask per column
    m = np.isfinite(ref)[:, None] & np.isfinite(X)
    if not m.any():
        return np.full(X.shape[1], np.nan, dtype=float)

    n = m.sum(axis=0).astype(float)
    n[n == 0] = np.nan  # avoid divide-by-zero

    # masked sums
    y_sum  = np.sum(np.where(m, ref[:, None], 0.0), axis=0)
    x_sum  = np.sum(np.where(m, X,            0.0), axis=0)
    yy_sum = np.sum(np.where(m, ref[:, None]**2, 0.0), axis=0)
    xx_sum = np.sum(np.where(m, X**2,             0.0), axis=0)
    xy_sum = np.sum(np.where(m, X * ref[:, None], 0.0), axis=0)

    # means
    y_bar = y_sum / n
    x_bar = x_sum / n

    # centered sums of squares / cross-products
    Sxx = xx_sum - n * x_bar**2
    Syy = yy_sum - n * y_bar**2
    Sxy = xy_sum - n * x_bar * y_bar

    with np.errstate(invalid="ignore", divide="ignore"):
        r = Sxy / np.sqrt(Sxx * Syy)
    r2 = r * r
    r2[~np.isfinite(r2)] = np.nan  # handles zero-variance or all-NaN columns
    return r2


# -------------------------------
# Block compute (fixed schema)
# -------------------------------

def _compute_block_fixed_schema(
    df_slice: pd.DataFrame,
    setting_cols: List[str],
    by_lc: bool,
    metric_cols: List[str],
) -> pd.DataFrame:
    """
    Returns a dataframe with keys + ALL metric_cols present.
    This allows final stitching via concat + groupby(first) (no outer merges).
    """
    rows: List[Dict[str, float]] = []
    key_cols = (["lc_id"] if by_lc else []) + ["Setting"]

    if by_lc:
        group_iter = df_slice.groupby("lc_id", dropna=False)
    else:
        group_iter = [(None, df_slice)]

    for lc_val, sub in group_iter:
        # Compute per-VPP metrics in blocks
        # Start a scratch dict per setting to fill all metrics
        base_records = []
        for s in setting_cols:
            rec = {"Setting": s}
            if by_lc:
                rec["lc_id"] = lc_val
            # initialize all metric columns to NaN (fixed schema)
            for mc in metric_cols:
                rec[mc] = np.nan
            base_records.append(rec)

        # ---- NEW: compute N per setting for this group ----
        gpp = pd.to_numeric(sub["GPP"], errors="coerce").to_numpy()
        X = sub[setting_cols].to_numpy(dtype=float)
        n_vec = (np.isfinite(gpp)[:, None] & np.isfinite(X)).sum(axis=0).astype(float)

        # write N once; same for all VPP metrics of this setting
        for j, s in enumerate(setting_cols):
            base_records[j]["N"] = float(n_vec[j])

        # For each VPP subset, compute stats vectorized across the block
        for vpp_name, part in sub.groupby("vpp", dropna=False):
            ref = part["GPP"].to_numpy()
            vals = part[setting_cols].to_numpy()  # (n_rows, n_block_cols)
            if vpp_name in DIRECT_VPPS:
                mae, rmse, bias = _masked_stats_direct(ref, vals)
                for j, s in enumerate(setting_cols):
                    base_records[j][f"{vpp_name}_MAE"]  = float(mae[j])
                    base_records[j][f"{vpp_name}_RMSE"] = float(rmse[j])
                    base_records[j][f"{vpp_name}_Bias"] = float(bias[j])
            else:
                r2 = _masked_r2(ref, vals)
                for j, s in enumerate(setting_cols):
                    base_records[j][f"{vpp_name}_R2"] = float(r2[j])

        rows.extend(base_records)

    if not rows:
        return pd.DataFrame(columns=key_cols + metric_cols)

    return pd.DataFrame(rows, columns=key_cols + metric_cols)

# -------------------------------
# Orchestrator
# -------------------------------

def _summarize(df: pd.DataFrame, *, by_lc: bool, workers: int, batch_size: int) -> pd.DataFrame:
    df = _ensure_vpp_column(df)
    settings = _detect_settings_columns(df)

    # Only columns needed in workers
    keep = ["GPP", "vpp"] + (["lc_id"] if by_lc else []) + settings
    df = df[keep]

    # Precompute full metric schema once, pass to workers
    metric_cols = _metric_names(df["vpp"].unique())

    # Build large batches to amortize overhead
    batches: List[List[str]] = [settings[i:i+batch_size] for i in range(0, len(settings), batch_size)]

    parts: List[pd.DataFrame] = []

    if workers <= 1:
        for cols in batches:
            parts.append(_compute_block_fixed_schema(df[["GPP", "vpp"] + (["lc_id"] if by_lc else []) + cols],
                                                     cols, by_lc, metric_cols))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    _compute_block_fixed_schema,
                    df[["GPP", "vpp"] + (["lc_id"] if by_lc else []) + cols],
                    cols,
                    by_lc,
                    metric_cols,
                )
                for cols in batches
            ]
            for f in as_completed(futs):
                parts.append(f.result())

    if not parts:
        base_cols = (["lc_id"] if by_lc else []) + ["Setting"]
        return pd.DataFrame(columns=base_cols + metric_cols)

    # Single concat + groupby to combine
    out = pd.concat(parts, ignore_index=True)
    key_cols = (["lc_id"] if by_lc else []) + ["Setting"]
    out = out.groupby(key_cols, dropna=False, as_index=False).first()

    # Stable ordering
    if by_lc:
        out.sort_values(["lc_id", "Setting"], inplace=True, kind="mergesort")
    else:
        out.sort_values(["Setting"], inplace=True, kind="mergesort")

    return out[key_cols + metric_cols]

def summarize_csv(inp: Path, out_all: Path, out_by_lc: Path, workers: int, batch_size: int, engine: str | None) -> None:
    # Fast CSV read (pyarrow uses multiple threads internally)
    read_kwargs = dict(dtype={"site": "string", "lc_id": "Int64", "id": "string"})
    if engine:
        read_kwargs["engine"] = engine
    else:
        try:
            read_kwargs["engine"] = "pyarrow"
        except Exception:
            pass

    df = pd.read_csv(inp, **read_kwargs)

    # Overall
    tbl_all = _summarize(df, by_lc=False, workers=workers, batch_size=batch_size)
    tbl_all.to_csv(out_all, index=False)

    # By land cover
    tbl_lc = _summarize(df, by_lc=True, workers=workers, batch_size=batch_size)
    tbl_lc.to_csv(out_by_lc, index=False)

# -------------------------------
# CLI
# -------------------------------

def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Summarize per-setting metrics from *_GPP_VPP.csv (vectorized + high-core parallel).")
    ap.add_argument("--input", default=VIname + "_GPP_VPP.csv", help="Input *_GPP_VPP.csv (from builder)")
    ap.add_argument("--out-all", default="output/" + VIname + "_SUMMARY__ALL.csv", help="Output CSV (overall)")
    ap.add_argument("--out-by-lc", default="output/" + VIname + "_SUMMARY__BY_LC.csv", help="Output CSV (by lc_id)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1),
                    help="Process workers (use physical/logical cores as desired)")
    ap.add_argument("--batch-size", type=int, default=4096,
                    help="Setting columns per task (use 2048–8192 for 90-core/1TB setups)")
    ap.add_argument("--engine", choices=["pyarrow", "c", "python"], default=None,
                    help="CSV engine (default: try pyarrow)")
    args = ap.parse_args(argv)

    summarize_csv(Path(args.input), Path(args.out_all), Path(args.out_by_lc),
                  workers=args.workers, batch_size=args.batch_size, engine=args.engine)
    print(f"Wrote: {Path(args.out_all).resolve()}")
    print(f"Wrote: {Path(args.out_by_lc).resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
