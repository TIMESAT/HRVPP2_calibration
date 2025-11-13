#!/usr/bin/env python3
"""
Summarize EVI2_GPP_VPP-style CSVs with accuracy metrics per setting.

Outputs two CSVs:
  1) Overall (all sites pooled): per-setting metrics across all rows
  2) By land cover (lc_id): per-setting metrics computed within each lc_id

Rules:
- For VPPs in {SOSD, EOSD, MAXD, LENGTH}: compute MAE, RMSE, Bias
- For all other VPPs: compute R2 (coefficient of determination)

Assumptions:
- Input CSV includes columns: site, lc_id, id, GPP, and N≥1 "settings *" columns
- The "id" column is formatted like "YYYY_sX_VPP"; VPP is the token after the last underscore
- The builder already converted YYDOY (>10000) to sequential days for day-like VPPs

Usage:
    python summarize_evi2_gpp_vpp.py \
        --input EVI2_GPP_VPP.csv \
        --out-all EVI2_SUMMARY__ALL.csv \
        --out-by-lc EVI2_SUMMARY__BY_LC.csv

Dependencies:
    pandas >= 2.0, numpy
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd

DIRECT_VPPS = {"SOSD", "EOSD", "MAXD", "LENGTH"}

# -------------------------------
# Metric functions (NaN-safe)
# -------------------------------

def _mask_xy(ref: np.ndarray, val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ref = pd.to_numeric(pd.Series(ref), errors="coerce").to_numpy(dtype=float)
    val = pd.to_numeric(pd.Series(val), errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(ref) & np.isfinite(val)
    return ref[m], val[m]


def mae_rmse_bias(ref: np.ndarray, val: np.ndarray) -> Tuple[float, float, float]:
    ref, val = _mask_xy(ref, val)
    if ref.size == 0:
        return np.nan, np.nan, np.nan
    diffs = val - ref
    mae = float(np.mean(np.abs(diffs)))
    rmse = float(np.sqrt(np.mean(diffs ** 2)))
    bias = float(np.mean(diffs))
    return mae, rmse, bias


def r2_score(ref: np.ndarray, val: np.ndarray) -> float:
    ref, val = _mask_xy(ref, val)
    if ref.size == 0:
        return np.nan
    ss_res = float(np.sum((val - ref) ** 2))
    mu = float(np.mean(ref))
    ss_tot = float(np.sum((ref - mu) ** 2))
    # If ref is constant / degenerate, R^2 is undefined; return NaN
    if ss_tot == 0.0:
        return np.nan
    return 1.0 - ss_res / ss_tot


# -------------------------------
# Core summarizers
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


def _summarize_one(df: pd.DataFrame, *, by_lc: bool) -> pd.DataFrame:
    """Compute per-setting metrics; optionally within lc_id groups.
    Returns a wide table: one row per (setting[, lc_id]).
    """
    df = _ensure_vpp_column(df)
    settings = _detect_settings_columns(df)

    # Prepare containers
    rows: List[Dict[str, float]] = []

    # Grouping levels
    group_levels = ["lc_id"] if by_lc else [None]

    for group_key in group_levels:
        if group_key is None:
            subgroups = [(None, df)]
        else:
            subgroups = list(df.groupby(group_key, dropna=False))

        for lc_val, subdf in subgroups:
            # Iterate settings and compute metrics by VPP
            for setting in settings:
                rec: Dict[str, float] = {"Setting": setting}
                if by_lc:
                    rec["lc_id"] = lc_val

                # Split by VPP
                for vpp_name, part in subdf.groupby("vpp", dropna=False):
                    ref = part["GPP"].to_numpy()
                    val = part[setting].to_numpy()
                    if vpp_name in DIRECT_VPPS:
                        mae, rmse, bias = mae_rmse_bias(ref, val)
                        rec[f"{vpp_name}_MAE"] = mae
                        rec[f"{vpp_name}_RMSE"] = rmse
                        rec[f"{vpp_name}_Bias"] = bias
                    else:
                        r2 = r2_score(ref, val)
                        rec[f"{vpp_name}_R2"] = r2

                rows.append(rec)

    out = pd.DataFrame(rows)

    # Stable ordering
    metric_cols: List[str] = []
    # Direct metrics first, in a tidy order
    for v in sorted(DIRECT_VPPS):
        metric_cols.extend([f"{v}_MAE", f"{v}_RMSE", f"{v}_Bias"])
    # Then all other VPPs' R2 in alpha order as present
    other_vpps = sorted({v for v in df["vpp"].unique() if v not in DIRECT_VPPS})
    metric_cols.extend([f"{v}_R2" for v in other_vpps])

    # Ensure columns exist, even if missing (filled with NaN)
    for c in metric_cols:
        if c not in out.columns:
            out[c] = np.nan

    base_cols = (["lc_id"] if by_lc else []) + ["Setting"]
    out = out[base_cols + metric_cols]
    return out


def summarize_csv(inp: Path, out_all: Path, out_by_lc: Path) -> None:
    # Read with pyarrow engine if available for speed
    try:
        df = pd.read_csv(inp, dtype={"site": "string", "lc_id": "Int64", "id": "string"}, engine="pyarrow")
    except Exception:
        df = pd.read_csv(inp, dtype={"site": "string", "lc_id": "Int64", "id": "string"})

    # Overall summary (all sites pooled)
    tbl_all = _summarize_one(df, by_lc=False)
    tbl_all.to_csv(out_all, index=False)

    # By land cover summary
    tbl_lc = _summarize_one(df, by_lc=True)
    tbl_lc.sort_values(["lc_id", "Setting"], inplace=True, kind="mergesort")
    tbl_lc.to_csv(out_by_lc, index=False)


# -------------------------------
# CLI
# -------------------------------

def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Summarize per-setting metrics from *_GPP_VPP.csv")
    ap.add_argument("--input", default="EVI2_GPP_VPP.csv", help="Input *_GPP_VPP.csv (from builder)")
    ap.add_argument("--out-all", default="EVI2_SUMMARY__ALL.csv", help="Output CSV (overall)")
    ap.add_argument("--out-by-lc", default="EVI2_SUMMARY__BY_LC.csv", help="Output CSV (by lc_id)")
    args = ap.parse_args(argv)

    summarize_csv(Path(args.input), Path(args.out_all), Path(args.out_by_lc))
    print(f"Wrote: {Path(args.out_all).resolve()}")
    print(f"Wrote: {Path(args.out_by_lc).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
