#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LC_PALETTE = {
    0:  "Other (not defined)",
    1:  "Water",
    2:  "Wetland",
    3:  "Non-vegetated",
    4:  "Snow & Ice",
    5:  "Lichens & Mosses",
    6:  "Sealed",
    7:  "Broadleaf Forest",
    8:  "Coniferous Forest",
    9:  "Periodically Grassland",
    10: "Permanent Grassland",
    11: "Annual Cropland",
    12: "Permanent Cropland",
    13: "Rice",
    14: "Permanent herbaceous",
    15: "Low Woody",
    16: "Evergreen forest",
    254: "Outside Area",
}

def parse_args():
    ap = argparse.ArgumentParser(
        description=("Per land cover (lc_id), scatter the actual VPP_SCORE points for the top-K "
                     "(VI, method_name, seasonmethod) combos. Color by seapar (or any column), "
                     "keep same x-axis grouping as prior boxplots, and star the single highest point.")
    )
    ap.add_argument("--csv", required=True, help="Path to merged CSV (with VPP_SCORE).")
    ap.add_argument("--score-col", default="VPP_SCORE", help="Score column (default: VPP_SCORE).")
    ap.add_argument("--vi-col", default="VI", help="VI column (default: VI).")
    ap.add_argument("--method-col", default="method_name", help="Method column (default: method_name).")
    ap.add_argument("--seasonmethod-col", default="seasonmethod", help="Season method column (default: seasonmethod).")
    ap.add_argument("--seapar-col", default="seapar",
                    help="Grouping column for point color (default: seapar).")
    ap.add_argument("--lc-col", default="lc_id", help="Land cover column (default: lc_id).")
    ap.add_argument("--min-per-group", type=int, default=3,
                    help="Minimum rows per (VI,method,season) to be eligible (default: 3).")
    ap.add_argument("--top-k", type=int, default=20, help="Max number of groups per lc_id (default: 20).")
    ap.add_argument("--order-by", choices=["median","mean"], default="median",
                    help="Rank groups by this statistic to pick top-K (default: median).")
    ap.add_argument("--annotate-setting", action="store_true",
                    help="Include 'Setting' in the star annotation when available.")
    ap.add_argument("--jitter", type=float, default=0.15,
                    help="Horizontal jitter half-width for points (default: 0.15).")
    ap.add_argument("--alpha", type=float, default=0.7, help="Point alpha (default: 0.7).")
    ap.add_argument("--markersize", type=float, default=22, help="Marker size (default: 22).")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--width", type=float, default=14.0)
    ap.add_argument("--height", type=float, default=6.0)
    ap.add_argument("--outdir", default="figures_points_by_triple", help="Output directory.")
    return ap.parse_args()

def sanitize_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s))

def lc_name_from_id(lc_val):
    try:
        key = int(lc_val)
        return LC_PALETTE.get(key, f"lc_id={lc_val}")
    except Exception:
        return f"lc_id={lc_val}"

def ensure_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable: {list(df.columns)}")

def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]

    ensure_columns(df, [args.score_col, args.vi_col, args.method_col, args.seasonmethod_col, args.lc_col, args.seapar_col])

    # Numeric score
    df[args.score_col] = pd.to_numeric(df[args.score_col], errors="coerce")
    df = df.dropna(subset=[args.score_col])

    # Combo label: "VI • method • season"
    combo_col = "__VI_METHOD_SEASON__"
    df[combo_col] = (
        df[args.vi_col].astype(str).str.strip() + " • " +
        df[args.method_col].astype(str).str.strip() + " • " +
        df[args.seasonmethod_col].astype(str).str.strip()
    )

    os.makedirs(args.outdir, exist_ok=True)

    rng = np.random.default_rng(42)  # stable jitter

    for lc_val, df_lc in df.groupby(args.lc_col):
        if df_lc.empty:
            continue

        # Best single point for this LC
        idx_best = df_lc[args.score_col].idxmax()
        best_row = df_lc.loc[idx_best]
        best_score = float(best_row[args.score_col])
        best_combo = best_row[combo_col]
        best_vi = best_row[args.vi_col]
        best_method = best_row[args.method_col]
        best_season = best_row[args.seasonmethod_col]
        best_setting = best_row["Setting"] if (args.annotate_setting and "Setting" in df_lc.columns) else None

        # Eligibility and ranking for top-K groups
        counts = df_lc.groupby(combo_col)[args.score_col].size()
        eligible = counts[counts >= args.min_per_group].index.tolist()

        if args.order_by == "median":
            stat_series = df_lc.groupby(combo_col)[args.score_col].median()
        else:
            stat_series = df_lc.groupby(combo_col)[args.score_col].mean()

        candidates = stat_series.loc[eligible].sort_values(ascending=False).index.tolist()

        # Always include best combo
        if best_combo not in candidates:
            candidates = [best_combo] + candidates

        # Take top-K unique
        top_groups, seen = [], set()
        for g in candidates:
            if g not in seen:
                top_groups.append(g)
                seen.add(g)
            if len(top_groups) >= args.top_k:
                break

        df_plot = df_lc[df_lc[combo_col].isin(top_groups)].copy()
        if df_plot.empty:
            continue

        # Map each combo to an x position (1..K), same as boxplot positions
        x_positions = {g: i + 1 for i, g in enumerate(top_groups)}

        # Figure size scales with number of groups
        width = max(args.width, 0.7 * len(top_groups))
        fig, ax = plt.subplots(figsize=(width, args.height))

        # Scatter points by seapar category
        seacats = [str(x) for x in sorted(df_plot[args.seapar_col].astype(str).unique())]
        handles = []
        labels = []

        for sea in seacats:
            sub = df_plot[df_plot[args.seapar_col].astype(str) == sea]
            if sub.empty:
                continue
            # x with jitter
            x = sub[combo_col].map(x_positions).astype(float).values
            jitter = rng.uniform(-args.jitter, args.jitter, size=len(x))
            xj = x + jitter
            y = sub[args.score_col].values
            h = ax.scatter(xj, y, alpha=args.alpha, s=args.markersize, label=sea)
            handles.append(h)
            labels.append(sea)

        # Star the best single point
        x_best = x_positions.get(best_combo, None)
        if x_best is not None:
            ax.plot([x_best], [best_score], marker="*", markersize=14, color="red", markeredgecolor="black")
            label = (f"best: {best_vi} | {best_method} | {best_season}\n"
                     f"{args.score_col}={best_score:.3f}")
            if best_setting is not None:
                label += f"\nSetting={best_setting}"
            ax.annotate(
                label,
                xy=(x_best, best_score),
                xytext=(6, 10),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=9
            )

        # Axes, labels, legend
        lc_name = lc_name_from_id(lc_val)
        ax.set_title(f"{args.score_col} points by (VI, method, season) — {lc_name}")
        ax.set_xlabel("VI • method • seasonmethod")
        ax.set_ylabel(args.score_col)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xlim(0.5, len(top_groups) + 0.5)
        ax.set_xticks(range(1, len(top_groups) + 1))
        ax.set_xticklabels(top_groups, rotation=30, ha="right")

        if handles:
            ax.legend(handles=handles, labels=labels, title=args.seapar_col, frameon=False, fontsize=9)

        fig.tight_layout()
        fname = f"points_{sanitize_filename(args.score_col)}_by_VI_method_season_{sanitize_filename(lc_name)}.png"
        fig.savefig(os.path.join(args.outdir, fname), dpi=args.dpi)
        plt.close(fig)

    print(f"Saved scatter plots to: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
