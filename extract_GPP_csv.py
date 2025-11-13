#!/usr/bin/env python3

import pandas as pd
import re
from pathlib import Path
from collections import defaultdict

# 🔹 Root now points to the folder that contains all years
IN_ROOT = Path("GPP/V2/gpp_dd")
OUT_DIR = Path("GPP/V2/gpp_final")

FILENAME_RE = re.compile(
    r'(?P<site>[A-Z]{2}-[A-Za-z0-9]+)_flx_gpp_(?P<year>\d{4})_dd\.csv$', re.IGNORECASE
)

REQUIRED_COLS = ["TIMESTAMP", "GPP_DT_VUT_REF"]


def find_csvs(root: Path):
    """Yield (path, site, year) for files matching the filename pattern anywhere under root."""
    for p in root.rglob("*.csv"):
        m = FILENAME_RE.search(p.name)
        if m:
            yield p, m.group("site"), m.group("year")


def read_and_trim(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{fp} missing required columns: {missing}")
    out = df[REQUIRED_COLS].copy()
    out["TIMESTAMP"] = pd.to_datetime(out["TIMESTAMP"], errors="coerce", utc=True)
    return out


def merge_frames(frames) -> pd.DataFrame:
    """Concat, drop NA in target col, sort, and de-duplicate by TIMESTAMP."""
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["GPP_DT_VUT_REF"])
    merged = merged.sort_values("TIMESTAMP")
    merged = merged.drop_duplicates(subset=["TIMESTAMP"], keep="last")
    return merged


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Buckets
    by_site = defaultdict(list)        # site -> list[Path]
    by_site_year = defaultdict(list)   # (site, year) -> list[Path]

    # Discover files for all years
    for fp, site, year in find_csvs(IN_ROOT):
        by_site[site].append(fp)
        by_site_year[(site, year)].append(fp)

    # ---- Write per-site merged across all years
    for site, files in sorted(by_site.items()):
        frames = []
        for fp in sorted(files):
            frames.append(read_and_trim(fp))
        if not frames:
            continue

        merged = merge_frames(frames)
        out_fp = OUT_DIR / f"{site}_GPP_DT_VUT_REF_dd_all_years.csv"
        merged.to_csv(out_fp, index=False)
        print(f"[OK] Wrote {out_fp} ({len(merged)} rows)")

    # ---- Write per-site-year files
    # Put them in a subfolder for clarity
    per_year_dir = OUT_DIR / "by_site_year"
    per_year_dir.mkdir(parents=True, exist_ok=True)

    for (site, year), files in sorted(by_site_year.items()):
        frames = []
        for fp in sorted(files):
            frames.append(read_and_trim(fp))
        if not frames:
            continue

        merged = merge_frames(frames)
        out_fp = per_year_dir / f"{site}_{year}_GPP_DT_VUT_REF_dd.csv"
        merged.to_csv(out_fp, index=False)
        print(f"[OK] Wrote {out_fp} ({len(merged)} rows)")

    # Optional: brief summary
    print("\nSummary:")
    print(f"  Sites found: {len(by_site)}")
    print(f"  Site-Year groups: {len(by_site_year)}")


if __name__ == "__main__":
    main()
