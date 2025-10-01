#!/usr/bin/env python3

import pandas as pd
import re
from pathlib import Path

# 🔹 Hard-coded paths (relative to repo root)
IN_ROOT = Path("GPP/V2/gpp_dd/2017")
OUT_DIR = Path("GPP/V2/gpp_final")

FILENAME_RE = re.compile(
    r'(?P<site>[A-Z]{2}-[A-Za-z0-9]+)_flx_gpp_(?P<year>\d{4})_dd\.csv$', re.IGNORECASE
)

REQUIRED_COLS = ["TIMESTAMP", "GPP_DT_VUT_REF"]


def find_csvs(root: Path):
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_site = {}

    for fp, site, year in find_csvs(IN_ROOT):
        by_site.setdefault(site, []).append(fp)

    for site, files in sorted(by_site.items()):
        frames = []
        for fp in sorted(files):
            df = read_and_trim(fp)
            frames.append(df)

        if not frames:
            continue

        merged = pd.concat(frames, ignore_index=True)
        merged = merged.dropna(subset=["GPP_DT_VUT_REF"])
        merged = merged.sort_values("TIMESTAMP")
        merged = merged.drop_duplicates(subset=["TIMESTAMP"], keep="last")

        out_fp = OUT_DIR / f"{site}_GPP_DT_VUT_REF_dd.csv"
        merged.to_csv(out_fp, index=False)
        print(f"[OK] Wrote {out_fp} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
