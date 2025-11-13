#!/usr/bin/env python3
"""
Parallel, memory‑savvy builder for:
  1) FAPAR_GPP_VPP__TEST2.csv  (first two FAPAR files)
  2) FAPAR_GPP_VPP.csv         (all FAPAR files)

Key behaviors (per request):
- Match by (year + season), not just by year.
- Convert YYDOY (>10000) for SOSD/EOSD/MAXD to sequential day via to_seq_day.
- Skip (year, season) groups where **all** GPP VPP are NaN.
- SITE_NAME and LC_ID parsed from FAPAR filename.
- Auto‑select matching GPP CSV per FAPAR file by site name.
- Union differing "settings *" columns across files; missing => NaN.

Performance:
- Uses ProcessPoolExecutor for per‑file parallelism.
- Reads only needed columns ("id" + "settings *") from FAPAR.
- Reads only needed columns from GPP ("id" + single value column).
- Writes per‑file Parquet shards to /tmp (configurable), then merges for final CSVs.

Requirements:
- Python 3.9+
- pandas >= 2.0
- pyarrow >= 14 (optional but strongly recommended)

Run:
    python build_fapar_gpp_vpp.py \
        --fapar-dir output/VI_lc/FAPAR/csv \
        --gpp-dir   output/GPP/csv \
        --out-all   FAPAR_GPP_VPP.csv \
        --out-test2 FAPAR_GPP_VPP__TEST2.csv \
        --workers 64

"""
import argparse
import os
import re
import sys
import glob
import uuid
import math
import shutil
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Iterable

import numpy as np
import pandas as pd

try:
    import pyarrow as pa  # noqa: F401
    import pyarrow.parquet as pq  # noqa: F401
    HAVE_PA = True
except Exception:
    HAVE_PA = False

VIname = "LAI" # PPI, FAPAR, LAI

DAYLIKE_VPPS = {"SOSD", "EOSD", "MAXD"}
FAPAR_PATTERN = VIname + "_*_LC*_mean_vpp.csv"

# ---------------------------
# Helpers
# ---------------------------

def parse_id_parts(_id: str) -> Tuple[int, str, str]:
    year, season, vpp = _id.split("_", 2)
    return int(year), season, vpp


def parse_site_lc_from_fname(fp: str) -> Tuple[str, int]:
    name = Path(fp).name
    m = re.search(r"_([A-Za-z0-9\-]+)_LC(\d+)_", name)
    if not m:
        raise ValueError(
            f"Cannot parse site/LC from filename: {name} (expected '*_<SITE>_LC<id>_*')"
        )
    return m.group(1), int(m.group(2))


def find_gpp_file(gpp_dir: str, site: str) -> str:
    strict_re = re.compile(rf"^{re.escape(site)}_GPP_.*_vpp\\.csv$", re.IGNORECASE)
    candidates = [fp for fp in glob.glob(str(Path(gpp_dir) / "*.csv"))
                  if strict_re.match(Path(fp).name)]
    if not candidates:
        for fp in glob.glob(str(Path(gpp_dir) / "*.csv")):
            name = Path(fp).name.lower()
            if site.lower() + "_gpp" in name and name.endswith("vpp.csv"):
                candidates.append(fp)
    if not candidates:
        raise FileNotFoundError(f"No GPP CSV found in {gpp_dir!r} for site {site!r}.")
    if len(candidates) > 1:
        candidates.sort(key=lambda p: len(Path(p).name), reverse=True)
    return candidates[0]


def to_seq_day(date):
    """Convert YYDOY (e.g., 17001, 18032, 17001.25) -> sequential day (2017-01-01 => 1)."""
    if not (isinstance(date, (int, float, np.floating)) or np.isscalar(date)):
        try:
            date = float(str(date))
        except Exception:
            return np.nan
    if not np.isfinite(date):
        return np.nan
    int_part = int(math.floor(float(date)))
    frac_part = float(date) - int_part
    s = str(int_part).zfill(5)
    try:
        year = int(s[:2])
        doy = int(s[-3:])
        return (year - 17) * 365 + doy + frac_part
    except ValueError:
        return np.nan


def convert_daylike_if_needed(vpp_name: str, value):
    if vpp_name in DAYLIKE_VPPS and pd.notna(value):
        try:
            v = float(value)
            return to_seq_day(v) if v > 10000 else v
        except Exception:
            return value
    return value


def _read_csv_header_cols(path: str) -> List[str]:
    # Fast header read
    return list(pd.read_csv(path, nrows=0).columns)


# ---------------------------
# Core per-file processor (worker)
# ---------------------------

def process_one_fapar_file(
    fapar_path: str,
    gpp_dir: str,
    shard_dir: str,
) -> Dict:
    """Process a single FAPAR CSV vs. its GPP; write a Parquet shard; return metadata."""
    site_name, lc_id = parse_site_lc_from_fname(fapar_path)
    gpp_path = find_gpp_file(gpp_dir, site_name)

    # ---- GPP fast scan to discover the single value column
    gpp_cols = _read_csv_header_cols(gpp_path)
    if "id" not in gpp_cols:
        raise ValueError(f"GPP file {Path(gpp_path).name} missing 'id' column")
    gpp_value_cols = [c for c in gpp_cols if c != "id"]
    if len(gpp_value_cols) != 1:
        raise ValueError(
            f"GPP file {Path(gpp_path).name} must have exactly one value column besides 'id'."
        )
    gpp_value_col = gpp_value_cols[0]

    gpp = pd.read_csv(
        gpp_path,
        usecols=["id", gpp_value_col],
        dtype={"id": "string"},
        engine="pyarrow" if HAVE_PA else None,
    )
    gpp_parsed = gpp.copy()
    gpp_parsed[["year", "season", "vpp"]] = gpp_parsed["id"].apply(
        lambda s: pd.Series(parse_id_parts(str(s)))
    )

    # Identify valid (year, season) with at least one non-NaN in the value column
    valid_groups = set()
    for (yr, ssn), sub in gpp_parsed.groupby(["year", "season"], dropna=False):
        if sub[gpp_value_col].notna().any():
            valid_groups.add((yr, ssn))

    if not valid_groups:
        # Nothing to emit for this file
        return {
            "ok": True,
            "rows": 0,
            "site": site_name,
            "lc_id": lc_id,
            "fapar_file": Path(fapar_path).name,
            "gpp_file": Path(gpp_path).name,
            "settings": [],
            "shard": None,
        }

    gpp_filtered = gpp_parsed[
        gpp_parsed[["year", "season"]].apply(tuple, axis=1).isin(valid_groups)
    ].copy()

    # Precompute GPP MAXD per (year, season) with conversion
    gpp_maxd_by_season: Dict[Tuple[int, str], float] = {}
    gpp_maxd_df = gpp_filtered[gpp_filtered["vpp"] == "MAXD"]
    for (yr, ssn), sub in gpp_maxd_df.groupby(["year", "season"], dropna=False):
        vals = sub[gpp_value_col].dropna()
        gm = vals.iloc[0] if not vals.empty else np.nan
        gm = convert_daylike_if_needed("MAXD", gm)
        gpp_maxd_by_season[(yr, ssn)] = float(gm) if pd.notna(gm) else np.nan

    # ---- FAPAR read: only id + settings*
    fapar_cols = _read_csv_header_cols(fapar_path)
    if "id" not in fapar_cols:
        raise ValueError(f"FAPAR file {Path(fapar_path).name} missing 'id' column")
    fapar_settings = [c for c in fapar_cols if c.startswith("settings ")]
    if not fapar_settings:
        raise ValueError(f"No FAPAR 'settings ' columns in {Path(fapar_path).name}")

    fapar = pd.read_csv(
        fapar_path,
        usecols=["id", *fapar_settings],
        dtype={"id": "string"},
        engine="pyarrow" if HAVE_PA else None,
    )

    # Decide FAPAR season per (year, season, setting)
    season_choice: Dict[Tuple[int, str, str], Optional[str]] = {}
    for (yr, ssn) in sorted(valid_groups):
        gm = gpp_maxd_by_season.get((yr, ssn), np.nan)
        for setting in fapar_settings:
            # compare FAPAR MAXD s1/s2 vs GPP MAXD
            s1_id = f"{yr}_s1_MAXD"
            s2_id = f"{yr}_s2_MAXD"
            s1 = fapar.loc[fapar["id"] == s1_id, setting]
            s2 = fapar.loc[fapar["id"] == s2_id, setting]
            s1_raw = s1.iloc[0] if not s1.empty else np.nan
            s2_raw = s2.iloc[0] if not s2.empty else np.nan
            s1_val = convert_daylike_if_needed("MAXD", s1_raw)
            s2_val = convert_daylike_if_needed("MAXD", s2_raw)
            s1_val = float(s1_val) if pd.notna(s1_val) else np.nan
            s2_val = float(s2_val) if pd.notna(s2_val) else np.nan
            chosen: Optional[str]
            if np.isnan(s1_val) and np.isnan(s2_val):
                chosen = None
            elif np.isnan(s1_val):
                chosen = "s2"
            elif np.isnan(s2_val):
                chosen = "s1"
            else:
                if pd.isna(gm):
                    chosen = "s1" if abs(s1_val) >= abs(s2_val) else "s2"
                else:
                    d1 = abs(s1_val - gm)
                    d2 = abs(s2_val - gm)
                    chosen = "s1" if d1 <= d2 else "s2"
            season_choice[(yr, ssn, setting)] = chosen

    # Build rows
    rows = []
    for _id, yr, ssn, vpp in gpp_filtered[["id", "year", "season", "vpp"]].itertuples(index=False):
        gpp_val_raw = gpp.loc[gpp["id"] == _id, gpp_value_col]
        gpp_val_raw = gpp_val_raw.iloc[0] if not gpp_val_raw.empty else np.nan
        gpp_val = convert_daylike_if_needed(vpp, gpp_val_raw)

        fapar_vals = []
        for setting in fapar_settings:
            chosen = season_choice.get((yr, ssn, setting))
            if chosen is None:
                fapar_vals.append(np.nan)
                continue
            fapar_id = f"{yr}_{chosen}_{vpp}"
            fv_raw = fapar.loc[fapar["id"] == fapar_id, setting]
            fv_raw = fv_raw.iloc[0] if not fv_raw.empty else np.nan
            fapar_vals.append(convert_daylike_if_needed(vpp, fv_raw))

        rows.append([site_name, lc_id, _id, gpp_val] + fapar_vals)

    if not rows:
        return {
            "ok": True,
            "rows": 0,
            "site": site_name,
            "lc_id": lc_id,
            "fapar_file": Path(fapar_path).name,
            "gpp_file": Path(gpp_path).name,
            "settings": fapar_settings,
            "shard": None,
        }

    out_cols = ["site", "lc_id", "id", "GPP"] + fapar_settings
    out_df = pd.DataFrame(rows, columns=out_cols)

    shard_path = Path(shard_dir) / f"shard_{uuid.uuid4().hex}.parquet"
    if HAVE_PA:
        out_df.to_parquet(shard_path, index=False)
    else:
        # Fallback: CSV (larger); we'll still treat path as a shard
        shard_path = shard_path.with_suffix(".csv")
        out_df.to_csv(shard_path, index=False)

    return {
        "ok": True,
        "rows": len(out_df),
        "site": site_name,
        "lc_id": lc_id,
        "fapar_file": Path(fapar_path).name,
        "gpp_file": Path(gpp_path).name,
        "settings": fapar_settings,
        "shard": str(shard_path),
    }


# ---------------------------
# Batch runner
# ---------------------------

def _collect_fapar_files(fapar_dir: str) -> List[str]:
    files = sorted(glob.glob(str(Path(fapar_dir) / FAPAR_PATTERN)))
    if not files:
        raise FileNotFoundError(
            f"No FAPAR files under {fapar_dir!r} with pattern {FAPAR_PATTERN!r}."
        )
    return files


def _normalize_and_write(shard_paths: List[str], out_csv: str) -> None:
    if not shard_paths:
        # create empty file
        pd.DataFrame(columns=["site", "lc_id", "id", "GPP"]).to_csv(out_csv, index=False)
        return

    # Read all shards; build union of setting columns
    frames: List[pd.DataFrame] = []
    settings_union: set = set()

    for sp in shard_paths:
        if sp.endswith(".parquet") and HAVE_PA:
            df = pd.read_parquet(sp)
        else:
            df = pd.read_csv(sp)
        frames.append(df)
        settings_union.update([c for c in df.columns if c.startswith("settings ")])

    union_cols = ["site", "lc_id", "id", "GPP"] + sorted(settings_union)
    norm_frames = []
    for df in frames:
        for c in union_cols:
            if c not in df.columns:
                df[c] = np.nan
        norm_frames.append(df[union_cols])

    out_df = pd.concat(norm_frames, ignore_index=True)
    out_df.sort_values(by=["site", "lc_id", "id"], inplace=True, kind="mergesort")
    out_df.to_csv(out_csv, index=False)


def run(fapar_dir: str, gpp_dir: str, out_all: str, out_test2: str, workers: int, tmpdir: str):
    Path(tmpdir).mkdir(parents=True, exist_ok=True)

    fapar_files = _collect_fapar_files(fapar_dir)
    print(f"Discovered {len(fapar_files)} FAPAR file(s).")

    # ---- Phase 1: TEST on first two (serial for clearer logs)
    test_subset = fapar_files[:2]
    test_shards: List[str] = []
    print("\n[TEST] Processing first two FAPAR files...")
    for fp in test_subset:
        try:
            info = process_one_fapar_file(fp, gpp_dir, tmpdir)
            if info.get("shard"):
                test_shards.append(info["shard"])
            print(f"  OK: {info['fapar_file']} -> GPP: {info['gpp_file']} (rows={info['rows']}, settings={len(info['settings'])})")
        except Exception as e:
            print(f"  ERROR {Path(fp).name}: {e}")
    if test_shards:
        _normalize_and_write(test_shards, out_test2)
        print(f"[TEST] Wrote: {Path(out_test2).resolve()}")
    else:
        print("[TEST] No outputs to write (all test files failed).")

    # ---- Phase 2: ALL files (parallel)
    print("\n[ALL] Processing all FAPAR files in parallel...")
    all_infos: List[Dict] = []
    all_shards: List[str] = []

    from concurrent.futures import ProcessPoolExecutor, as_completed
    # cap workers sensibly
    if workers < 1:
        workers = max(1, (os.cpu_count() or 1) - 1)

    # Windows/macOS need 'spawn'; on Linux default 'fork' is fine but 'spawn' is safer with PyArrow
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(process_one_fapar_file, fp, gpp_dir, tmpdir): fp
            for fp in fapar_files
        }
        for fut in as_completed(futs):
            fp = futs[fut]
            try:
                info = fut.result()
                all_infos.append(info)
                if info.get("shard"):
                    all_shards.append(info["shard"])
                print(f"  OK: {Path(fp).name} -> GPP: {info['gpp_file']} (rows={info['rows']}, settings={len(info['settings'])})")
            except Exception as e:
                print(f"  ERROR {Path(fp).name}: {e}")

    if not all_shards:
        raise RuntimeError("No per-file outputs produced; please check errors above.")

    _normalize_and_write(all_shards, out_all)
    print(f"[ALL] Wrote: {Path(out_all).resolve()}")

    # Summary
    try:
        summary_df = pd.DataFrame([
            {
                "site": i.get("site"),
                "lc_id": i.get("lc_id"),
                "gpp_file": i.get("gpp_file"),
            }
            for i in all_infos if i.get("ok")
        ]).drop_duplicates()
        sites = ", ".join(sorted(summary_df["site"].astype(str).unique()))
        print("\nSummary:")
        print(f"  Sites: {sites or '(none)'}")
        print(f"  Files processed: {len(all_shards)} FAPAR CSV(s)")
        print(f"  Unique GPP files used: {summary_df.shape[0]}")
    except Exception:
        pass

    # Optional cleanup: keep shards for audit; uncomment to remove
    # for p in all_shards + test_shards:
    #     try:
    #         Path(p).unlink(missing_ok=True)
    #     except Exception:
    #         pass


# ---------------------------
# CLI
# ---------------------------

def main(argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(description="Build combined FAPAR_GPP_VPP CSVs (parallel)")
    p.add_argument("--fapar-dir", default="output/VI_lc/" + VIname + "/csv", help=VIname + " CSV dir")
    p.add_argument("--gpp-dir",   default="output/GPP/csv", help="GPP CSV dir")
    p.add_argument("--out-all",   default=VIname + "_GPP_VPP.csv", help="Output CSV (all files)")
    p.add_argument("--out-test2", default=VIname + "_GPP_VPP__TEST2.csv", help="Output CSV (first two files)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1), help="Parallel workers")
    p.add_argument("--tmpdir", default=str(Path("/tmp")/"fapar_gpp_vpp_shards"), help="Temp dir for shards")
    args = p.parse_args(argv)

    # Friendly warnings
    if not HAVE_PA:
        warnings.warn("pyarrow not found; falling back to slower CSV/Parquet handling.")

    run(
        fapar_dir=args.fapar_dir,
        gpp_dir=args.gpp_dir,
        out_all=args.out_all,
        out_test2=args.out_test2,
        workers=args.workers,
        tmpdir=args.tmpdir,
    )


if __name__ == "__main__":
    sys.exit(main())
