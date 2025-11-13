"""
Create FAPAR_GPP_VPP.csv by matching each GPP season to the *closest*
FAPAR season (per setting) using MAXD as the selector.

Revisions:
- Matching is **per season** (year+season), not per year.
- Convert ALL SOSD/EOSD/MAXD values (both GPP and FAPAR) from YYDOY (>10000) to sequential
  day counts via `to_seq_day` whenever encountered.
- Skip entire GPP seasons (year+season) that are completely empty (all VPP NaN).
- SITE_NAME and LC_ID parsed from the FAPAR filename.
- Auto-select the GPP CSV from a directory by matching the site name.
"""

import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------
# Configuration
# ---------------------------
fapar_path = "output/VI_lc/FAPAR/csv/FAPAR_2017_01_01_2024_12_31_FR-CLt_LC14_mean_vpp.csv"
gpp_dir    = "output/GPP/csv/"
out_path   = "FAPAR_GPP_VPP.csv"

DAYLIKE_VPPS = {"SOSD", "EOSD", "MAXD"}


# ---------------------------
# Helpers
# ---------------------------
def parse_id_parts(_id: str):
    """Parse 'YYYY_sX_VPP' -> (year:int, season:str, vpp:str)."""
    year, season, vpp = _id.split("_", 2)
    return int(year), season, vpp

def parse_site_lc_from_fname(fp: str):
    """
    Extract site and LC id from FAPAR filename, e.g.
    '..._FR-CLt_LC14_mean_vpp.csv' -> ('FR-CLt', 14)
    """
    name = Path(fp).name
    m = re.search(r"_([A-Za-z0-9\-]+)_LC(\d+)_", name)
    if not m:
        raise ValueError(
            f"Cannot parse site/LC from filename: {name} "
            "(expected pattern '*_<SITE>_LC<id>_*')"
        )
    site, lc = m.group(1), int(m.group(2))
    return site, lc

def find_gpp_file(gpp_dir: str, site: str) -> str:
    """Pick the GPP CSV for the given site from a directory."""
    strict_re = re.compile(rf"^{re.escape(site)}_GPP_.*_vpp\.csv$", re.IGNORECASE)
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
    """
    Convert a YYDOY date (e.g., 17001, 18032, 17001.25) into a sequential
    day count starting from 2017-01-01 = 1, ignoring leap years.
    Keeps fractional part. Formula: (year - 17) * 365 + DOY + fraction
    """
    if not np.isfinite(date):
        return np.nan
    int_part = int(np.floor(date))
    frac_part = date - int_part
    s = str(int_part).zfill(5)
    try:
        year = int(s[:2])
        doy  = int(s[-3:])
        return (year - 17) * 365 + doy + frac_part
    except ValueError:
        return np.nan

def convert_daylike_if_needed(vpp_name: str, value):
    """If vpp is day-like and value > 10000, apply to_seq_day; else return value."""
    if vpp_name in DAYLIKE_VPPS and pd.notna(value):
        try:
            v = float(value)
            if v > 10000:
                return to_seq_day(v)
            return v
        except Exception:
            return value
    return value

def choose_season_for_setting(fapar_df: pd.DataFrame,
                              year: int,
                              gpp_season: str,
                              setting_col: str,
                              gpp_maxd_value: float) -> str | None:
    """
    For a given (year, GPP season, setting), choose which FAPAR season (s1 or s2)
    to align with by comparing FAPAR MAXD(s1/s2) to the **GPP MAXD of that same
    (year, season)**. Returns "s1", "s2", or None if FAPAR lacks both.
    """
    s1_id = f"{year}_s1_MAXD"
    s2_id = f"{year}_s2_MAXD"

    s1 = fapar_df.loc[fapar_df["id"] == s1_id, setting_col]
    s2 = fapar_df.loc[fapar_df["id"] == s2_id, setting_col]
    s1_raw = s1.iloc[0] if not s1.empty else np.nan
    s2_raw = s2.iloc[0] if not s2.empty else np.nan

    s1_val = convert_daylike_if_needed("MAXD", s1_raw)
    s2_val = convert_daylike_if_needed("MAXD", s2_raw)

    s1_val = float(s1_val) if pd.notna(s1_val) else np.nan
    s2_val = float(s2_val) if pd.notna(s2_val) else np.nan

    if np.isnan(s1_val) and np.isnan(s2_val):
        return None
    if np.isnan(s1_val):
        return "s2"
    if np.isnan(s2_val):
        return "s1"

    # gpp_maxd_value is the (possibly converted) GPP MAXD for (year, gpp_season)
    if pd.isna(gpp_maxd_value):
        # Fallback: pick the larger |MAXD|
        return "s1" if abs(s1_val) >= abs(s2_val) else "s2"

    d1 = abs(s1_val - gpp_maxd_value)
    d2 = abs(s2_val - gpp_maxd_value)
    return "s1" if d1 <= d2 else "s2"


# ---------------------------
# Load data
# ---------------------------
fapar = pd.read_csv(fapar_path)
site_name, lc_id = parse_site_lc_from_fname(fapar_path)

gpp_path = find_gpp_file(gpp_dir, site_name)
gpp = pd.read_csv(gpp_path)

# ---------------------------
# Prepare GPP (parse IDs, drop empty seasons)
# ---------------------------
gpp_parsed = gpp.copy()
gpp_parsed[["year", "season", "vpp"]] = gpp_parsed["id"].apply(
    lambda s: pd.Series(parse_id_parts(s))
)

# Ensure only one value column in GPP
gpp_value_cols = [c for c in gpp.columns if c != "id"]
if len(gpp_value_cols) != 1:
    raise ValueError(
        f"GPP file {Path(gpp_path).name} must have exactly one value column besides 'id'."
    )
gpp_col = gpp_value_cols[0]

# Identify valid (year, season) groups where at least one VPP is non-NaN
valid_groups = set()
for (yr, ssn), sub in gpp_parsed.groupby(["year", "season"], dropna=False):
    if sub[gpp_col].notna().any():
        valid_groups.add((yr, ssn))

# Filter out completely empty seasons
gpp_filtered = gpp_parsed[gpp_parsed[["year", "season"]].apply(tuple, axis=1).isin(valid_groups)].copy()

# ---------------------------
# Precompute GPP MAXD per (year, season) with conversion
# ---------------------------
gpp_maxd_by_season = {}
for (yr, ssn), sub in gpp_filtered[gpp_filtered["vpp"] == "MAXD"].groupby(["year", "season"]):
    vals = sub[gpp_col].dropna()
    gm = vals.iloc[0] if not vals.empty else np.nan
    gm = convert_daylike_if_needed("MAXD", gm)
    gpp_maxd_by_season[(yr, ssn)] = float(gm) if pd.notna(gm) else np.nan

# ---------------------------
# Decide FAPAR season per (year, season, setting)
# ---------------------------
fapar_settings = [c for c in fapar.columns if c.startswith("settings ")]
if not fapar_settings:
    raise ValueError("No FAPAR 'settings ' columns found.")

season_choice = {}
for (yr, ssn) in sorted(valid_groups):
    gm = gpp_maxd_by_season.get((yr, ssn), np.nan)
    for setting in fapar_settings:
        season_choice[(yr, ssn, setting)] = choose_season_for_setting(
            fapar_df=fapar,
            year=yr,
            gpp_season=ssn,
            setting_col=setting,
            gpp_maxd_value=gm
        )

# ---------------------------
# Build output
# ---------------------------
rows = []
# Preserve original order but only for valid (year, season) rows
for _id, yr, ssn, vpp in gpp_filtered[["id", "year", "season", "vpp"]].itertuples(index=False):
    # GPP value (convert SOSD/EOSD/MAXD on the fly)
    gpp_val_raw = gpp.loc[gpp["id"] == _id, gpp_col]
    gpp_val_raw = gpp_val_raw.iloc[0] if not gpp_val_raw.empty else np.nan
    gpp_val = convert_daylike_if_needed(vpp, gpp_val_raw)

    # FAPAR values per setting, using the chosen FAPAR season for this (yr, ssn, setting)
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

out_cols = ["site", "lc_id", "id", "GPP"] + fapar_settings
out_df = pd.DataFrame(rows, columns=out_cols)
out_df.to_csv(out_path, index=False)

print(f"Selected GPP file: {Path(gpp_path).name}")
print(f"Saved: {Path(out_path).resolve()}")
