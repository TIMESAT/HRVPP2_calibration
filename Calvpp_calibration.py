import os
import glob
import pandas as pd
import numpy as np

# Config
VPP_LIST = ["SOSD", "SOSV", "LSLOPE", "EOSD", "EOSV", "RSLOPE", "LENGTH",
            "MINV", "MAXD", "MAXV", "AMPL", "TPROD", "SPROD"]
VPP_COMPARE = [v for v in VPP_LIST if v != "MAXD"]  # compare all except MAXD

def ensure_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure there's an 'id' column like 'YYYY_s1_VPP'.
    If not present, try to construct from 'year','season','vpp' columns.
    """
    df = df.copy()
    cols_lower = {c.lower(): c for c in df.columns}
    if 'id' in df.columns:
        return df
    if all(k in cols_lower for k in ('year', 'season', 'vpp')):
        y = cols_lower['year']; s = cols_lower['season']; v = cols_lower['vpp']
        df['id'] = df[y].astype(str) + '_' + df[s].astype(str) + '_' + df[v].astype(str)
        return df
    raise ValueError("CSV must have 'id' column or (year, season, vpp) columns to construct it.")

def season_key_from_id(id_str: str) -> str:
    """Return 'YYYY_sX' part from 'YYYY_sX_VPP'."""
    return id_str.rsplit('_', 1)[0]

def split_by_vpp(df: pd.DataFrame, vpp_name: str) -> pd.DataFrame:
    """
    Filter rows for a given VPP (by id suffix), add 'season_key' = 'YYYY_sX',
    and keep only numeric setting columns.
    """
    sub = df[df['id'].str.endswith('_' + vpp_name)].copy()
    if sub.empty:
        return sub
    sub['season_key'] = sub['id'].apply(season_key_from_id)
    # Keep numeric setting columns (settings are columns)
    numeric_cols = sub.select_dtypes(include=[np.number]).columns.tolist()
    # If settings are not numeric dtypes yet, try coercion
    if not numeric_cols:
        # try convert all non-id columns to numeric (errors='ignore' for non-setting cols)
        tmp = sub.drop(columns=['id', 'season_key'], errors='ignore').apply(pd.to_numeric, errors='coerce')
        sub[tmp.columns] = tmp
        numeric_cols = sub.select_dtypes(include=[np.number]).columns.tolist()
    # Return only season_key + setting columns
    return sub[['season_key'] + numeric_cols]

def pearsonr_safe(x: np.ndarray, y: np.ndarray) -> float:
    """Safe Pearson correlation that handles small N."""
    if len(x) < 2 or len(y) < 2:
        return np.nan
    if np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((a - b) ** 2)))

def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.nanmean(np.abs(a - b)))

def bias(a: np.ndarray, b: np.ndarray) -> float:
    """Mean (a - b); sign indicates over/underestimation of a vs b."""
    return float(np.nanmean(a - b))

# Accumulator for all sites
all_results = []   # list of dicts: {site, setting, vpp, n, r, rmse, mae, bias}

# Define input and output directories

GPP_output_folder = 'output/GPP/csv'

FAPAR_output_folder = 'output/FAPAR/csv'

# Get all CSV files in the input folder
csv_files = [f for f in os.listdir(FAPAR_output_folder) if f.endswith('_vpp.csv')]

# Loop over all CSV files
for csv_file in csv_files:
    vi_file_path = os.path.join(FAPAR_output_folder, csv_file)
    print(vi_file_path)
    # Extract site name (assumes the filename ends with '_<site>.csv')
    site_name = os.path.splitext(os.path.basename(vi_file_path))[0].split('_')[-2]
    print(f"Site name: {site_name}")

    # Search for the reference file in GPP_input_folder that starts with the site name
    pattern = os.path.join(GPP_output_folder, f"{site_name}_*.csv")
    matches = glob.glob(pattern)

    if matches:
        ref_file_path = matches[0]  # Take the first match
        print(f"Reference file found: {ref_file_path}")

        # Load VI and GPP CSVs
        vi_df = pd.read_csv(vi_file_path)
        gpp_df = pd.read_csv(ref_file_path)

        # Ensure 'id' exists in both
        vi_df = ensure_id(vi_df)
        gpp_df = ensure_id(gpp_df)

        # --- MAXD-based season matching ---
        vi_maxd = split_by_vpp(vi_df, "MAXD")    # columns: season_key + settings
        gpp_maxd = split_by_vpp(gpp_df, "MAXD")

        if vi_maxd.empty or gpp_maxd.empty:
            print(f"MAXD missing in one of the files for site {site_name}. Skipping statistics.")
        else:
            # Find common seasons by season_key
            common = pd.merge(vi_maxd, gpp_maxd, on='season_key', how='inner', suffixes=('_vi', '_gpp'))
            if common.empty:
                print(f"No overlapping seasons (by MAXD season_key) for site {site_name}.")
            else:
                # Determine the shared settings: intersection of numeric columns (excluding season_key)
                vi_settings = [c.replace('_vi', '') for c in common.columns if c.endswith('_vi') and c != 'season_key_vi']
                gpp_settings = [c.replace('_gpp', '') for c in common.columns if c.endswith('_gpp') and c != 'season_key_gpp']
                shared_settings = sorted(set(vi_settings).intersection(gpp_settings))
                if not shared_settings:
                    print(f"No shared setting columns between VI and GPP for site {site_name}.")
                else:
                    # Prepare a dict of matched season_keys (for re-use across other VPPs)
                    matched_seasons = common['season_key'].tolist()

                    # For each VPP (except MAXD), compute stats per setting
                    for vpp in VPP_COMPARE:
                        vi_vpp = split_by_vpp(vi_df, vpp)
                        gpp_vpp = split_by_vpp(gpp_df, vpp)

                        if vi_vpp.empty or gpp_vpp.empty:
                            # If this vpp isn't present, skip gracefully
                            continue

                        # Filter to matched seasons only (as determined by MAXD)
                        vi_vpp = vi_vpp[vi_vpp['season_key'].isin(matched_seasons)]
                        gpp_vpp = gpp_vpp[gpp_vpp['season_key'].isin(matched_seasons)]

                        # Join on season_key
                        merged = pd.merge(vi_vpp, gpp_vpp, on='season_key', how='inner', suffixes=('_vi', '_gpp'))
                        if merged.empty:
                            continue

                        # For each shared setting, compute stats
                        for setting in shared_settings:
                            vi_col = setting + '_vi'
                            gpp_col = setting + '_gpp'
                            if vi_col not in merged.columns or gpp_col not in merged.columns:
                                continue

                            x = merged[vi_col].to_numpy(dtype=float)
                            y = merged[gpp_col].to_numpy(dtype=float)

                            # Drop NaN pairs
                            mask = ~np.isnan(x) & ~np.isnan(y)
                            x = x[mask]; y = y[mask]
                            if x.size == 0:
                                continue

                            res = {
                                'site': site_name,
                                'setting': setting,
                                'vpp': vpp,
                                'n': int(x.size),
                                'r': pearsonr_safe(x, y),
                                'rmse': rmse(x, y),
                                'mae': mae(x, y),
                                'bias': bias(x, y)  # (VI - GPP)
                            }
                            all_results.append(res)

                    print(f"Computed statistics for site {site_name} over {len(matched_seasons)} matched seasons.")

    else:
        ref_file_path = None
        print(f"No reference file found for site {site_name} in {GPP_output_folder}")

    

    

# After processing all files/sites:
if all_results:
    results_df = pd.DataFrame(all_results)
    # Aggregate per (setting, vpp) across sites
    summary = (results_df
               .groupby(['setting', 'vpp'], as_index=False)
               .agg(n=('n', 'sum'),
                    r_mean=('r', 'mean'),
                    r_median=('r', 'median'),
                    rmse_mean=('rmse', 'mean'),
                    mae_mean=('mae', 'mean'),
                    bias_mean=('bias', 'mean')))
    print("\n=== Summary across sites (by setting × VPP) ===")
    print(summary.head(20))

    # Save the summary DataFrame to CSV
    summary.to_csv("summary_across_sites.csv", index=False)
    print("\nSummary saved to 'summary_across_sites.csv'.")
    
else:
    print("No statistics were computed (check inputs/columns).")
