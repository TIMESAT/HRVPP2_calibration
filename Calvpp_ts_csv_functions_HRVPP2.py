"""
This script is part of the TIMESAT toolkit for the RAMONA project.

Author: Zhanzhang Cai

This Python script is designed to perform a variety of tasks related to time-series analysis of satellite sensor data, in the context of the RAMONA project.

Instructions for running the code:
1. Ensure that you have the necessary packages installed and the data available in the correct directories.
2. In the terminal, navigate to the directory containing this script.
3. Enter 'python run.py -help'.
4. Follow any prompts that appear in the terminal.

Please ensure you have read the associated documentation and understand the functions within this code before running it.

Please modify L20 in settings_test_evi2.json for the path of output.
For test:
Enter 'python run.py -i settings_test_evi2.json' in your terminal

For any further questions or assistance, contact Zhanzhang Cai.
"""
# import sys
# sys.path.append("/Users/zhanzhangcai/Documents/GitHub/TIMESAT_python")  # Replace with the actual path
# from memory_profiler import profile
import json
import math
import os
import numpy as np
import timesat
import time
import copy
import re
from datetime import datetime, timedelta, date
import pandas as pd
import matplotlib.pyplot as plt
import csv, gc
import glob


def assign_qa_weight(p_a, qa):
    p_a = np.asarray(p_a)
    
    # If p_a is empty, return the default qa_out.
    if p_a.size == 0:
        return qa

    # Create a new output array with default value 0.
    qa_out = np.zeros_like(qa, dtype=float)
    
    if p_a.shape[1] == 2:
        # Mapping: [qa_value, weight]
        for rule in p_a:
            qa_value, weight = rule
            mask = (qa == qa_value)
            qa_out[mask] = weight
    elif p_a.shape[1] == 3:
        # Mapping: [min_value, max_value, weight]
        for rule in p_a:
            min_val, max_val, weight = rule
            mask = (qa >= min_val) & (qa <= max_val)
            qa_out[mask] = weight
    else:
        raise ValueError("p_a must have either 2 or 3 columns.")
    
    return qa_out


def read_table_data(df, p_a):
    global ym3, wm3, tv_yyyymmdd, tv_yyyydoy

    # Identify the first column dynamically
    first_col_name = df.columns[0]

    # Convert the first column to datetime
    dates = pd.to_datetime(df[first_col_name])

    # Convert dates to the desired format yyyyDOY
    tv_yyyydoy = dates.dt.strftime('%Y%j').tolist()

    # Convert the first column to the desired format (assuming it contains dates)
    tv_yyyymmdd = dates.dt.strftime('%Y%m%d').tolist()

    # Get the number of rows
    npt = len(df)

    col = 1

    # Initialize 3D arrays
    ym3 = np.zeros((col, 1, npt), dtype=np.float32)
    wm3 = np.ones((col, 1, npt), dtype=np.float32)
    lc2 = np.ones((1, 1), dtype=np.float32)

    # Save data from the second column to the end into ym3
    # Transpose the slice to align with the shape (col, 1, npt)
    ym3[0, 0, :] = df.iloc[:, 1].T.values.astype(np.float32)
    try:
        wm3[0, 0, :] = df.iloc[:, 2].values.astype(np.float32)
    except:
        pass


    # Convert lists to numpy arrays
    tv_yyyymmdd = np.array(tv_yyyymmdd, dtype=int)
    tv_yyyydoy = np.array(tv_yyyydoy, dtype=int)

    # Extract year and calculate the number of unique years from YYYYMMDD format
    yv = tv_yyyymmdd // 10000  # Extract year from YYYYMMDD (integer division by 10000)
    yrstart = np.min(yv)  # First year
    yrend = np.max(yv)  # First year
    nyear = np.max(yv) - yrstart + 1  # Number of unique years (inclusive)

    # Compute qa_weight using the 2-column mapping
    wm3 = assign_qa_weight(p_a, wm3)

    return tv_yyyydoy, tv_yyyymmdd, ym3, wm3, lc2, nyear, yrstart, yrend, npt, col


def yydoy_float_to_datetime(sosd_day):
    """
    Convert a YYDOY float (e.g., 24123.5 -> 2024-05-02 12:00)
    into a Python datetime object.
    Handles fractional days and two-digit years.
    """

    yy = int(sosd_day // 1000)
    doy_float = sosd_day % 1000  # retains decimals (e.g., 123.5)


    # Interpret 2-digit year: 00–69 => 2000–2069, 70–99 => 1970–1999
    year = 2000 + yy

    doy_int = int(doy_float)

    sos_date = datetime(year, 1, 1) + timedelta(days=doy_int - 1)

    return sos_date



def assign_qa_weight(p_a, qa):
    p_a = np.asarray(p_a)
    
    # If p_a is empty, return the default qa_out.
    if p_a.size == 0:
        return qa

    # Create a new output array with default value 0.
    qa_out = np.zeros_like(qa, dtype=float)
    
    if p_a.shape[1] == 2:
        # Mapping: [qa_value, weight]
        for rule in p_a:
            qa_value, weight = rule
            mask = (qa == qa_value)
            qa_out[mask] = weight
    elif p_a.shape[1] == 3:
        # Mapping: [min_value, max_value, weight]
        for rule in p_a:
            min_val, max_val, weight = rule
            mask = (qa >= min_val) & (qa <= max_val)
            qa_out[mask] = weight
    else:
        raise ValueError("p_a must have either 2 or 3 columns.")
    
    return qa_out


def read_table_data(df, p_a):
    global ym3, wm3, tv_yyyymmdd, tv_yyyydoy

    # Identify the first column dynamically
    first_col_name = df.columns[0]

    # Convert the first column to datetime
    dates = pd.to_datetime(df[first_col_name])

    # Convert dates to the desired format yyyyDOY
    tv_yyyydoy = dates.dt.strftime('%Y%j').tolist()

    # Convert the first column to the desired format (assuming it contains dates)
    tv_yyyymmdd = dates.dt.strftime('%Y%m%d').tolist()

    # Get the number of rows
    npt = len(df)

    col = 1

    # Initialize 3D arrays
    ym3 = np.zeros((col, 1, npt), dtype=np.float32)
    wm3 = np.ones((col, 1, npt), dtype=np.float32)
    lc2 = np.ones((1, 1), dtype=np.float32)

    # Save data from the second column to the end into ym3
    # Transpose the slice to align with the shape (col, 1, npt)
    ym3[0, 0, :] = df.iloc[:, 1].T.values.astype(np.float32)
    try:
        wm3[0, 0, :] = df.iloc[:, 2].values.astype(np.float32)
    except:
        pass


    # Convert lists to numpy arrays
    tv_yyyymmdd = np.array(tv_yyyymmdd, dtype=int)
    tv_yyyydoy = np.array(tv_yyyydoy, dtype=int)

    # Extract year and calculate the number of unique years from YYYYMMDD format
    yv = tv_yyyymmdd // 10000  # Extract year from YYYYMMDD (integer division by 10000)
    yrstart = np.min(yv)  # First year
    yrend = np.max(yv)  # First year
    nyear = np.max(yv) - yrstart + 1  # Number of unique years (inclusive)

    # Compute qa_weight using the 2-column mapping
    wm3 = assign_qa_weight(p_a, wm3)

    return tv_yyyydoy, tv_yyyymmdd, ym3, wm3, lc2, nyear, yrstart, yrend, npt, col


def yydoy_float_to_datetime(sosd_day):
    """
    Convert a YYDOY float (e.g., 24123.5 -> 2024-05-02 12:00)
    into a Python datetime object.
    Handles fractional days and two-digit years.
    """

    yy = int(sosd_day // 1000)
    doy_float = sosd_day % 1000  # retains decimals (e.g., 123.5)


    # Interpret 2-digit year: 00–69 => 2000–2069, 70–99 => 1970–1999
    year = 2000 + yy

    doy_int = int(doy_float)

    sos_date = datetime(year, 1, 1) + timedelta(days=doy_int - 1)

    return sos_date





def _ts_run_(csvfilename,
             output_yfit_file_name,
             output_vpp_file_name,
             output_png_name,
             output_settings_file_name):

    # ---- Early exit if the main output already exists ----
    if os.path.isfile(output_vpp_file_name) and os.path.getsize(output_vpp_file_name) > 0:
        print(f"[SKIP] {output_vpp_file_name} already exists; skipping processing.")
        return

    p_ignoreday = 366

    if 'GPP' in csvfilename:
        p_a = [[-9999., 9999., 1.], [1., 99.9, 1.], [0., 0., 1.]]
        p_ylu = [0., 9999.]
    elif 'LAI' in csvfilename:
        p_ylu = [0, 10]
        p_a = [[-10000., 10000., 1.], [-10000., 10000., 1.], [-10000., 10000., 1.]]
    elif 'NDVI' in csvfilename:
        p_ylu = [0, 1]
        p_a = [[-10000., 10000., 1.], [-10000., 10000., 1.], [-10000., 10000., 1.]]
    elif 'PPI' in csvfilename:
        p_ylu = [0, 5]
        p_a = [[-10000., 10000., 1.], [-10000., 10000., 1.], [-10000., 10000., 1.]]
    elif 'EVI2' in csvfilename:
        p_ylu = [0, 1]
        p_a = [[-10000., 10000., 1.], [-10000., 10000., 1.], [-10000., 10000., 1.]]
    elif 'FAPAR' in csvfilename:
        p_ylu = [0, 1]
        p_a = [[-10000., 10000., 1.], [-10000., 10000., 1.], [-10000., 10000., 1.]]
    else:
        p_ylu = [0, 1]
        p_a = [[100., 100., 1.], [1., 99.9, 0.5], [0., 0., 0.]]
    p_outststep = 1
    p_nodata = -9999
    p_davailwin = 45
    p_outlier = 0


    p_hrvppformat = 1
    p_printflag = 0
    p_nclasses = 1

    # Initialize TIMESAT parameter vectors (length 255)
    landuse = np.zeros(255, dtype='uint8')
    p_fitmethod = np.zeros(255, dtype='uint8')
    p_smooth_vec = np.zeros(255, dtype='double')
    p_nenvi_vec = np.ones(255, dtype='uint8')
    p_wfactnum_vec = np.ones(255, dtype='double')
    p_startmethod_vec = np.ones(255, dtype='uint8')
    p_startcutoff_vec = np.full((255, 2), 0.5, order='F', dtype='double')
    p_low_percentile_vec = np.full(255, 0.05, dtype='double')
    p_fillbase_vec = np.ones(255, dtype='uint8')
    p_seasonmethod_vec = np.zeros(255, dtype='uint8')
    p_seapar_vec = np.zeros(255, dtype='double')

    landuse[0] = 1

    df = pd.read_csv(csvfilename)
    timevector, tv_yyyymmdd, vi, qa, lc, yr, yrstart, yrend, npt, col = read_table_data(df, p_a)

    p_outindex = np.arange(1, yr * 365 + 1)[::p_outststep]
    p_outindex_num = len(p_outindex)

    daily_timestep = []
    for year in range(yrstart, yrstart + yr):
        for day in range(365):
            daily_timestep.append(datetime(year, 1, 1) + timedelta(days=day))
    daily_timestep = daily_timestep[::p_outststep]

    # --- Prepare static tables (no repeated concat) ---
    seasons = ["s1", "s2"]
    vppname = ["SOSD", "SOSV", "LSLOPE", "EOSD", "EOSV", "RSLOPE", "LENGTH",
               "MINV", "MAXD", "MAXV", "AMPL", "TPROD", "SPROD"]
    vpplist = [
        f"{year}_{season}_{name}"
        for year in range(yrstart, yrend + 1)
        for season in seasons
        for name in vppname
    ]

    # Parameter grids
    if 'GPP' in csvfilename:
        p_fitmethod_options = [2]
        p_smooth_options_method_2 = [5000]
        p_smooth_options_method_1 = [5000]
        p_seasonmethod_options = [1]
        p_seapar_options = [0.5]
        p_startcutoff_options = [0.4]
        p_nenvi_vec[0] = 3
        p_wfactnum_vec[0] = 10
    else:
        p_fitmethod_options = [1, 2]
        p_smooth_options_method_2 = [100, 300, 1000, 3000, 10000]
        p_smooth_options_method_1 = [1000]
        p_seasonmethod_options = [1, 2]
        p_seapar_options = [0.0, 0.25, 0.5, 0.75, 1.0]
        p_startcutoff_options = np.arange(0.05, 0.55, 0.05)

    # Collect columns in memory
    vpp_columns = []        # list of 1D arrays length len(vpplist)
    yfit_columns = []       # list of 1D arrays length len(daily_timestep)
    settings_rows = []      # list of dicts (one per setting)

    counter = 0

    for fitmethod_value in p_fitmethod_options:
        if fitmethod_value == 1:
            method_name = 'DL'
            p_smooth_options = p_smooth_options_method_1
        elif fitmethod_value == 2:
            method_name = 'SP'
            p_smooth_options = p_smooth_options_method_2

        p_fitmethod.fill(fitmethod_value)

        for p_smooth_val in p_smooth_options:
            p_smooth_vec.fill(p_smooth_val)

            for seasonmethod_value in p_seasonmethod_options:
                p_seasonmethod_vec.fill(seasonmethod_value)

                for seapar_value in p_seapar_options:
                    p_seapar_vec.fill(seapar_value)

                    for soscutoff_value in p_startcutoff_options:
                        p_startcutoff_vec[0, 0] = soscutoff_value

                        for eoscutoff_value in p_startcutoff_options:
                            p_startcutoff_vec[1, 0] = eoscutoff_value

                            if 0:
                                print(f"method_name: {method_name}")
                                print(f"p_smooth_val: {p_smooth_val}")
                                print(f"seasonmethod_value: {seasonmethod_value}")
                                print(f"seapar_value: {seapar_value}")
                                print(f"soscutoff_value: {soscutoff_value}")
                                print(f"eoscutoff_value: {eoscutoff_value}")
                                print("\n--- DEBUG: shapes before timesat.tsf2py() ---")
                                def safe_shape(x):
                                    try:
                                        return np.shape(x)
                                    except Exception:
                                        return type(x)

                                debug_vars = {
                                    "yr": yr,
                                    "vi": safe_shape(vi),
                                    "qa": safe_shape(qa),
                                    "timevector": safe_shape(timevector),
                                    "lc": safe_shape(lc),
                                    "p_nclasses": p_nclasses,
                                    "landuse": safe_shape(landuse),
                                    "p_outindex": safe_shape(p_outindex),
                                    "p_ignoreday": p_ignoreday,
                                    "p_ylu": safe_shape(p_ylu),
                                    "p_fitmethod": safe_shape(p_fitmethod),
                                    "p_smooth_vec": safe_shape(p_smooth_vec),
                                    "p_nodata": p_nodata,
                                    "p_outlier": p_outlier,
                                    "p_davailwin": p_davailwin,
                                    "p_nenvi_vec": safe_shape(p_nenvi_vec),
                                    "p_wfactnum_vec": safe_shape(p_wfactnum_vec),
                                    "p_startmethod_vec": safe_shape(p_startmethod_vec),
                                    "p_startcutoff_vec": safe_shape(p_startcutoff_vec),
                                    "p_low_percentile_vec": safe_shape(p_low_percentile_vec),
                                    "p_fillbase_vec": safe_shape(p_fillbase_vec),
                                    "p_hrvppformat": p_hrvppformat,
                                    "p_seasonmethod_vec": safe_shape(p_seasonmethod_vec),
                                    "p_seapar_vec": safe_shape(p_seapar_vec),
                                    "col": col,
                                    "npt": npt,
                                    "p_outindex_num": p_outindex_num,
                                }
                                for k, v in debug_vars.items():
                                    print(f"{k:25s} : {v}")
                                print("--- END DEBUG ---\n")
                                exit()

                            # Run TIMESAT
                            vpp, vppqa, nseason, yfit, yfitqa, seasonfit, tseq = timesat.tsf2py(
                                yr, vi, qa, timevector, lc, p_nclasses, landuse, p_outindex,
                                p_ignoreday, p_ylu, p_printflag, p_fitmethod, p_smooth_vec, p_nodata,
                                p_davailwin, p_outlier, p_nenvi_vec, p_wfactnum_vec, p_startmethod_vec,
                                p_startcutoff_vec, p_low_percentile_vec, p_fillbase_vec, p_hrvppformat,
                                p_seasonmethod_vec, p_seapar_vec, col, 1, npt, p_outindex_num
                            )

                            # Reshape to 1D columns
                            vpp = np.squeeze(np.moveaxis(vpp, -1, 0), axis=-1)

                            if "GPP" in csvfilename:
                                vppqa = np.squeeze(np.moveaxis(vppqa, -1, 0), axis=-1)
                                # print(vpp.shape)
                                # print(vppqa.shape)
                                # print(vpp)
                                # print(vppqa)
                                n = vppqa.shape[0]
                                for i in range(n):
                                    if vppqa[i] < 3:
                                        vpp[i*13:(i+1)*13] = p_nodata
                                # print(vpp)

                            counter += 1
                            settings_id = f"settings {counter}"

                            # Save columns (defer DataFrame creation)
                            vpp_columns.append((settings_id, np.asarray(vpp, dtype=np.float32)))
                            
                            if 'GPP' in csvfilename:
                                yfit = np.squeeze(np.moveaxis(yfit, -1, 0).astype('float32'), axis=-1)
                                yfit_columns.append((settings_id, np.asarray(yfit, dtype=np.float32)))

                            # Record settings row
                            settings_rows.append({
                                "settings_id": settings_id,
                                "fitmethod": int(fitmethod_value),
                                "method_name": method_name,
                                "smooth": float(p_smooth_val),
                                "seasonmethod": int(seasonmethod_value),
                                "seapar": float(seapar_value),
                                "sos_cutoff": float(soscutoff_value),
                                "eos_cutoff": float(eoscutoff_value)
                            })

                            if p_printflag == 1:
                                print(counter)

    # --- Build DataFrames once (fast path) ---
    # Settings table
    settings_df = pd.DataFrame(settings_rows)

    # VPP table
    vpp_array = np.column_stack([col_vals for _, col_vals in vpp_columns]) if vpp_columns else np.empty((len(vpplist), 0))
    vpp_cols_names = [sid for sid, _ in vpp_columns]
    vpp_data = pd.DataFrame({"id": vpplist})
    if vpp_array.size > 0:
        vpp_df = pd.DataFrame(vpp_array, columns=vpp_cols_names)
        vpp_data = pd.concat([vpp_data, vpp_df], axis=1)
    vpp_data.replace(-9999, np.nan, inplace=True)
    vpp_data.to_csv(output_vpp_file_name, index=False)

    # Save settings CSV
    if counter > 1:
        settings_df.to_csv(output_settings_file_name, index=False)

    print(counter)
    lcname = ""

    
    if 'GPP' in csvfilename and yfit_columns:

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
            254: "Outside Area"
        }

        def extract_site_code(csvfilename: str) -> str | None:
            """
            Extract a site code like 'DE-Gri' or 'US-Ne1' from a filename or path.
            Allows letters or digits after the dash.
            """
            basename = os.path.basename(csvfilename)
            match = re.search(r"[A-Z]{2}-[A-Za-z0-9]{3}", basename)
            return match.group(0) if match else None

        def find_site_lc_csv(sitename: str, base_dir: str) -> str | None:
            """
            Find the single LC CSV file in the given base directory that contains the sitename.
            """
            pattern = os.path.join(base_dir, f"**/*{sitename}*LC*csv")
            matches = glob.glob(pattern, recursive=True)
            return matches[0] if matches else None

        def get_landcover_from_path(lc_csv_path: str) -> str:
            """
            Extract the LC value from a filename (e.g. '...LC12.csv') 
            and return the corresponding land cover label.
            """
            basename = os.path.basename(lc_csv_path)
            match = re.search(r"LC(\d+)", basename, flags=re.IGNORECASE)
            if not match:
                return "Unknown LC"
            lc_value = int(match.group(1))
            return LC_PALETTE.get(lc_value, "Unknown LC")

        sitename = extract_site_code(csvfilename)
        vicsvfilename = find_site_lc_csv(sitename, "VI_lc/EVI2/csv_lc")
        lcname = get_landcover_from_path(vicsvfilename)

        # YFIT table (GPP case only)
        yfit_array = np.column_stack([col_vals for _, col_vals in yfit_columns])
        yfit_cols_names = [sid for sid, _ in yfit_columns]
        yfit_data = pd.DataFrame({"Date": daily_timestep})
        yfit_df = pd.DataFrame(yfit_array, columns=yfit_cols_names)
        yfit_data = pd.concat([yfit_data, yfit_df], axis=1)
        yfit_data.to_csv(output_yfit_file_name, index=False)

        # --- Plot (unchanged logic, but uses last settings column name) ---
        # original data
        y_row = vi.ravel()
        w_row = qa.ravel()
        t_row = [datetime.strptime(str(d), "%Y%m%d") for d in tv_yyyymmdd]

        clr_mask = (w_row != 0)
        qa_mask = (w_row > 0)
        t_row_clr = [t_row[i] for i in range(len(t_row)) if clr_mask[i]]
        y_row_clr = y_row[clr_mask]
        t_row_qa = [t_row[i] for i in range(len(t_row)) if qa_mask[i]]
        y_row_qa = y_row[qa_mask]

        # last setting
        last_name, last_yfit = yfit_columns[-1]

        plt.figure(figsize=(8, 5))
        # plt.plot(t_row_qa, y_row_qa, 'o', color='lightgrey', label='median-qa')
        plt.plot(t_row_clr, y_row_clr, 'o', color='lightgrey', label='clear-sky')
        plt.plot(daily_timestep, last_yfit[:], 'r-', label='TIMESAT')

        # SOSD/SOSV and EOSD/EOSV overlays from last setting
        try:
            tmp_vpp = pd.DataFrame({"id": vpplist, last_name: vpp_data[last_name].values})

            def extract_year(s):
                try:
                    return int(s.split('_', 1)[0])
                except Exception:
                    return None

            def season_frames(season: str):
                sosd = tmp_vpp[tmp_vpp['id'].str.endswith(f'_{season}_SOSD')][['id', last_name]].copy()
                sosv = tmp_vpp[tmp_vpp['id'].str.endswith(f'_{season}_SOSV')][['id', last_name]].copy()
                eosd = tmp_vpp[tmp_vpp['id'].str.endswith(f'_{season}_EOSD')][['id', last_name]].copy()
                eosv = tmp_vpp[tmp_vpp['id'].str.endswith(f'_{season}_EOSV')][['id', last_name]].copy()

                if sosd.empty or sosv.empty or eosd.empty or eosv.empty:
                    return pd.DataFrame()

                for df in [sosd, sosv, eosd, eosv]:
                    df['year'] = df['id'].apply(extract_year)

                sosd = sosd.rename(columns={last_name: 'SOSD'})
                sosv = sosv.rename(columns={last_name: 'SOSV'})
                eosd = eosd.rename(columns={last_name: 'EOSD'})
                eosv = eosv.rename(columns={last_name: 'EOSV'})

                df = pd.merge(sosd[['year', 'SOSD']], sosv[['year', 'SOSV']], on='year', how='inner')
                df = pd.merge(df, eosd[['year', 'EOSD']], on='year', how='inner')
                df = pd.merge(df, eosv[['year', 'EOSV']], on='year', how='inner')

                return df.sort_values('year')

            added_label = {
                's1_sos_line': False, 's1_sos_point': False, 's1_eos_line': False, 's1_eos_point': False,
                's2_sos_line': False, 's2_sos_point': False, 's2_eos_line': False, 's2_eos_point': False
            }

            for season in ['s1', 's2']:
                df_season = season_frames(season)
                if df_season.empty:
                    continue

                for _, row in df_season.iterrows():
                    # --- Start of season ---
                    if not pd.isna(row['SOSD']) and not pd.isna(row['SOSV']):
                        sos_date = yydoy_float_to_datetime(float(row['SOSD']))
                        if sos_date is not None:
                            # plt.axvline(
                            #     sos_date,
                            #     color='black', linestyle='--',
                            #     label=(f'SOSD ({season})' if not added_label[f'{season}_sos_line'] else None)
                            # )
                            plt.scatter(
                                sos_date, float(row['SOSV']), s=70, zorder=5,
                                color='black', edgecolors='none', marker='^',
                                label=(f'SOS' if not added_label[f'{season}_sos_point'] else None)
                            )
                            # added_label[f'{season}_sos_line'] = True
                            added_label[f'{season}_sos_point'] = True

                    # --- End of season ---
                    if not pd.isna(row['EOSD']) and not pd.isna(row['EOSV']):
                        eos_date = yydoy_float_to_datetime(float(row['EOSD']))
                        if eos_date is not None:
                            # plt.axvline(
                            #     eos_date,
                            #     color='black', linestyle=':',
                            #     label=(f'EOSD ({season})' if not added_label[f'{season}_eos_line'] else None)
                            # )
                            plt.scatter(
                                eos_date, float(row['EOSV']), s=70, zorder=5,
                                color='black', edgecolors='none', marker='v',
                                label=(f'EOS' if not added_label[f'{season}_eos_point'] else None)
                            )
                            # added_label[f'{season}_eos_line'] = True
                            added_label[f'{season}_eos_point'] = True

        except Exception as e:
            print(f"Warning: Could not plot SOSD/SOSV/EOSD/EOSV — {e}")

        plt.xlabel("Date")
        if 'GPP' in csvfilename:
            plt.ylabel("Flux tower GPP")
        elif 'FAPAR' in csvfilename:
            plt.ylabel("FAPAR")
        else:
            plt.ylabel("VI")
        plt.grid(True)
        plt.legend()
        plt.title(sitename + ' (' + lcname + ')')  # ← Title from filename only
        plt.savefig(output_png_name, dpi=300)
        plt.close()

    return lcname
