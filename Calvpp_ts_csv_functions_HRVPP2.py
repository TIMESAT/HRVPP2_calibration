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





def _ts_run_(csvfilename, output_yfit_file_name, output_vpp_file_name, output_png_name):
    
    p_ignoreday = 366
    
    if 'GPP' in csvfilename:
        p_a = [[-9999., 9999., 1.], [1., 99.9, 1.], [0., 0., 1.]]
        p_ylu = [0., 9999.]
    else:
        p_ylu = [0, 1]
        p_a = [[100., 100., 1.], [1., 99.9, 0.5], [0., 0., 0.]]
    p_outststep = 1
    p_nodata = -9999
    p_davailwin = 45
    p_outlier = 0

    p_smooth = 10000
    p_nenvi = 1
    p_wfactnum = 2
    p_startmethod = 1
    p_startcutoff = [0.5, 0.5]
    p_low_percentile = 0.0
    p_fillbase = 1
    p_hrvppformat = 1
    p_seasonmethod = 1
    p_seapar = 1
    p_printflag = 0
    p_nclasses = 1

    # Initialize each variable with a vector of 255 elements filled with 0
    landuse = np.zeros(255, dtype='uint8')
    p_fitmethod = np.zeros(255, dtype='uint8')
    p_smooth = np.zeros(255, dtype='double')
    p_nenvi = np.ones(255, dtype='uint8')
    p_wfactnum = np.ones(255, dtype='double')
    p_startmethod = np.ones(255, dtype='uint8')
    p_startcutoff = np.full((255, 2), 0.5, order='F', dtype='double')
    p_low_percentile = np.full(255, 0.05, dtype='double')
    p_fillbase = np.zeros(255, dtype='uint8')
    p_seasonmethod = np.zeros(255, dtype='uint8')
    p_seapar = np.zeros(255, dtype='double')

    landuse[0] = 1



    df = pd.read_csv(csvfilename)

    timevector, tv_yyyymmdd, vi, qa, lc, yr, yrstart, yrend, z, y = read_table_data(df, p_a)

    #timevector, indices = np.unique(timevector, return_index=True)

    p_outindex = np.arange(1, yr*365+1)[::p_outststep]
    p_outindex_num = len(p_outindex)


    daily_timestep = []
    for year in range(yrstart, yrstart + yr):
        for day in range(365):
            # Start at January 1st and add `day` days to ensure only 365 days
            daily_timestep.append(datetime(year, 1, 1) + timedelta(days=day))
    daily_timestep = daily_timestep[::p_outststep]

    # Create the Date column first
    yfit_data = pd.DataFrame({"Date": daily_timestep})

    seasons = ["s1", "s2"]
    vppname = ["SOSD", "SOSV", "LSLOPE", "EOSD", "EOSV", "RSLOPE", "LENGTH", "MINV", "MAXD", "MAXV", "AMPL",
               "TPROD", "SPROD"]

    # Generate all combinations
    vpplist = [
        f"{year}_{season}_{name}"
        for year in range(yrstart, yrend + 1)
        for season in seasons
        for name in vppname
    ]

    # Create DataFrame
    vpp_data = pd.DataFrame({"id": vpplist})

    if 'GPP' in csvfilename:
        p_fitmethod_options = [2]
        p_smooth_options_method_2 = [1000]  # When p_fitmethod = 1
        p_smooth_options_method_1 = [1000]  # When p_fitmethod = 1 DL
        p_seasonmethod_options = [1]  # Depends on p_fitmethod
        p_seapar_options = [0.5]  # Depends on p_fitmethod
        p_startcutoff_options = [0.4]
    else:
        # Define parameter ranges
        p_fitmethod_options = [1, 2]
        p_smooth_options_method_2 = [100, 250, 500, 1000, 2500, 5000, 10000]  # When p_fitmethod = 1
        p_smooth_options_method_1 = [1000]  # When p_fitmethod = 1 DL
        p_seasonmethod_options = [1, 2]  # Depends on p_fitmethod
        p_seapar_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Depends on p_fitmethod
        p_startcutoff_options = np.arange(0.05, 0.55, 0.05)

    # Loop through all parameter combinations
    counter = 1
    

    for fitmethod_value in p_fitmethod_options:
        if fitmethod_value == 1:
            method_name = 'DL'
            p_smooth_options = p_smooth_options_method_1 
        elif fitmethod_value == 2:
            method_name = 'SP'
            p_smooth_options = p_smooth_options_method_2
        # Ensure `p_fitmethod` is an array with the same shape
        p_fitmethod.fill(fitmethod_value)  # Fill the entire array with the current fitmethod_value
        
        for p_smooth_val in p_smooth_options:
            p_smooth.fill(p_smooth_val)  # Ensure p_smooth is also in the correct shape
        
            for seasonmethod_value in p_seasonmethod_options:
                p_seasonmethod.fill(seasonmethod_value)  # Assign the same shape
        
                for seapar_value in p_seapar_options:
                    p_seapar.fill(seapar_value)  # Assign the same shape

                    for soscutoff_value in p_startcutoff_options:
                        p_startcutoff[0, 0] = soscutoff_value

                        for eoscutoff_value in p_startcutoff_options:
                            p_startcutoff[0, 1] = eoscutoff_value

                            # Generate yfit
                            vpp, vppqa, nseason, yfit, yfitqa, seasonfit, tseq = timesat.tsf2py(
                                yr, vi, qa, timevector, lc, p_nclasses, landuse, p_outindex,
                                p_ignoreday, p_ylu, p_printflag, p_fitmethod, p_smooth, p_nodata, p_outlier, p_davailwin, p_nenvi, p_wfactnum,
                                p_startmethod, p_startcutoff, p_low_percentile, p_fillbase, p_hrvppformat, p_seasonmethod, p_seapar,
                                y, 1, z, p_outindex_num)

                            if p_printflag == 1:
                                print(counter)

                            vpp = np.moveaxis(vpp, -1, 0)
                            vpp = np.squeeze(vpp, axis=-1)

                            yfit = np.moveaxis(yfit, -1, 0).astype('float32')
                            yfit = np.squeeze(yfit, axis=-1)
                            
                            # Generate column names dynamically
                            column_name = f"yfit_{counter}_fit{p_fitmethod[0]}_smooth{p_smooth[0]}_season{p_seasonmethod[0]}_seapar{p_seapar[0]}"
                            
                            # Convert to DataFrame
                            yfit_df = pd.DataFrame(yfit, columns=[column_name])
                            
                            # Append to the main DataFrame
                            yfit_data = pd.concat([yfit_data, yfit_df], axis=1)

                            # Convert to DataFrame
                            vpp_df = pd.DataFrame(vpp, columns=[column_name])
                            
                            # Append to the main DataFrame
                            vpp_data = pd.concat([vpp_data, vpp_df], axis=1)

                            # Increment counter for naming consistency
                            counter += 1
    
    print(counter)
    
    # Save the DataFrame to a CSV file
    # Create output folder if it doesn't exist
    if 'GPP' in csvfilename:
        yfit_data.to_csv(output_yfit_file_name, index=False)

    vpp_data.replace(-9999, np.nan, inplace=True)
    vpp_data.to_csv(output_vpp_file_name, index=False)


    if 'GPP' in csvfilename:
        # Convert to datetime objects
        y_row = vi.ravel()
        w_row = qa.ravel()
        t_row = [datetime.strptime(str(date), "%Y%m%d") for date in tv_yyyymmdd]

        # Filter data where w_row == 100 (clear-sky points)
        filtered_indices = w_row == 100
        t_row_filtered = [t_row[i] for i in range(len(t_row)) if filtered_indices[i]]
        y_row_filtered = y_row[filtered_indices]

        # Filter data where w_row > 0 (QA-filtered points)
        filtered_indices2 = w_row > 0
        t_row_filtered2 = [t_row[i] for i in range(len(t_row)) if filtered_indices2[i]]
        y_row_filtered2 = y_row[filtered_indices2]

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.plot(t_row_filtered2, y_row_filtered2, 'o', color='lightgrey', label='median-qa')
        plt.plot(t_row_filtered, y_row_filtered, 'ko', label='clear-sky')
        plt.plot(daily_timestep, yfit[:], 'r-', label=method_name)

            # ---- ADDITION: plot ALL SOSD (YYDOY float) and SOSV for all years & both seasons ----
        try:
            last_yfit_col = yfit_data.columns[-1]  # use the latest parameter combination

            # helper: extract 4-digit year from vpp_data['id'] like "2020_s1_SOSD"
            def extract_year(s):
                try:
                    return int(s.split('_', 1)[0])
                except Exception:
                    return None

            # build season-wise frames for SOSD/SOSV and align by year
            def season_frames(season: str):
                sosd = vpp_data[vpp_data['id'].str.endswith(f'_{season}_SOSD')][['id', last_yfit_col]].copy()
                sosv = vpp_data[vpp_data['id'].str.endswith(f'_{season}_SOSV')][['id', last_yfit_col]].copy()
                if sosd.empty or sosv.empty:
                    return pd.DataFrame()

                sosd['year'] = sosd['id'].apply(extract_year)
                sosv['year'] = sosv['id'].apply(extract_year)

                sosd = sosd.rename(columns={last_yfit_col: 'SOSD'})
                sosv = sosv.rename(columns={last_yfit_col: 'SOSV'})

                # inner-merge on year to align rows
                return pd.merge(sosd[['year', 'SOSD']], sosv[['year', 'SOSV']], on='year', how='inner').sort_values('year')

            # avoid duplicate legend entries
            added_label = {'s1_line': False, 's1_point': False, 's2_line': False, 's2_point': False}

            for season, line_key, point_key, line_label, point_label, vline_kwargs in [
                ('s1', 's1_line', 's1_point', 'SOSD (s1)', 'SOSV (s1)', dict(color='tab:blue', linestyle='--')),
                ('s2', 's2_line', 's2_point', 'SOSD (s2)', 'SOSV (s2)', dict(color='tab:orange', linestyle='--')),
            ]:
                df_season = season_frames(season)
                if df_season.empty:
                    continue

                for _, row in df_season.iterrows():
                    sosd_val = row['SOSD']
                    sosv_val = row['SOSV']
                    if pd.isna(sosd_val) or pd.isna(sosv_val):
                        continue

                    # convert YYDOY float to datetime
                    sos_date = yydoy_float_to_datetime(float(sosd_val))
                    if sos_date is None:
                        continue

                    # vertical line at SOS day
                    plt.axvline(
                        sos_date,
                        label=(line_label if not added_label[line_key] else None),
                        **vline_kwargs
                    )
                    added_label[line_key] = True

                    # point at (SOSD, SOSV)
                    plt.scatter(
                        sos_date,
                        float(sosv_val),
                        s=60,
                        zorder=5,
                        label=(point_label if not added_label[point_key] else None),
                        edgecolors='none'
                    )
                    added_label[point_key] = True

        except Exception as e:
            print(f"Warning: Could not plot SOSD/SOSV — {e}")
        # --------------------------------------------------------------------------------------


        # Formatting
        plt.xlabel("Date")
        if 'GPP' in csvfilename:
            plt.ylabel("Flux tower GPP")
        elif 'FAPAR' in csvfilename:
            plt.ylabel("FAPAR")
        else:
            plt.ylabel("VI")
        plt.grid(True)
        plt.legend()

        # Save the plot
        plt.savefig(output_png_name, dpi=300)
        plt.close()



