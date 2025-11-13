import os, re
import glob
import Calvpp_ts_csv_functions_HRVPP2
from collections import Counter


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
    pattern = os.path.join(base_dir, f"{sitename}*csv")
    matches = glob.glob(pattern)  # searches only in base_dir
    return matches[0] if matches else None

# Define input and output directories

input_folder = 'FAPAR/V5/timeseries/50m/'
output_folder = 'output/FAPAR/'

input_folder = 'GPP/V2/gpp_cal/'
output_folder = 'output/GPP/'

input_folder = 'VI_lc/LAI/csv_lc/'
output_folder = 'output/VI_lc/LAI/'

input_folder = 'VI_lc/PPI/csv_lc/'
output_folder = 'output/VI_lc/PPI/'

input_folder = 'VI_lc/NDVI/csv_lc/'
output_folder = 'output/VI_lc/NDVI/'

input_folder = 'VI_lc/EVI2/csv_lc/'
output_folder = 'output/VI_lc/EVI2/'

input_folder = 'VI_lc/FAPAR/csv_lc_100m/'
output_folder = 'output/VI_lc/FAPAR/'


csvpath = os.path.join(output_folder, 'csv')
pngpath = os.path.join(output_folder, 'png')
# Create output folder if it doesn't exist
os.makedirs(csvpath, exist_ok=True)
os.makedirs(pngpath, exist_ok=True)

# Get all CSV files in the input folder
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
# Initialize a counter to collect counts by land-cover name
lc_counts = Counter()

# Loop over all CSV files
for csv_file in csv_files:

    if find_site_lc_csv(extract_site_code(csv_file), "GPP/V2/gpp_cal/") is None:
        print(f"Cannot find GPP data for {csv_file}... continue")
        continue

    input_file_path = os.path.join(input_folder, csv_file)
    print(input_file_path)

    output_yfit_file_path = os.path.join(csvpath, os.path.splitext(csv_file)[0] + "_yfit.csv")
    output_vpp_file_path = os.path.join(csvpath, os.path.splitext(csv_file)[0] + "_vpp.csv")
    output_png_path = os.path.join(pngpath, os.path.splitext(csv_file)[0] + ".png")

    print(f"Processing {input_file_path}...")
    lcname = Calvpp_ts_csv_functions_HRVPP2._ts_run_(input_file_path, output_yfit_file_path, output_vpp_file_path, output_png_path, "settings_index.csv")
    
    # Record this land-cover type
    if lcname:
        lc_counts[lcname] += 1
    else:
        lc_counts["Unknown"] += 1

    print(f"Output saved to {output_yfit_file_path}")
    print(f"Output saved to {output_vpp_file_path}")

    # if "BE-Dor" in csv_file:
    #     exit()
# --- After loop: summarize results ---
print("\n=== Land-cover summary ===")
for lcname, count in lc_counts.items():
    print(f"{lcname:25s}: {count}")

