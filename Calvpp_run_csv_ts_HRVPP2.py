import os
import glob
import Calvpp_ts_csv_functions_HRVPP2

# Define input and output directories

input_folder = 'FAPAR/V5/timeseries/50m/'
output_folder = 'output/FAPAR/'

#input_folder = 'GPP/V2/gpp_final/'
#output_folder = 'output/GPP/'
csvpath = os.path.join(output_folder, 'csv')
pngpath = os.path.join(output_folder, 'png')
# Create output folder if it doesn't exist
os.makedirs(csvpath, exist_ok=True)
os.makedirs(pngpath, exist_ok=True)

# Get all CSV files in the input folder
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

# Loop over all CSV files
for csv_file in csv_files:
    input_file_path = os.path.join(input_folder, csv_file)
    print(input_file_path)

    output_yfit_file_path = os.path.join(csvpath, os.path.splitext(csv_file)[0] + "_yfit.csv")
    output_vpp_file_path = os.path.join(csvpath, os.path.splitext(csv_file)[0] + "_vpp.csv")
    output_png_path = os.path.join(pngpath, os.path.splitext(csv_file)[0] + ".png")

    print(f"Processing {input_file_path}...")
    Calvpp_ts_csv_functions_HRVPP2._ts_run_(input_file_path, output_yfit_file_path, output_vpp_file_path, output_png_path)
    print(f"Output saved to {output_yfit_file_path}")
    print(f"Output saved to {output_vpp_file_path}")
    

