import os
import ts_csv_functions_FAPAR_ST

# Define input and output directories
input_folder = 'FAPAR/V5/timeseries/50m/'
output_folder = 'output/FAPAR/'

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
    output_file_path = os.path.join(csvpath, csv_file)
    output_png_path = os.path.join(pngpath, os.path.splitext(csv_file)[0] + ".png")

    if 1:#input_file_path == '/Users/zhanzhangcai/Library/CloudStorage/OneDrive-LundUniversity/HR-VPP2/FAPAR/flux_cal/V5/timeseries/50m/FAPAR_2017_01_01_2024_12_31_ES-LJu.csv':
        print(f"Processing {input_file_path}...")
        ts_csv_functions_FAPAR_ST._ts_run_(input_file_path, output_file_path, output_png_path)
        print(f"Output saved to {output_file_path}")
    

