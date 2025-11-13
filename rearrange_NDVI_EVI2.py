import os
import pandas as pd

# Input directory containing original CSVs
base_dir = r"C:\Users\Zhanzhang.Cai\Documents\GitHub\HRVPP2_calibration\VI_lc\NDVI_EVI2\csv_lc"

# Output directories
ndvi_dir = r"C:\Users\Zhanzhang.Cai\Documents\GitHub\HRVPP2_calibration\VI_lc\NDVI\csv_lc"
evi2_dir = r"C:\Users\Zhanzhang.Cai\Documents\GitHub\HRVPP2_calibration\VI_lc\EVI2\csv_lc"

# Make sure output folders exist
os.makedirs(ndvi_dir, exist_ok=True)
os.makedirs(evi2_dir, exist_ok=True)

# Loop through each CSV file in the base directory
for file in os.listdir(base_dir):
    if not file.lower().endswith(".csv"):
        continue

    file_path = os.path.join(base_dir, file)

    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Identify possible time column (t or date)
        time_cols = [c for c in df.columns if c.lower() in ["t", "date", "time", "datetime"]]
        if len(time_cols) == 0:
            print(f"⚠️ Skipping {file} — no time column found.")
            continue
        time_col = time_cols[0]

        # Identify NDVI and EVI2 columns (case-insensitive match)
        ndvi_cols = [c for c in df.columns if "ndvi" in c.lower()]
        evi2_cols = [c for c in df.columns if "evi2" in c.lower()]

        # --- NDVI subset ---
        if ndvi_cols:
            ndvi_df = df[[time_col] + ndvi_cols]
            ndvi_filename = os.path.splitext(file)[0] + "_NDVI.csv"
            ndvi_path = os.path.join(ndvi_dir, ndvi_filename)
            ndvi_df.to_csv(ndvi_path, index=False)
            print(f"✅ Saved NDVI file → {ndvi_path}")

        # --- EVI2 subset ---
        if evi2_cols:
            evi2_df = df[[time_col] + evi2_cols]
            evi2_filename = os.path.splitext(file)[0] + "_EVI2.csv"
            evi2_path = os.path.join(evi2_dir, evi2_filename)
            evi2_df.to_csv(evi2_path, index=False)
            print(f"✅ Saved EVI2 file → {evi2_path}")

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")

print("\n🎯 Finished separating NDVI and EVI2 CSV files.")
