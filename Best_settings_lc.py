import pandas as pd

# Path to your merged file with VPP_SCORE
csv_path = "output/ALL_VIs_SUMMARY_BY_LC_merged_with_score.csv"

# Load data
df = pd.read_csv(csv_path)

# Be robust to whitespace in column names
df.columns = [c.strip() for c in df.columns]

# Find the lc_id column name (in case of weird spaces / case)
lc_col = None
for c in df.columns:
    if c.strip().lower() == "lc_id":
        lc_col = c
        break

if lc_col is None:
    raise ValueError(f"No 'lc_id' column found. Available columns: {list(df.columns)}")

# Ensure VPP_SCORE is numeric
df["VPP_SCORE"] = pd.to_numeric(df["VPP_SCORE"], errors="coerce")
df = df.dropna(subset=["VPP_SCORE"])

# Sort by lc_id and then by VPP_SCORE descending
df_sorted = df.sort_values([lc_col, "VPP_SCORE"], ascending=[True, False])

# Take top 5 within each lc_id
top5_all = (
    df_sorted
    .groupby(lc_col, group_keys=False)
    .head(5)
    .copy()
)

# Optional: add rank within each land cover
top5_all["rank_within_lc"] = (
    top5_all.groupby(lc_col).cumcount() + 1
)


# Landcover names
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
    254: "Outside Area",
}

# Create a new descriptive column
top5_all["lc_name"] = top5_all["lc_id"].map(LC_PALETTE)

# Save all columns (+ rank)
out_file = "output/top5_settings_per_landcover_all_columns.csv"
top5_all.to_csv(out_file, index=False)

print("Saved:", out_file)
print(top5_all[[lc_col, "rank_within_lc", "VPP_SCORE"]].head(20))
