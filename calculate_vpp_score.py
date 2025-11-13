import glob
import os
import re
import pandas as pd

# --- user inputs ---
PATTERN = "output/*_SUMMARY__BY_LC.csv"
SETTINGS_LOOKUP_PATH = "settings_index.csv"   # keep your path

# --- helpers ---
def read_any_csv(path):
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            raise ValueError("single column after comma read; retrying as TSV")
        return df
    except Exception:
        try:
            df = pd.read_csv(path, sep="\t")
            if df.shape[1] == 1:
                raise ValueError("single column after TSV read; retrying with sep=None")
            return df
        except Exception:
            return pd.read_csv(path, sep=None, engine="python")

def infer_vi_from_filename(fname):
    base = os.path.basename(fname)
    return base.split("_SUMMARY__BY_LC.csv")[0]

def normalize_setting_text(s: pd.Series) -> pd.Series:
    """
    Normalize 'Setting' strings to improve join robustness:
    - cast to str
    - strip leading/trailing whitespace
    - collapse internal runs of whitespace to a single space
    - lowercase (use .str.casefold for safety)
    """
    s = s.astype(str).str.strip()
    # collapse multiple spaces/tabs to single space
    s = s.str.replace(r"\s+", " ", regex=True)
    # unify case
    s = s.str.casefold()
    return s

# --- load settings lookup (string key join) ---
settings_lookup = None
lookup_meta_cols = []
if SETTINGS_LOOKUP_PATH and os.path.exists(SETTINGS_LOOKUP_PATH):
    settings_lookup = read_any_csv(SETTINGS_LOOKUP_PATH)
    settings_lookup.columns = [c.strip() for c in settings_lookup.columns]

    # Identify/rename the join key to 'Setting'
    if "Setting" not in settings_lookup.columns:
        key_aliases = ["setting", "SettingID", "setting_id", "settings_id", "SettingIndex", "setting_index", "index"]
        found = next((c for c in key_aliases if c in settings_lookup.columns), None)
        if not found:
            raise ValueError(
                f"{SETTINGS_LOOKUP_PATH} must contain a 'Setting' column or a recognizable alias "
                f"(e.g., {', '.join(key_aliases)})."
            )
        settings_lookup = settings_lookup.rename(columns={found: "Setting"})

    # Build normalized join key
    settings_lookup["Setting_key"] = normalize_setting_text(settings_lookup["Setting"])

    # Ensure uniqueness of the key in the lookup
    dups = settings_lookup[settings_lookup["Setting_key"].duplicated(keep=False)]
    if not dups.empty:
        raise ValueError(
            "Duplicate Setting entries detected in the lookup after normalization. "
            "Please ensure unique rows per Setting.\n"
            f"{dups[['Setting']].to_string(index=False)}"
        )

    # Meta columns to insert after 'Setting' (everything except the join keys)
    lookup_meta_cols = [c for c in settings_lookup.columns if c not in ("Setting", "Setting_key")]

# --- read and prepare all VI files ---
frames = []
for path in glob.glob(PATTERN):
    vi = infer_vi_from_filename(path)
    df = read_any_csv(path)

    # Clean headers
    df.columns = [c.strip() for c in df.columns]

    # Checks
    if "Setting" not in df.columns:
        raise ValueError(f"'Setting' column not found in {path}")

    # Add VI column first
    df.insert(0, "VI", vi)

    # Normalized key for merge
    df["Setting_key"] = normalize_setting_text(df["Setting"])

    # Merge lookup (string join on normalized key)
    if settings_lookup is not None:
        df = df.merge(
            settings_lookup[["Setting_key", "Setting"] + lookup_meta_cols],
            on="Setting_key",
            how="left",
            suffixes=("", "_lkp"),
        )

        # Column order: VI, lc_id, Setting (original from df if present), [lookup meta...], then the rest
        leading = [c for c in ["VI", "lc_id", "Setting"] if c in df.columns]
        placed = set(leading + ["Setting_key"])
        # Insert meta columns right after Setting
        ordered = leading + lookup_meta_cols
        placed.update(lookup_meta_cols)

        remaining = [c for c in df.columns if c not in placed]
        df = df[ordered + remaining]

    else:
        # No lookup: keep a tidy order
        leading = [c for c in ["VI", "lc_id", "Setting"] if c in df.columns]
        others = [c for c in df.columns if c not in leading + ["Setting_key"]]
        df = df[leading + others]

    # Drop helper key
    if "Setting_key" in df.columns:
        df = df.drop(columns=["Setting_key"])

    frames.append(df)

# --- concatenate all VIs ---
if not frames:
    raise FileNotFoundError(f"No files matched pattern: {PATTERN}")

all_df = pd.concat(frames, ignore_index=True)

# Save
out_path = "output/ALL_VIs_SUMMARY_BY_LC_merged.csv"
all_df.to_csv(out_path, index=False)
print(f"Wrote: {out_path}")

# --- compute composite score ---
# metrics to use and their weights
score_config = {
    "SOSD_RMSE": {"weight": 0.2, "higher_is_better": False},
    "EOSD_RMSE": {"weight": 0.2, "higher_is_better": False},
    "LENGTH_RMSE": {"weight": 0.2, "higher_is_better": False},
    "AMPL_R2": {"weight": 0.1, "higher_is_better": True},
    "TPROD_R2": {"weight": 0.3, "higher_is_better": True},
}

# copy to avoid chained assignment warnings
df_scores = all_df.copy()

for metric, cfg in score_config.items():
    if metric not in df_scores.columns:
        raise ValueError(f"Column '{metric}' not found in merged dataset.")
    vals = df_scores[metric].astype(float)
    mmin, mmax = vals.min(), vals.max()
    if mmax == mmin:
        # avoid divide by zero
        df_scores[f"{metric}_score"] = 1.0
    else:
        if cfg["higher_is_better"]:
            df_scores[f"{metric}_score"] = (vals - mmin) / (mmax - mmin)
        else:
            df_scores[f"{metric}_score"] = 1 - (vals - mmin) / (mmax - mmin)

# Weighted composite
df_scores["VPP_SCORE"] = sum(
    cfg["weight"] * df_scores[f"{metric}_score"]
    for metric, cfg in score_config.items()
)

# Keep your original column order, add score at the end
all_df = df_scores
all_df.to_csv("output/ALL_VIs_SUMMARY_BY_LC_merged_with_score.csv", index=False)
print("Wrote: output/ALL_VIs_SUMMARY_BY_LC_merged_with_score.csv")
