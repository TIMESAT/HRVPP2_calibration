# Project README

## Overview
Calibration workflow for the Copernicus High Resolution Vegetation Phenology and Productivity (HRVPP) Product: provides scripts and procedures to calibrate time-series and productivity estimates for high-resolution vegetation monitoring (using TIMESAT).

## Prerequisites
- Python 3.10–3.12 recommended
- `pip` available in your environment
- (Optional) A virtual environment tool such as `venv` or `conda`

## 1) Set up a Virtual Environment (recommended)

### Using `venv`
```bash
# Create and activate (Linux/macOS)
python -m venv .venv
source .venv/bin/activate

# Create and activate (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Using Conda (optional)
```bash
conda create -n timesat-env python=3.12 -y
conda activate timesat-env
```

## 2) Install TIMESAT
Install from TestPyPI, allowing dependencies to come from PyPI:

```bash
pip install timesat
```

## 3) Run the processing script
Step 1: calcualte GPP and VIs VPP
```bash
python Calvpp_run_csv_ts_HRVPP.py
```
Step 2: match GPP and VIs VPP
```bash
python match_GPP_VIs.py
```
```bash
python match_GPP_VIs_para.py
```
```bash
python match_GPP_VIs_para_NDVI_EVI2.py
```
Step 3: do statistics
```bash
python summarize_VI_gpp_vpp.py
```
```bash
python summarize_VI_gpp_vpp_para.py
```
Step 4: calculate VPP score
```bash
python calculate_vpp_score.py
```
Step 5: plot
```bash
python plot_vpp_points_by_triple_per_lc.py \
  --csv output/ALL_VIs_SUMMARY_BY_LC_merged_with_score.csv \
  --top-k 20 --min-per-group 3 \
  --seapar-col seapar --annotate-setting
```


## License

This project is licensed under the terms of the [Apache License 2.0](LICENSE) file.

- **Precompiled wheels (PyPI download)**  
  **timesat model** is **proprietary and closed-source**.  
  All rights reserved by Zhanzhang Cai(Lund University), Lars Eklundh(Lund University), and Per Jönsson(Malmö University).  
  Usage is subject to [PROPRIETARY-LICENSE.txt](./vendor/PROPRIETARY-LICENSE.txt).  
  Redistribution, modification, or reverse engineering of these libraries is strictly prohibited.
