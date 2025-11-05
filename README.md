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
```bash
python run_csv_ts_HRVPP2.py
```

## License

This project is licensed under the terms of the [Apache License 2.0](LICENSE) file.

- **Precompiled wheels (TestPypi download)**  
  **timesat model** is **proprietary and closed-source**.  
  All rights reserved by Zhanzhang Cai(Lund University), Lars Eklundh(Lund University), and Per Jönsson(Malmö University).  
  Usage is subject to [PROPRIETARY-LICENSE.txt](./vendor/PROPRIETARY-LICENSE.txt).  
  Redistribution, modification, or reverse engineering of these libraries is strictly prohibited.
