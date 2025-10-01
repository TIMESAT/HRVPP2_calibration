# Project README

## Overview
This repository provides a workflow to install **TIMESAT** and then run the processing script `run_csv_ts_HRVPP2.py`.

## Prerequisites
- Python 3.9–3.12 recommended
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
conda create -n timesat-env python=3.11 -y
conda activate timesat-env
```

## 2) Install TIMESAT
Install from TestPyPI, allowing dependencies to come from PyPI:

```bash
python -m pip install --upgrade pip
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple timesat==4.1.7.dev0
```

### Verify the installation and native extension
```bash
python -c "import timesat, timesat._timesat as _; print('timesat', timesat.__version__, 'OK')"
```
Expected output includes the version and `OK`, e.g.:
```
timesat 4.1.7.dev0 OK
```

## 3) Run the processing script

Ensure you are in the repository root where `run_csv_ts_HRVPP2.py` resides. If the script depends on local modules, keep the current working directory at the project root.

### Basic usage
```bash
python run_csv_ts_HRVPP2.py --input <path_to_input_csv> --output <path_to_output_folder>
```

### Example
```bash
python run_csv_ts_HRVPP2.py --input data/example_inputs/series.csv --output results/
```

> Replace paths with your actual input data and desired output location. If the script accepts additional options (e.g., date columns, separators, site IDs, parameters), include them as needed, for example:
```bash
python run_csv_ts_HRVPP2.py   --input data/series.csv   --output results/   --date-col date   --value-col value   --site-col site_id
```

## 4) Troubleshooting

- **`ModuleNotFoundError: No module named 'timesat._timesat'`**  
  Ensure the install succeeded and you are using the same Python interpreter/venv to run the script. Re-run the verification command above.

- **Build/Compiler errors on installation**  
  Make sure you are on a supported Python version and have platform build tools installed (e.g., Xcode Command Line Tools on macOS, Build Tools for Visual Studio on Windows, or `build-essential` on Linux).

- **`pip` cannot find TIMESAT**  
  Confirm you included the TestPyPI index and PyPI fallback exactly as shown:
  ```bash
  python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple timesat==4.1.7.dev0
  ```

## 5) Reproducible environment (optional)
Freeze your environment for later reuse:
```bash
python -m pip freeze > requirements.txt
```
Recreate:
```bash
python -m pip install -r requirements.txt
```

## Repository Structure (suggested)
```
.
├── run_csv_ts_HRVPP2.py
├── data/
│   └── example_inputs/
├── results/
└── README.md
```

## License
Add your project license information here.

## Acknowledgements
- TIMESAT authors and contributors.
