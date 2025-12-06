# Calibration of the Schwartz 3-Factor (1997) Model

This project contains the setup required to run the calibration framework for the Schwartz 3-factor (1997) commodity model.  
Below are the installation instructions and the data access policy for the project.

---

## Installation

Ensure you have **Python >=3.10 and <3.13** installed.  
The project uses **[UV](https://docs.astral.sh/uv/)** for dependency and environment management.

### 1. Install `uv`

If you haven't installed it yet:

```bash
pip install uv
```

### 2. Install project dependencies

From the root of the project (where `pyproject.toml` is located):

```bash
```
.
├── data/                         # Private market data (CSV files, not included)
│
├── schwartz1997/                 # Main Python package
│   ├── __init__.py
│   ├── dashboard.py              # Streamlit dashboard / demo UI
│   ├── calibration/              # Core calibration logic
│   │   ├── __init__.py
│   │   ├── calibrationSchwartz3.py
│   │   ├── SchwartzModel.py
│   │   └── vasicekCalibration.py
│   ├── helper/                   # Utility and data-loading functions
│   │   ├── __init__.py
│   │   ├── importdata.py
│   │   └── savetmp.py
│   └── monteCarlo/               # Monte Carlo simulation utilities
│       └── monteCarlo.py
│
├── notebooks/                     # Analysis notebooks (presentation, examples)
│   └── Model presentation.ipynb
|
├── README.md                     # Documentation (this file)
├── pyproject.toml                # Project config (UV + hatchling)
├── uv.lock                       # UV lockfile
└── .venv/                        # Virtual environment (created by uv sync)
```

The project is structured as follows:

```
.
├── data/                         # Private market data (not included in repo)
│
├── schwartz1997/                 # Main Python package
│   ├── __init__.py
│   │
│   ├── calibration/              # Core calibration logic
│   │   ├── __init__.py
│   │   ├── CalibrationSchwartz.py
│   │   └── Vasicekcalibration.py
│   │
│   ├── helper/                   # Utility and data-loading functions
│   │   ├── __init__.py
│   │   └── import_data.py
│   │
│   └── __init__.py               # Package initializer
│
├── README.md                     # Documentation (this file)
├── pyproject.toml                # Project config (UV + hatchling)
├── uv.lock                       # UV lockfile
└── .venv/                        # Virtual environment (created by uv sync)
```

Notes:

- All Python code is neatly grouped inside the schwartz1997/ package.

- Calibration logic lives in the calibration/ subpackage.

- Utility and helper functions live in helper/.

- Private data remains in the data/ folder and is not part of the distributed package

## Data Access

This project relies on **market data that is not publicly shareable** due to confidentiality and licensing restrictions.

- Expected input files must be placed in the **`data/`** folder at the project root.
- No real datasets are included in the repository.
