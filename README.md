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
uv sync
```

This will:

- Create a `.venv` virtual environment  
- Install all required dependencies into it  

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## Project Architecture 

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
- If you are interested in obtaining access to the data for testing or research, **please contact us**.

> 📂 **Summary**  
> - Private datasets belong in `data/`  
> - Data is *not* public  
> - Contact us if you require access  
