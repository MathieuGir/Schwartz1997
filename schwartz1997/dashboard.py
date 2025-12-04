import json
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

# from schwartz1997.calibration.calibrationSchwartz3 import calibrate_schwartz3


TMP_DIR = Path("tmp")

COMMODITY_NAME_TO_TICKER = {
    "Cocoa": "CC",
    "Cotton": "CT",
    "Coffee": "KC",
    "Sugar": "SB",
}


def load_saved_calibrations():
    """Return a list of (Path, dict) for all JSON files in tmp/."""
    if not TMP_DIR.exists():
        return []

    files = sorted(TMP_DIR.glob("*.json"))
    calibrations = []
    for f in files:
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
            calibrations.append((f, data))
        except Exception as e:
            # You may want to log this somewhere
            print(f"Failed to load {f}: {e}")
    return calibrations


def show_calibration_details(data: dict):
    """Display a single calibration JSON as a nice dashboard."""
    st.subheader(f"Calibration for {data.get('commodity', 'Unknown')}")

    sample = data.get("sample", {})
    st.markdown(
        f"**Sample:** Vasicek start: `{sample.get('vasicek_start_date', 'N/A')}`, "
        f"Schwartz calibration: `{sample.get('calibration_start_date', 'N/A')}` → `{sample.get('calibration_end_date', 'N/A')}`"
    )

    loglik = data.get("log_likelihood", None)
    if loglik is not None:
        st.markdown(f"**Maximized log-likelihood:** `{loglik:.4f}`")

    params = data.get("parameters", {})
    if params:
        st.markdown("### Parameters")
        df = pd.DataFrame(
            {
                "parameter": list(params.keys()),
                "value": list(params.values()),
            }
        )
        st.table(df)


def run_new_calibration():
    st.header("Run a New Calibration")

    col1, col2 = st.columns(2)

    with col1:
        commodity_name = st.selectbox(
            "Commodity",
            options=list(COMMODITY_NAME_TO_TICKER.keys()),
            index=0,
        )
        commodity_ticker = COMMODITY_NAME_TO_TICKER[commodity_name]

        calibration_start_date = st.date_input(
            "Calibration start date", value=date(2015, 1, 1)
        )
        calibration_end_date = st.date_input(
            "Calibration end date", value=date.today()
        )

    with col2:
        use_separate_vasicek_start = st.checkbox(
            "Use different Vasicek calibration start date", value=False
        )
        verbosity = st.checkbox("Verbose output in console", value=False)

        # ⬇️ Move the conditional inside the same `with col2:` block
        if use_separate_vasicek_start:
            vasicek_start_date = st.date_input(
                "Vasicek calibration start date", value=date(2010, 1, 1)
            )
        else:
            vasicek_start_date = calibration_start_date

    save_results = st.checkbox(
        "Save calibration to tmp/ as JSON", value=True,
        help="If checked, results are saved as a JSON file in the tmp/ folder."
    )

    submitted = st.button("Run calibration")

    if not submitted:
        return

    # Convert dates to ISO strings for your calibration function
    calib_start_str = calibration_start_date.isoformat()
    calib_end_str = calibration_end_date.isoformat()
    vasicek_start_str = vasicek_start_date.isoformat() if vasicek_start_date else calib_start_str

    st.write("🚀 Starting calibration…")

    # Run calibration

    params = calibrate_schwartz3(
        commodity_ticker=commodity_ticker,
        calibration_start_date=calib_start_str,
        vasicek_calibration_start_date=vasicek_start_str,
        calibration_end_date=calib_end_str,
        vasicek_estimates=None,  # let your function estimate Vasicek if needed
        verbosity=verbosity,
        verbosity_cooldown=10,
        save_results=save_results,
    )

    st.success("Calibration completed.")


    st.markdown("### Estimated Parameters (this run)")
    param_names = [
        "kappa",
        "alpha_hat",
        "a",
        "m_star",
        "sigma_1",
        "sigma_2",
        "sigma_3",
        "rho_12",
        "rho_23",
    ]
    df = pd.DataFrame({"parameter": param_names, "value": np.array(params)})
    st.table(df)
    st.info("If `save_results` was checked, this calibration is now also available in the sidebar list.")


def main():
    st.set_page_config(
        page_title="Schwartz 3-Factor Calibration Dashboard",
        layout="wide",
    )

    st.title("Schwartz 3-Factor (1997) Calibration Dashboard")

    # --- Sidebar: list saved calibrations ---
    st.sidebar.header("Saved Calibrations")
    calibrations = load_saved_calibrations()

    selected_data = None
    if calibrations:
        # Use file names as labels
        options = [c[0].name for c in calibrations]
        selected_label = st.sidebar.selectbox("Choose a calibration", options)

        selected_file, selected_data = next(
            (f, d) for (f, d) in calibrations if f.name == selected_label
        )
        st.sidebar.markdown(f"**Selected file:** `{selected_file.name}`")
    else:
        st.sidebar.info("No saved calibration found in `tmp/`.")

    # Layout: left = details / current selection, right = run new calibration
    col_left, col_right = st.columns([2, 3])

    with col_left:
        st.header("Selected Calibration")
        if selected_data is not None:
            show_calibration_details(selected_data)
        else:
            st.write("No calibration selected or none available yet.")

    with col_right:
        run_new_calibration()


if __name__ == "__main__":
    main()
