from pathlib import Path
from datetime import date
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from schwartz1997.calibration.SchwartzModel import SchwartzModel
from schwartz1997.monteCarlo.monteCarlo import (
    calibrate_schwartz3,
    monte_carlo_simulation_schwartz3_states,
)

TMP_DIR = Path("tmp")

COMMODITY_NAME_TO_TICKER = {
    "Cocoa": "CC",
    "Cotton": "CT",
    "Coffee": "KC",
    "Sugar": "SB",
}

# ========== UTILITIES FOR SAVED CALIBRATIONS ==========

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
            print(f"Failed to load {f}: {e}")
    return calibrations


def _extract_params_from_json(data: dict):
    """Return (names, values) from a JSON calibration entry."""
    params = data.get("parameters", {})
    if isinstance(params, dict):
        names = list(params.keys())
        values = [params[k] for k in names]
    else:
        names = [
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
        values = list(params)
    return names, values


def show_parameter_table_and_chart(names, values):
    df = pd.DataFrame({"parameter": names, "value": values})
    col_table, col_chart = st.columns([2, 3])

    with col_table:
        st.markdown("#### Parameters (table)")
        st.table(df)

    with col_chart:
        st.markdown("#### Parameters (bar chart)")
        chart_df = df.set_index("parameter")
        st.bar_chart(chart_df)


# ========== CALIBRATION VIEW ==========

def show_calibration_details(data: dict):
    """Display a saved calibration json as summary + param plots."""
    st.subheader(f"Calibration for {data.get('commodity', 'Unknown')}")

    sample = data.get("sample", {})
    st.markdown(
        f"**Sample:** Vasicek start: `{sample.get('vasicek_start_date', 'N/A')}`, "
        f"Schwartz calibration: `{sample.get('calibration_start_date', 'N/A')}` → `{sample.get('calibration_end_date', 'N/A')}`"
    )

    loglik = data.get("log_likelihood", None)
    if loglik is not None:
        st.markdown(f"**Maximized log-likelihood:** `{loglik:.4f}`")

    names, values = _extract_params_from_json(data)
    if names:
        show_parameter_table_and_chart(names, values)


def run_new_calibration():
    """
    Returns:
        (params, latent_factors) or (None, None) if calibration not run.
    """
    st.header("Calibration")

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
        save_results = st.checkbox(
            "Save calibration to tmp/ as JSON",
            value=True,
        )

        if use_separate_vasicek_start:
            vasicek_start_date = st.date_input(
                "Vasicek calibration start date",
                value=date(2010, 1, 1),
                max_value=calibration_end_date,
            )
        else:
            vasicek_start_date = calibration_start_date

    submitted = st.button("Run calibration")

    # If button not pressed: nothing to do
    if not submitted:
        return None, None

    calib_start_str = calibration_start_date.isoformat()
    calib_end_str = calibration_end_date.isoformat()
    vasicek_start_str = vasicek_start_date.isoformat()

    st.write("Starting calibration...")

    model = SchwartzModel(
        commodity_ticker=commodity_ticker,
        calibration_start_date=calib_start_str,
        calibration_end_date=calib_end_str,
        vasicek_calibration_start_date=vasicek_start_str,
        dt=1 / 252,
    )

    calibration_results = calibrate_schwartz3(
        model,
        verbosity=verbosity,
        verbosity_cooldown=10,
        save_results=save_results,
    )
    params = calibration_results["parameters"]
    latent_factors = calibration_results["latent_factors"]

    # Persist for later Monte Carlo runs
    st.session_state["schwartz_params"] = params
    st.session_state["schwartz_latent_factors"] = latent_factors

    st.success("Calibration completed.")

    # Show parameter table + bar chart
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
    show_parameter_table_and_chart(param_names, list(params))

    # Plot latent factors
    st.markdown("### Latent Factors (Kalman filter)")
    lf = latent_factors.copy()
    lf.index.name = "time_index"
    lf_reset = lf.reset_index()

    col_l, col_r = st.columns(2)
    with col_l:
        lf_reset["Spot_S_t"] = np.exp(lf_reset["lnS_t"])
        fig_lnS = px.line(
            lf_reset,
            x="time_index",
            y="Spot_S_t",
            title="Spot Price S_t",
        )
        st.plotly_chart(fig_lnS, width="stretch")

        fig_cy = px.line(
            lf_reset,
            x="time_index",
            y="convenience_yield",
            title="Convenience Yield δ_t",
        )
        st.plotly_chart(fig_cy, width="stretch")

    with col_r:
        fig_r = px.line(
            lf_reset,
            x="time_index",
            y="short_rate",
            title="Short Rate r_t",
        )
        st.plotly_chart(fig_r, width="stretch")

        st.markdown("Preview of latent factor DataFrame")
        st.dataframe(lf.head())

    return params, latent_factors


# ========== MONTE CARLO VIEW ==========

def run_monte_carlo_section(calibrated_params, latent_factors):
    st.header("Monte Carlo Simulation")

    # Recover from session_state if not passed in
    if calibrated_params is None:
        calibrated_params = st.session_state.get("schwartz_params")
    if latent_factors is None:
        latent_factors = st.session_state.get("schwartz_latent_factors")

    if calibrated_params is None or latent_factors is None:
        st.info("Run a calibration first (above) to enable Monte Carlo simulation.")
        return

    col1, col2 = st.columns(2)
    with col1:
        num_sim = st.number_input(
            "Number of simulations",
            min_value=100,
            max_value=50000,
            value=5000,
            step=100,
        )
        num_steps = st.number_input(
            "Number of steps (horizon in days)",
            min_value=10,
            max_value=252 * 5,
            value=252,
            step=10,
        )
    with col2:
        delta_t = 1 / 252
        st.markdown(f"Time step Δt is fixed to `{delta_t}` (years).")

        # NEW: how many paths to display (1–10)
        n_plot = st.slider(
            "Number of paths to display",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="This only affects the chart, not the number of simulated paths.",
        )

    run_sim = st.button("Run Monte Carlo simulation")

    if not run_sim:
        return

    st.write("Running Monte Carlo simulation...")

    states = monte_carlo_simulation_schwartz3_states(
        calibrated_params=calibrated_params,
        latent_factors=latent_factors,
        num_simulations=int(num_sim),
        num_steps=int(num_steps),
        delta_t=delta_t,
    )

    st.success(f"Simulation completed. Shape of states: {states.shape}")

    # -------- Sample paths of S_t (spot), at most n_plot paths --------
    st.markdown("### Sample Paths of Spot Price S_t")

    n_plot = min(int(num_sim), int(n_plot))  # in case num_sim < n_plot
    lnS_paths = states[:n_plot, :, 0]        # (n_plot, num_steps+1)
    S_paths = np.exp(lnS_paths)             # convert to spot prices
    t_grid = np.arange(S_paths.shape[1])    # 0 .. num_steps

    df_paths = pd.DataFrame(
        S_paths.T,
        index=t_grid * delta_t,             # time in years
        columns=[f"path_{i+1}" for i in range(n_plot)],
    )
    df_paths.index.name = "time (years)"

    # Build a nicer Plotly line chart with legend and tight y‑axis
    df_melt = df_paths.reset_index().melt(
        id_vars="time (years)", var_name="path", value_name="S_t"
    )
    fig_paths = px.line(
        df_melt,
        x="time (years)",
        y="S_t",
        color="path",
        title=f"Sample Spot Price Paths (first {n_plot} of {int(num_sim)} simulations)",
    )

    # Set y‑axis range with a small margin
    y_min = df_melt["S_t"].min()
    y_max = df_melt["S_t"].max()
    margin = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    fig_paths.update_yaxes(range=[y_min - margin, y_max + margin])
    fig_paths.update_layout(legend_title_text="Path", hovermode="x unified")

    st.plotly_chart(fig_paths, width="stretch")

    # -------- Distribution at horizon --------
    st.markdown("### Distribution at Horizon T (last step)")

    last = states[:, -1, :]
    df_last = pd.DataFrame(
        {
            "lnS_T": last[:, 0],
            "S_T": np.exp(last[:, 0]),
            "convenience_yield_T": last[:, 1],
            "short_rate_T": last[:, 2],
        }
    )

    colh1, colh2, colh3 = st.columns(3)
    with colh1:
        fig_ST_hist = px.histogram(
            df_last, x="S_T", nbins=40, title="Spot Price S_T distribution"
        )
        st.plotly_chart(fig_ST_hist, width="stretch")
    with colh2:
        fig_cy_hist = px.histogram(
            df_last,
            x="convenience_yield_T",
            nbins=40,
            title="δ_T distribution",
        )
        st.plotly_chart(fig_cy_hist, width="stretch")
    with colh3:
        fig_r_hist = px.histogram(
            df_last, x="short_rate_T", nbins=40, title="r_T distribution"
        )
        st.plotly_chart(fig_r_hist, width="stretch")

    st.markdown("Preview of horizon distribution sample:")
    st.dataframe(df_last.head())


# ========== MAIN APP ==========

def main():
    st.set_page_config(
        page_title="Schwartz 3-Factor Calibration & Monte Carlo",
        layout="wide",
    )

    st.title("Schwartz 3-Factor (1997) – Calibration & Monte Carlo Dashboard")

    # Sidebar: saved calibrations
    st.sidebar.header("Saved Calibrations")
    calibrations = load_saved_calibrations()

    selected_data = None
    if calibrations:
        options = [c[0].name for c in calibrations]
        selected_label = st.sidebar.selectbox("Choose a saved calibration", options)
        selected_file, selected_data = next(
            (f, d) for (f, d) in calibrations if f.name == selected_label
        )
        st.sidebar.markdown(f"**Selected file:** `{selected_file.name}`")
    else:
        st.sidebar.info("No saved calibration found in `tmp/`.")

    st.subheader("Saved Calibration Summary")
    if selected_data is not None:
        show_calibration_details(selected_data)
    else:
        st.write("No saved calibration selected.")

    st.markdown("---")

    calibrated_params, latent_factors = run_new_calibration()
    run_monte_carlo_section(calibrated_params, latent_factors)


if __name__ == "__main__":
    main()