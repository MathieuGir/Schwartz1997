import json
from pathlib import Path

def save_tmp_results(
    commodity_ticker: str,
    params,
    neg_loglik,
    vasicek_start_date: str,
    calibration_start_date: str,
    calibration_end_date: str
):
    """
    Save Schwartz 3-factor calibration results to tmp/ as JSON.
    """

    # Prepare the result dictionary
    output = {
        "commodity": commodity_ticker,
        "parameters": {
            "kappa": float(params[0]),
            "alpha_hat": float(params[1]),
            "a": float(params[2]),
            "m_star": float(params[3]),
            "sigma_1": float(params[4]),
            "sigma_2": float(params[5]),
            "sigma_3": float(params[6]),
            "rho_12": float(params[7]),
            "rho_23": float(params[8]),
        },
        "log_likelihood": float(-neg_loglik),   # convert back to max loglik
        "sample": {
            "vasicek_start_date": vasicek_start_date,
            "calibration_start_date": calibration_start_date,
            "calibration_end_date": calibration_end_date,
        }
    }

    # Ensure tmp/ exists
    out_dir = Path("tmp")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filename includes ticker and date range
    filename = (
        f"{commodity_ticker}_"
        f"{vasicek_start_date}_"
        f"{calibration_start_date}_"
        f"{calibration_end_date}.json"
    )

    out_path = out_dir / filename

    # Write JSON file
    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\n[✔] Calibration saved to: {out_path}")
