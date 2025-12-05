from streamlit import dataframe
from schwartz1997.calibration.SchwartzModel import SchwartzModel
import numpy as np
import pandas as pd


def C_tau(kappa, alpha_hat, a, m_star, sigma_1, sigma_2, sigma_3, rho_1, tau, eps=1e-4):
    # Use first-order Taylor if kappa or a are too small
    if abs(kappa) < eps:
        term1 = -(alpha_hat + sigma_1*sigma_2*rho_1) * tau**2 / 2
        term2 = -sigma_2 * tau**2 / 4
    else:
        term1 = 1/kappa**2 * (kappa * alpha_hat + sigma_1 * sigma_2 * rho_1) * (1 - np.exp(-kappa * tau) - kappa * tau)
        term2 = sigma_2 /(4 * kappa**3) * (4 * (1 - np.exp(-kappa * tau)) - (1 - np.exp(-2 * kappa * tau)) - 2 * kappa * tau)
    
    if abs(a) < eps:
        term3 = -m_star * tau**2 / 2
        term4 = -sigma_3**2 * tau**2 / 4
    else:
        term3 = m_star / a * (1 - np.exp(-a * tau) - a * tau)
        term4 = sigma_3**2 / (4 * a**3) * (4 * (1 - np.exp(-a * tau)) - (1 - np.exp(-2 * a * tau)) - 2 * a * tau)

    return term1 - term2 - term3 - term4

def calibrate_schwartz3(model: SchwartzModel, verbosity: bool = False, verbosity_cooldown: int = 10, 
                        save_results: bool = False) -> dict:
    """
    Calibrate the Schwartz 3-Factor model to commodity futures data.
    Args:
        commodity_ticker (str): Ticker symbol for the commodity.
        calibration_start_date (str): Start date for calibration in 'YYYY-MM-DD'.
        calibration_end_date (str): End date for calibration in 'YYYY-MM-DD'.
        verbosity (bool): If True, print detailed output during calibration.
        verbosity_cooldown (int): Frequency of verbose output.
        save_results (bool): If True, save calibration results to a JSON file.
    Returns:
        tuple: Calibrated parameters of the Schwartz 3-Factor model.
    """ 

    calibrated_params = model.calibrate_schwartz(
        verbosity=verbosity,
        verbosity_cooldown=verbosity_cooldown,
        save_results=save_results
    )

    latent_factors = model.get_latent_factors(calibrated_params=calibrated_params)

    return {
        "parameters": calibrated_params,
        "latent_factors": latent_factors
    }


def monte_carlo_simulation_schwartz3_states(
    calibrated_params: list,
    latent_factors: pd.DataFrame,
    num_simulations: int,
    num_steps: int = 252,
    delta_t: float = 1/252
) -> np.ndarray:
    """
    Return simulated latent factor paths X_t = (ln S_t, delta_t, r_t)
    Shape: (num_simulations, num_steps+1, 3)
    """

    kappa, alpha_hat, a, m_star, sigma_1, sigma_2, sigma_3, rho_12, rho_23 = calibrated_params
    dt = delta_t

    F = np.array([
        [1.0,      -dt,           dt],
        [0.0, 1.0 - kappa * dt,   0.0],
        [0.0,      0.0,      1.0 - a * dt]
    ])
    c = np.array([
        -0.5 * sigma_1**2 * dt,
        kappa * alpha_hat * dt,
        a * m_star * dt
    ])

    Q = np.array([
        [sigma_1**2 * dt,                 rho_12 * sigma_1 * sigma_2 * dt,              0.0],
        [rho_12 * sigma_1 * sigma_2 * dt, sigma_2**2 * dt,                              rho_23 * sigma_2 * sigma_3 * dt],
        [0.0,                             rho_23 * sigma_2 * sigma_3 * dt,              sigma_3**2 * dt]
    ])
    Q = 0.5 * (Q + Q.T)
    L = np.linalg.cholesky(Q)

    last_state = latent_factors.iloc[-1]
    x0 = np.array([
        float(last_state["lnS_t"]),
        float(last_state["convenience_yield"]),  # this is δ_t
        float(last_state["short_rate"])
    ])

    states = np.zeros((num_simulations, num_steps + 1, 3), dtype=float)
    states[:, 0, :] = x0

    for t in range(num_steps):
        z = np.random.normal(size=(num_simulations, 3))
        eps = z @ L.T
        x_t = states[:, t, :]          # (num_sim, 3)
        drift = (x_t @ F.T) + c        # (num_sim, 3)
        states[:, t+1, :] = drift + eps

    return states   # <--- latent factors for all paths


def plot_last_state_distribution(states: np.ndarray):
    """
    Plot distribution of ln S_T, δ_T, r_T across simulations
    using the last time step states[:, -1, :].
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # last time step across all simulations: shape (num_sim, 3)
    last = states[:, -1, :]
    lnS  = last[:, 0]
    delt = last[:, 1]
    r    = last[:, 2]

    df = pd.DataFrame(
        {"lnS_T": lnS, "convenience_yield_T": delt, "short_rate_T": r}
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.histplot(df["lnS_T"], bins=30, kde=True, ax=axes[0], color="blue")
    axes[0].set_title("Distribution of ln S_T (last step)")
    axes[0].set_xlabel("ln S_T")

    sns.histplot(df["convenience_yield_T"], bins=30, kde=True, ax=axes[1], color="green")
    axes[1].set_title("Distribution of δ_T (last step)")
    axes[1].set_xlabel("δ_T")

    sns.histplot(df["short_rate_T"], bins=30, kde=True, ax=axes[2], color="red")
    axes[2].set_title("Distribution of r_T (last step)")
    axes[2].set_xlabel("r_T")

    plt.tight_layout()
    plt.savefig("sim_last_state_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    # Example usage
    commodity_ticker = "KC"  # Crude Oil
    calibration_start_date = "2025-10-01"
    calibration_end_date = "2025-11-13"


    shwartz_model = SchwartzModel(
        commodity_ticker=commodity_ticker,
        calibration_start_date=calibration_start_date,
        calibration_end_date=calibration_end_date,
        vasicek_calibration_start_date=None,
        dt=1/252,
    )

    print("Starting calibration...")


    calibration_results = calibrate_schwartz3(
        shwartz_model,
        verbosity=True,
        verbosity_cooldown=10,
        save_results=False
    )

    calibrated_params = calibration_results["parameters"]
    latent_factors = calibration_results["latent_factors"]


    states = monte_carlo_simulation_schwartz3_states(
        calibrated_params,
        latent_factors,
        num_simulations=10000,
        num_steps=252,
        delta_t=1/252
    )

    print("Simulated latent states shape:", states.shape)
    print("Statesgt Parameters:", states)

    print("printing latent factor distribution summary...")
    plot_last_state_distribution(states)


if __name__ == "__main__":
    main()



# def monte_carlo_simulation_schwartz3_prices(
#     states: np.ndarray,

#     kappa: float,
#     a: float,
#     T_maturity: float,
#     t_idx: int = -1,
# ) -> np.ndarray:
#     """
#     Compute futures prices F_t(T) from simulated latent states.

#     states: array of shape (num_sim, num_steps+1, 3)
#         [:, :, 0] = ln S_t
#         [:, :, 1] = δ_t (convenience yield)
#         [:, :, 2] = r_t (short rate)

#     T_maturity: time-to-maturity T (in years, or same time units as kappa, a)
#     t_idx: which time index to use as 't' (default: last time step)

#     Returns:
#         F: array of shape (num_sim,) – one futures price per path
#     """
#     lnS   = states[:, t_idx, 0]
#     delta = states[:, t_idx, 1]
#     r     = states[:, t_idx, 2]

#     A_delta = (1.0 - np.exp(-kappa * T_maturity)) / kappa
#     A_r     = (1.0 - np.exp(-a * T_maturity)) / a
#     C_T     = C_tau(kappa, alpha_hat=0.0, a=a, m_star=0.0,
#                       sigma_1=sigma_1, sigma_2=sigma_2, sigma_3=sigma_3,
#                       rho_1=rho_12, tau=T_maturity)

#     lnF = lnS - delta * A_delta + r * A_r + C_T
#     F   = np.exp(lnF)

#     return F



