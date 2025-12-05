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

def future_prices(params, time_to_maturity, S_t, delta_t, r_t):
    kappa, alpha_hat, a, m_star, sigma_1, sigma_2, sigma_3, rho_12, rho_23 = params
    # The theoretical formula gives log-futures: ln F_t(T) = ln S_t - delta * A_tau + r_t * B_tau + C_tau
    # Ensure we compute lnF and return the futures price F = exp(lnF).
    lnS_t = S_t  # here S_t is expected to be ln(S) (see callers)
    A_tau = (1 - np.exp(-kappa * time_to_maturity)) / kappa
    B_tau = (1.0 - np.exp(-a * time_to_maturity)) / a
    C_tau = (
        (kappa * alpha_hat + sigma_1 * sigma_2 * rho_12)
        * (1 - np.exp(-kappa * time_to_maturity) - kappa * time_to_maturity)
        / kappa**2
        - (sigma_2**2 / (4 * kappa**3))
        * (4 * (1 - np.exp(-kappa * time_to_maturity)) - (1 - np.exp(-2 * kappa * time_to_maturity)) - 2 * kappa * time_to_maturity)
        - (m_star / a) * (1 - np.exp(-a * time_to_maturity) - a * time_to_maturity)
        - (sigma_3**2 / (4 * a**3))
        * (4 * (1 - np.exp(-a * time_to_maturity)) - (1 - np.exp(-2 * a * time_to_maturity)) - 2 * a * time_to_maturity)
    )

    lnF = lnS_t - delta_t * A_tau + r_t * B_tau + C_tau
    return float(np.exp(lnF))


tenor_to_business_days = {
    "1m": 21,
    "2m": 42,
    "3m": 63,
    "6m": 126,
    "9m": 189,
    "12m": 252,
    "15m": 315,
    "18m": 378,
    "24m": 504
}

def monte_carlo_simulation_schwartz3_prices(
    states: np.ndarray,
    calibrated_params: list,
    t_idx: int = -1,
) -> dict:
    """
    For each simulated path, compute a dictionary of futures prices across
    the predefined tenors in `tenor_to_business_days` using the provided
    `future_prices` helper.
    We start from the last time step by default (t_idx = -1). 
    Returns a list of results where each element is
    a dict: {'path': <int>, 'prices': {<tenor>: <price>, ...}}.
    """

    results = {}

    for path_idx, state in enumerate(states):
        lnS_t = float(state[t_idx, 0])
        delta_t = float(state[t_idx, 1])
        r_t = float(state[t_idx, 2])

        futures_prices = {}
        for tenor, days in tenor_to_business_days.items():
            time_to_maturity = days / 252
            # pass lnS_t (log-spot) to future_prices which returns F = exp(lnF)
            F_t_T = future_prices(calibrated_params, time_to_maturity, lnS_t, delta_t, r_t)
            futures_prices[tenor] = float(F_t_T)

        # store prices dict keyed by path index
        results[path_idx] = futures_prices

    return results


def main():
    # Example usage
    commodity_ticker = "KC"  # Coffee Futures
    calibration_start_date = "2025-11-01"
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
        verbosity_cooldown=5,
        save_results=False
    )

    calibrated_params = calibration_results["parameters"]
    latent_factors = calibration_results["latent_factors"]


    states = monte_carlo_simulation_schwartz3_states(
        calibrated_params,
        latent_factors,
        num_simulations=100,
        num_steps=25,
        delta_t=1/252
    )

    print("Simulated latent states shape:", states.shape)
    print("Calibrated Parameters:", states)

    print("printing latent factor distribution summary...")
    plot_last_state_distribution(states)

    future_prices = monte_carlo_simulation_schwartz3_prices(
        states= states,
        calibrated_params= calibrated_params,
        )
    
    print("Sample futures prices for first 5 paths:")
    for path_idx in range(5):
        print(f"Path {path_idx}: {future_prices[path_idx]}")


if __name__ == "__main__":
    main()
