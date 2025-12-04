import pandas as pd
import numpy as np
import os
import time
from datetime import date
from scipy.optimize import minimize
from pykalman import KalmanFilter
from schwartz1997.calibration.vasicekCalibration import calibrate_vasicek
from schwartz1997.helper import load_short_rate_data, load_calibration_data, save_tmp_results

# Map for clean parameter display
param_groups = {
    "SPOT / CONVENIENCE YIELD": [
        ("kappa",     0),
        ("alpha_hat", 1),
        ("sigma_1",   4),
        ("sigma_2",   5),
        ("rho_12",    7),
    ],
    "VASICEK PARAMETERS": [
        ("a",        2),
        ("m_star",   3),
        ("sigma_3",  6),
        ("rho_23",   8),
    ]
}

_likelihood_counter = 0
def reset_likelihood_counter():
    """Reset the global likelihood evaluation counter."""
    global _likelihood_counter
    _likelihood_counter = 0

def incr_likelihood_counter():
    """Increment and return the global likelihood evaluation counter."""
    global _likelihood_counter
    _likelihood_counter += 1
    return _likelihood_counter

# SCHWARTZ MODEL FUNCTIONS

def A_tau(a, tau, eps=1e-4):
    tau = np.asarray(tau)
    out = np.zeros_like(tau, dtype=float)
    for i, t in enumerate(tau):
        if abs(a) < eps:
            out[i] = t  # first-order Taylor
        else:
            out[i] = (1 - np.exp(-a * t)) / a
    return out

def B_tau(kappa, tau, eps=1e-4):
    tau = np.asarray(tau)
    out = np.zeros_like(tau, dtype=float)
    for i, t in enumerate(tau):
        if abs(kappa) < eps:
            out[i] = t  # first-order Taylor
        else:
            out[i] = (1 - np.exp(-kappa * t)) / kappa
    return out

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


def schwartz_3_futures_prices(A_tau, B_tau, C_tau, lnS_t, delta_t, r_t, tau):
    return lnS_t - delta_t * A_tau + r_t * B_tau + C_tau


# KALMAN FILTER

def get_transition_matrix(kappa, a, dt):
    F = np.array([
        [1, -dt, dt],
        [0, 1 - kappa * dt, 0],
        [0, 0, 1 - a * dt]
    ])
    return F

def get_transition_offset(alpha_hat, m_star, sigma_1, kappa, a, dt):
    c = np.array([
        -0.5 * sigma_1**2 * dt,
        kappa * alpha_hat * dt,
        a * m_star * dt
    ])
    return c

def get_transition_covariance(sigma_1, sigma_2, sigma_3, rho_12, rho_23, dt):
    Q = np.array([
        [sigma_1**2 * dt, rho_12 * sigma_1 * sigma_2 * dt, 0],
        [rho_12 * sigma_1 * sigma_2 * dt, sigma_2**2 * dt, rho_23 * sigma_2 * sigma_3 * dt],
        [0, rho_23 * sigma_2 * sigma_3 * dt, sigma_3**2 * dt]
    ])
    return Q


def _ensure_positive_definite(mat, init_eps=1e-8, max_tries=12):
    """
    Ensure matrix is (numerically) positive definite by adding jitter to the diagonal.
    Returns a copy of the matrix that is PD or raises a ValueError if unable.
    """
    M = np.array(mat, dtype=float).copy()
    eps = init_eps
    for i in range(max_tries):
        try:
            # Symmetrize to avoid numerical asymmetry
            M_sym = (M + M.T) / 2.0
            np.linalg.cholesky(M_sym)
            return M_sym
        except np.linalg.LinAlgError:
            M = M + eps * np.eye(M.shape[0])
            eps *= 10
    raise ValueError("Unable to make matrix positive definite")

def get_observation_matrix(kappa, a, maturities):
    maturities = np.asarray(maturities).ravel()
    Z = np.column_stack([
        np.ones(len(maturities)),
        [-val for val in A_tau(kappa, maturities)],
        [val for val in B_tau(a, maturities)]
    ])
    print(f"Z_t shape: {Z.shape}\n{Z}\n")
    return Z

def get_observation_offset(kappa, alpha_hat, a, m_star,
                           sigma_1, sigma_2, sigma_3, rho_12,
                           maturities):
    maturities = np.asarray(maturities).ravel()
    d = np.array([
        C_tau(kappa, alpha_hat, a, m_star, sigma_1, sigma_2, sigma_3, rho_12, tau)
        for tau in maturities
    ]).reshape(-1, 1)
    print(f"d_t shape: {d.shape}\n{d}\n")
    return d

def get_observation_covariance(sigma_1, sigma_2, sigma_3, rho_12, rho_23, dt, obs_dim=None):
    """
    Build a (possibly time-invariant) observation noise covariance matrix.
    For simplicity we use a small diagonal covariance and allow the caller
    to specify the observation dimension `obs_dim`.
    """
    if obs_dim is None:
        # fallback to scalar noise if dimension unknown
        return np.array([[0.01]])
    base = 0.01
    H = base * np.eye(obs_dim)
    # ensure PD with adaptive jitter
    H = _ensure_positive_definite(H)
    return H

def build_schwartz_kalman_filter(params, log_futures_list, maturities_list, r_t_list, dt):
    # Unpack params
    kappa, alpha_hat, a, m_star, sigma_1, sigma_2, sigma_3, rho_12, rho_23 = params
    # print(f"Params: kappa={kappa}, alpha_hat={alpha_hat}, a={a}, m_star={m_star}, sigma_1={sigma_1}, sigma_2={sigma_2}, sigma_3={sigma_3}, rho_12={rho_12}, rho_23={rho_23}")

    # Transition matrices
    F = get_transition_matrix(kappa, a, dt)
    c = get_transition_offset(alpha_hat, m_star, sigma_1, kappa, a, dt)
    Q = get_transition_covariance(sigma_1, sigma_2, sigma_3, rho_12, rho_23, dt)
    # Ensure Q is positive definite (adaptive jitter)
    try:
        Q = _ensure_positive_definite(Q)
    except ValueError:
        # fallback: add large diagonal to keep optimizer from exploring this region
        Q = Q + 1e-6 * np.eye(3)

    y_obs_list, Z_list, d_list = [], [], []

    # determine maximum number of contracts observed on any date
    max_obs = max(len(np.asarray(t).ravel()) for t in maturities_list)

    for i, (y_t, tau_t, r_t_t) in enumerate(zip(log_futures_list, maturities_list, r_t_list)):
        # print(f"\n--- Date index {i} ---")
        # print(f"Original y_t: {y_t}")
        # print(f"Original tau_t: {tau_t}")
        # print(f"Original r_t_t: {r_t_t}")

        # Ensure correct shapes
        y_t = np.asarray(y_t).reshape(-1, 1)
        tau_t = np.asarray(tau_t).ravel()
        r_t_t = np.asarray(r_t_t).ravel()

        # print(f"Reshaped y_t: {y_t.shape}")
        # print(f"Reshaped tau_t: {tau_t.shape}")
        # print(f"Reshaped r_t_t: {r_t_t.shape}")

        # Build padded observation vector (obs_dim = max_obs)
        obs_dim = max_obs
        y_pad = np.full((obs_dim, ), np.nan, dtype=float)
        y_flat = np.asarray(y_t).ravel()
        y_pad[: len(y_flat)] = y_flat
        y_obs_list.append(y_pad)

        # Observation matrix Z_t (obs_dim x state_dim). For missing observations
        # we put zeros in the corresponding rows — the Kalman update will skip
        # observation entries that are NaN in the observation vector.
        Z_t = np.zeros((obs_dim, 3), dtype=float)
        Z_t[: len(tau_t), 0] = 1.0
        Z_t[: len(tau_t), 1] = -A_tau(kappa, tau_t)
        Z_t[: len(tau_t), 2] = B_tau(a, tau_t)
        Z_list.append(Z_t)

        # Observation offset d_t (vector of length obs_dim)
        d_t = np.zeros((obs_dim, ), dtype=float)
        d_comp = np.array([
            C_tau(kappa, alpha_hat, a, m_star,
                  sigma_1, sigma_2, sigma_3, rho_12, tau)
            for tau in tau_t
        ]).ravel()
        d_t[: len(d_comp)] = d_comp
        d_list.append(d_t)

        # print(f"d_t is of shape {d_t.shape} and values:\n{d_t}")
        # print(f"Z_t is of shape {Z_t.shape} and values:\n{Z_t}")
        # print(f"y_t appended: {y_t.shape}")

    # Observation noise covariance (time-invariant, dimension = max_obs)
    H = get_observation_covariance(sigma_1, sigma_2, sigma_3, rho_12, rho_23, dt, obs_dim=max_obs)
    # ensure H is PD (should be already handled in get_observation_covariance)
    try:
        H = _ensure_positive_definite(H)
    except ValueError:
        H = H + 1e-6 * np.eye(H.shape[0])

    # Initial state
    # find a sensible initial lnS_t (first non-nan observation)
    first_y = None
    for v in y_obs_list:
        v = np.asarray(v).ravel()
        non_nans = v[~np.isnan(v)]
        if non_nans.size > 0:
            first_y = non_nans[0]
            break
    if first_y is None:
        first_y = 0.0

    # initial state: lnS_t (from first observation), convenience yield, short rate
    x0 = np.array([first_y, 0.1, np.asarray(r_t_list[0]).ravel()[0]])
    P0 = np.diag([0.5, 1, 0.001])

    # Convert lists into arrays with the shapes pykalman expects:
    # observation_matrices: (n_timesteps, obs_dim, state_dim)
    # observation_offsets: (n_timesteps, obs_dim)
    observation_matrices = np.stack(Z_list, axis=0)
    observation_offsets = np.stack(d_list, axis=0)

    kf = KalmanFilter(
        transition_matrices=F,
        transition_offsets=c,
        transition_covariance=Q,
        observation_matrices=observation_matrices,
        observation_offsets=observation_offsets,
        observation_covariance=H,
        initial_state_mean=x0,
        initial_state_covariance=P0
    )

    # Stack observations into shape (n_timesteps, obs_dim) with NaNs for missing
    y_obs = np.vstack(y_obs_list)
    return kf, y_obs


def schwartz_loglik(params, log_futures, maturities, r_t, dt, verbosity=False, cooldown=10):
    global likelihood_counter
    likelihood_counter = incr_likelihood_counter()

    try:
        kf, y_obs = build_schwartz_kalman_filter(params, log_futures, maturities, r_t, dt)
        log_lik = kf.loglikelihood(y_obs)

        if verbosity and (likelihood_counter % cooldown == 0):
            print("CURRENT LIKELIHOOD EVALUATION".center(50, '='))
            for section, entries in param_groups.items():
                print(f"{section:<25}{'Value':>25}")
                for name, idx in entries:
                    print(f"{name:<25}{params[idx]:>25.6f}")
                print("-" * 50)
            print(f"{'Log likelihood':<25}{log_lik:>25,.2f} \n")

    except Exception as e:
        print("Kalman error:", e)
        return np.inf

    return -log_lik



# SCHWARTZ MODEL CLASS

class SchwartzModel:
    def __init__(
            self, 
            commodity_ticker: str, 
            calibration_start_date: str, 
            calibration_end_date: str, 
            vasicek_calibration_start_date: str = None, 
            dt=1/252
            ):
        self.commodity_ticker = commodity_ticker
        self.calibration_start_date = calibration_start_date
        self.calibration_end_date = calibration_end_date
        self.vasicek_calibration_start_date = vasicek_calibration_start_date if vasicek_calibration_start_date is not None else calibration_start_date #allows for Vasicek calibration on longer period than Schwartz (more computational heavy)        
        self.dt = dt

        self.data = load_calibration_data(
            commodity_ticker=self.commodity_ticker, 
            start_date=self.calibration_start_date, 
            end_date=self.calibration_end_date
            )
        
        self.data['log_price'] = np.log(self.data['price'])

        # Group by date to handle multiple contracts per date
        grouped = self.data.groupby(level=0)  # level=0 = 'date'

        # Store as lists of arrays, one per date
        self.log_futures_list = [group['log_price'].values.reshape(-1, 1) for _, group in grouped]
        self.maturities_list = [group['time_to_maturity'].values for _, group in grouped]
        self.r_t_list = [group['r_t'].values for _, group in grouped]

        # Placeholder for calibrated parameters
        self.calibrated_params: list = None

        print("SchwartzModel initialized.")

    def vasicek_calibration(self, start_date:str = None, end_date:str = None, verbosity: bool = False):
        if start_date is None:
            start_date = self.vasicek_calibration_start_date
        if end_date is None:
            end_date = self.calibration_end_date
        
        # print(f"Calibrating Vasicek model for short rates from {self.vasicek_calibration_start_date} to {self.end_date}...")
        calibration_rates = load_short_rate_data(file_path= 'data/DTB3.csv',start_date = start_date, end_date = end_date)
        vasicek_params = calibrate_vasicek(
            calibration_rates=calibration_rates,
            verbosity=verbosity
        )
        self.vasicek_params = vasicek_params
        return vasicek_params
    
    def calibrate_schwartz(self, verbosity: bool = False, verbosity_cooldown: int = 10, 
                           save_results: bool = False) -> list:

        if not hasattr(self, 'vasicek_params'):
            self.vasicek_calibration(verbosity=verbosity)
        
        a, m_star, sigma_3 = self.vasicek_params

        initial_params = [
            1,    # kappa
            0.25,    # alpha_hat
            a,    # a
            m_star,    # m_star
            0.3,    # sigma_1
            0.3,    # sigma_2
            sigma_3,   # sigma_3
            0.3,    # rho_12
            0     # rho_23
            ]
        
        bounds = [
            (1e-6, 1e3),  # kappa
            (-0.5, 1e3),  # alpha_hat
            (a, a),  # a
            (m_star, m_star),  # m_star
            (1e-6, 5.0),  # sigma_1
            (1e-6, 5.0),  # sigma_2
            (sigma_3, sigma_3),  # sigma_3
            (-0.9999, 0.9999),  # rho_12
            (0, 0)   # rho_23
            ]
        
        calibration_start_time = time.time()
        
        #set max iteration for optimizer
        result = minimize(
            fun=schwartz_loglik,
            x0=initial_params,
            args=(self.log_futures_list, self.maturities_list, self.r_t_list, self.dt, verbosity, verbosity_cooldown),
            bounds=bounds,
            method='L-BFGS-B'
        )


        calibration_end_time = time.time()

        print(f'\n Schwartz Model calibration completed in {calibration_end_time - calibration_start_time:,.2f} seconds.')
        # print("Optimization result:", result)
        # print("Number of iterations:", result.nit)
        # print("Final parameters:", result.x)


        if save_results:
            save_tmp_results(
                self.commodity_ticker,
                result.x,
                result.fun,
                self.vasicek_calibration_start_date,
                self.calibration_start_date,
                self.calibration_end_date
            )

        
        print(f"\n========= Estimated Parameters for {self.commodity_ticker} =========")
        print(f'Sample size: {len(self.log_futures_list)} observations, from {self.calibration_start_date} to {self.calibration_end_date}')

        for section, entries in param_groups.items():
            print(f"{section:<25}{'Value':>25}")
            for name, idx in entries:
                print(f"{name:<25}{result.x[idx]:>25.6f}")
            print("-" * 50)

        print(f"{'Log likelihood':<25}{-result.fun:>25,.2f}")
        print("-" * 50)

        self.calibrated_params = result.x
        return result.x.tolist()


    def get_latent_factors(self, verbosity: bool = False, verbosity_cooldown: int = 10, calibrated_params = None) -> pd.DataFrame:
        if calibrated_params is None:
            if self.calibrated_params is None:
                self.calibrated_params = self.calibrate_schwartz(verbosity=verbosity, verbosity_cooldown=verbosity_cooldown)
    
            calibrated_params = self.calibrated_params

        kf, y_obs = build_schwartz_kalman_filter(
            calibrated_params,
            self.log_futures_list,
            self.maturities_list,
            self.r_t_list,
            self.dt
        )
        # Run the Kalman filter to get filtered state means
        state_means, state_covariances = kf.filter(y_obs)

        # Build index (dates) matching the per-day observations
        if hasattr(self, 'dates') and self.dates is not None:
            dates = list(self.dates)
        else:
            try:
                # extract unique dates from the first level of the MultiIndex (or Index)
                dates = list(self.data.index.get_level_values(0).unique())
            except Exception:
                dates = None

        # If dates length does not match number of time steps, fallback to integer index
        n_steps = state_means.shape[0]
        if dates is None or len(dates) != n_steps:
            if dates is not None:
                print(f"Warning: date index length ({len(dates)}) does not match Kalman time steps ({n_steps}); using integer index.")
            dates = list(range(n_steps))

        latent_factors = pd.DataFrame(
            state_means,
            index=dates,
            columns=['lnS_t', 'convenience_yield', 'short_rate']
        )
        # latent_factors = latent_factors.groupby(latent_factors.index)
        return latent_factors
