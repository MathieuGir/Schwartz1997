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
def A_tau(a, tau):
    return (1 - np.exp(-a * tau)) / a

def B_tau(kappa, tau):
    return (1 - np.exp(-kappa * tau)) / kappa

def C_tau(kappa, alpha_hat, a, m_star, sigma_1, sigma_2, sigma_3, rho_1, tau):
    term1 = 1/kappa**2 * (kappa * alpha_hat + sigma_1 * sigma_2 * rho_1) * (1 - np.exp(-kappa * tau) - kappa * tau)
    term2 = sigma_2 /(4 * kappa**3) * (4 * (1 - np.exp(-kappa * tau)) - (1 - np.exp(-2 * kappa * tau)) - 2 * kappa * tau)
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

def get_observation_matrix(kappa, a, maturities):
    Z = np.column_stack([
        np.ones(len(maturities)),  # 1
        -A_tau(kappa, maturities),  # A_tau
        B_tau(a, maturities)  # B_tau
    ])
    return Z

def get_observation_offset(kappa, alpha_hat, a, m_star,
                           sigma_1, sigma_2, sigma_3, rho_12,
                           maturities):
    d = np.array([
        C_tau(kappa, alpha_hat, a, m_star,
             sigma_1, sigma_2, sigma_3, rho_12,
             tau) for tau in maturities
    ])
    return d

def build_schwartz_kalman_filter(params, log_futures, maturities, r_t, dt):
    # Unpack params

    kappa, alpha_hat, a, m_star, sigma_1, sigma_2, sigma_3, rho_12, rho_23 = params

    F = get_transition_matrix(kappa, a, dt)
    c = get_transition_offset(alpha_hat, m_star, sigma_1, kappa, a, dt)
    Q = get_transition_covariance(sigma_1, sigma_2, sigma_3, rho_12, rho_23, dt)

    y_list, Z_list, d_list = [], [], []

    for i in range(len(log_futures)):
        y_list.append([float(log_futures.iloc[i])])
        tau = float(maturities.iloc[i])

        Z_list.append([[1, -A_tau(a, tau), B_tau(kappa, tau)]])
        d_list.append([float(C_tau(kappa, alpha_hat, a, m_star, sigma_1, sigma_2, sigma_3, rho_12, tau))])

    y_obs = np.array(y_list)
    H = np.array([[0.01]])

    x0 = np.array([float(log_futures.iloc[0]), 0.1, float(r_t.iloc[0])])
    P0 = np.diag([0.5, 1, 0.001])

    kf = KalmanFilter(
        transition_matrices=F,
        transition_offsets=c,
        transition_covariance=Q,
        observation_matrices=Z_list,
        observation_offsets=d_list,
        observation_covariance=H,
        initial_state_mean=x0,
        initial_state_covariance=P0
    )

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

        self.data = load_calibration_data(commodity_ticker = self.commodity_ticker, 
                                               start_date=self.calibration_start_date, 
                                               end_date=self.calibration_end_date)
        
        self.log_futures = np.log(self.data['price'])
        self.maturities = self.data['time_to_maturity']
        self.r_t = self.data['r_t']
        self.dates = self.data.index

        # Placeholder for calibrated parameters
        self.calibrated_params = None


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

    
    def calibrate_schwartz(self, verbosity: bool = False, verbosity_cooldown: int = 10, save_results: bool = False):

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
            (1e-6, None),  # kappa
            (None, None),  # alpha_hat
            (a, a),  # a
            (m_star, m_star),  # m_star
            (1e-6, None),  # sigma_1
            (1e-6, None),  # sigma_2
            (sigma_3, sigma_3),  # sigma_3
            (-0.9999, 0.9999),  # rho_12
            (0, 0)   # rho_23
            ]
        
        calibration_start_time = time.time()

        result = minimize(
            fun=schwartz_loglik,
            x0=initial_params,
            args=(self.log_futures, self.maturities, self.r_t, self.dt, verbosity, verbosity_cooldown),
            bounds=bounds,
            method='L-BFGS-B'
        )

        calibration_end_time = time.time()

        print(f'\n Schwartz Model calibration completed in {calibration_end_time - calibration_start_time:,.2f} seconds.')


        if save_results:
            save_tmp_results(
                self.commodity_ticker,
                result.x,
                result.fun,
                self.vasicek_calibration_start_date,
                self.calibration_start_date,
                self.calibration_end_date,
                

            )

        
        print(f"\n========= Estimated Parameters for {self.commodity_ticker} =========")
        print(f'Sample size: {len(self.log_futures)} observations, from {self.calibration_start_date} to {self.calibration_end_date}')

        for section, entries in param_groups.items():
            print(f"{section:<25}{'Value':>25}")
            for name, idx in entries:
                print(f"{name:<25}{result.x[idx]:>25.6f}")
            print("-" * 50)

        print(f"{'Log likelihood':<25}{-result.fun:>25,.2f}")
        print("-" * 50)

        self.calibrated_params = result.x
        return result.x

    def get_latent_factors(self, verbosity: bool = False, verbosity_cooldown: int = 10, calibrated_params = None):
        if calibrated_params is None:
            if self.calibrated_params is None:
                self.calibrated_params = self.calibrate_schwartz(verbosity=verbosity, verbosity_cooldown=verbosity_cooldown)
    
            calibrated_params = self.calibrated_params

        kf, y_obs = build_schwartz_kalman_filter(
            calibrated_params,
            self.log_futures,
            self.maturities,
            self.r_t,
            self.dt
        )

        state_means, state_covariances = kf.filter(y_obs)
        latent_factors = pd.DataFrame(
            state_means,
            index=self.dates,
            columns=['lnS_t', 'convenience_yield', 'short_rate']
        )
        # latent_factors = latent_factors.groupby(latent_factors.index)
        return latent_factors

# Model = SchwartzModel(
#     commodity_ticker='KC', 
#     calibration_start_date='2025-11-04', 
#     calibration_end_date='2025-11-13', 
#     vasicek_calibration_start_date='2024-06-01'
#     )

# latent_factors = Model.get_latent_factors(verbosity=True, verbosity_cooldown=70)
# print(latent_factors)
