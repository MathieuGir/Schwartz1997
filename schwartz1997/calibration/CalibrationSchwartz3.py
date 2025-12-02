import pandas as pd
import numpy as np
import os
import time
from datetime import date
from scipy.optimize import minimize
from pykalman import KalmanFilter
from schwartz1997.calibration.Vasicekcalibration import calibrate_vasicek
from schwartz1997.helper.import_data import load_short_rate_data, load_calibration_data


calibration_start_date = '2025-11-01'
calibration_end_date = '2025-11-13'

calibration_ticker = 'KC'  # Commodity ticker for calibration
vasicek_calibration_start_date = '2022-04-01'


likelihood_counter = 0
verbosity_cooldown = 5

##### Schwartz 3 

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

# Transiiton matrix, offset, covariance for Kalman filter
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


def negative_log_likelihood_schwartz3(params, log_futures, maturities, r_t, dt, verbosity=False, verbosity_cooldown=10):
    # Unpack parameters
    (kappa, alpha_hat, a, m_star,
     sigma_1, sigma_2, sigma_3,
     rho_12, rho_23) = params
    global likelihood_counter
    likelihood_counter += 1

    # Transition components
    F = get_transition_matrix(kappa, a, dt)
    c = get_transition_offset(alpha_hat, m_star, sigma_1, kappa, a, dt)
    Q = get_transition_covariance(sigma_1, sigma_2, sigma_3, rho_12, rho_23, dt)

    # Prepare lists for each contract
    y_list = []     # List of T×1 observations
    Z_list = []     # List of 1×3 matrices
    d_list = []     # List of T×1 offsets

    # Iterate over each contract
    for i in range(len(log_futures)):
        # Scalar log futures price
        y_t = float(log_futures.iloc[i])
        y_list.append([y_t])
        # Scalar maturity
        tau_t = float(maturities.iloc[i])
        # Observation matrix (1×3)
        Z_t = np.array([
            [1,
             -A_tau(a, tau_t),
             B_tau(kappa, tau_t)]
        ])
        Z_list.append(Z_t)
        # Offset d_t (scalar)
        d_t = float(
            C_tau(kappa, alpha_hat, a, m_star,
                 sigma_1, sigma_2, sigma_3, rho_12,
                 tau_t)
        )
        d_list.append([d_t])

    # Convert observations to array
    y_obs = np.array(y_list)
    # Observation covariance
    H = np.array([[0.01]])

    # Initial state
    x0 = np.array([
        float(log_futures.iloc[0]),   # S_0
        0.1,                          # δ_0
        float(r_t.iloc[0])            # r_0
    ])
    P0 = np.diag([0.5, 1, 0.001])

    # Kalman filter
    try:
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
        log_lik = kf.loglikelihood(y_obs)
        if verbosity and (likelihood_counter % verbosity_cooldown == 0):
            print(f"{'--- Current Parameters ---':^50}")
            print(f"{'SPOT / CONVENIENCE YIELD':<25}{'Value':>25}")
            print(f"{'kappa':<25}{kappa:>25.6f}")
            print(f"{'alpha_hat':<25}{alpha_hat:>25.6f}")
            print(f"{'sigma_1':<25}{sigma_1:>25.6f}")
            print(f"{'sigma_2':<25}{sigma_2:>25.6f}")
            print(f"{'rho_12':<25}{rho_12:>25.6f}")

            print(f"{'VASICEK PARAMETERS':<25}{'':>25}")
            print(f"{'a':<25}{a:>25.6f}")
            print(f"{'m*':<25}{m_star:>25.6f}")
            print(f"{'sigma_3':<25}{sigma_3:>25.6f}")
            print(f"{'rho_23':<25}{rho_23:>25.6f}")
            print("-" * 50)
            print(f"{'Log likelihood':<25}{log_lik:>25,.2f}")
            print("-" * 50)
    except Exception as e:
        print("Kalman error:", e)
        return np.inf
    return -log_lik


##### VASICEK ESTIMATION TO GET SHORT RATE PARAMETERS

def calibrate_schwartz3(commodity_ticker: str= None, calibration_start_date: str= None, vasicek_calibration_start_date:str= None, calibration_end_date: str=date.today().strftime("%Y-%m-%d"), vasicek_estimates: tuple = None, verbosity = False, verbosity_cooldown = 10):
    """
    Calibrate the Schwartz 3-factor model to commodity futures prices and short rate data using MLE via Kalman filter.
    """

    vprint = print if verbosity else lambda *a, **k: None
    if vasicek_calibration_start_date is None:
        vasicek_calibration_start_date = calibration_start_date

    if vasicek_estimates is None:
        calibration_rates = load_short_rate_data('data/raw_data/DTB3.csv', start_date=vasicek_calibration_start_date, end_date=calibration_end_date)
        vasicek_estimates = calibrate_vasicek(calibration_rates= calibration_rates,verbosity=verbosity)
        vprint("\n =============== VASICEK ESTIMATION DONE, PARAMETERS PLUGGED TO SCHWARTZ MLE ===============")
    
    a, m_star, sigma_3 = vasicek_estimates


    commodity_calibration = load_calibration_data(commo_ticker=commodity_ticker, start_date=calibration_start_date, end_date=calibration_end_date)
    vprint(f'Commodity calibration dataframe {commodity_ticker} successfully loaded: length:', len(commodity_calibration))
    commodity_calibration['log_futures'] = np.log(commodity_calibration['price'])
    
    maturities_schwartz3 = commodity_calibration['time_to_maturity']
    log_futures_schwartz3 = commodity_calibration['log_futures']
    r_t_schwartz3 = commodity_calibration['r_t']

    ### Initial parameters for optimization
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

    ### Initial bounds for optimization
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
        negative_log_likelihood_schwartz3,
        initial_params,
        args=(log_futures_schwartz3, maturities_schwartz3, r_t_schwartz3, 1/252, verbosity, verbosity_cooldown),
        method='L-BFGS-B',
        bounds=bounds
    )

    calibration_end_time = time.time()
    calibration_duration = calibration_end_time - calibration_start_time
    print(f"\nCalibration completed in {calibration_duration:.2f} seconds.")

    # Print estimated parameters 
    param_names = ["kappa", "alpha_hat", "a", "m_star", "sigma_1", "sigma_2", "sigma_3", "rho_12", "rho_23"]

    print(f"\n========= Estimated Parameters for {commodity_ticker} =========")
    print(f'Sample size: {len(commodity_calibration)} observations, from {calibration_start_date} to {calibration_end_date}')
    for name, value in zip(param_names, result.x):
        print(f"{name:12s} = {value: .6f}")

    print(f"\nMaximized Log-Likelihood: { -result.fun :.6f}")
    return result.x


schwartz_3_estimation = calibrate_schwartz3(commodity_ticker=calibration_ticker, calibration_start_date=calibration_start_date, calibration_end_date=calibration_end_date, verbosity=False, verbosity_cooldown=10)
print(schwartz_3_estimation)
