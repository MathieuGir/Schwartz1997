import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
import time
from scipy.optimize import minimize
from typing import Optional, Union
from schwartz1997.helper.importdata import download_calibration_rates_data

start_date = '2010-01-01'
end_date = '2020-01-01'
# end_date = date.today().strftime('%Y-%m-%d')


def neg_log_likelihood_vasicek_euler(parameters: Optional[Union[np.ndarray, list]] = None,r_t: np.ndarray = None, r_t_plus_1: np.ndarray = None, delta_t: float = 1/252, verbosity: bool = False) -> float:
    """
    Calculate the log-likelihood for the Vasicek model.
    Returns:
        float: Log-likelihood value.
    """

    if isinstance(parameters, np.ndarray):
        parameters = parameters.tolist()

    if parameters is None:
        parameters = [0.1, 0.05, 0.05] # default parameters
    
    a, m_star, sigma_3 = parameters # parameters of the Vasicek model
    T = len(r_t) #number of observations

    sum_sq = np.sum((r_t_plus_1 - r_t - a * (m_star - r_t) * delta_t) ** 2) # for the third member

    log_likelihood = - (T/2) * np.log(2* np.pi) - T * np.log(sigma_3 * np.sqrt(delta_t)) - 1/(2 * sigma_3**2 * delta_t) * sum_sq
    if verbosity:
        print(f'Parameters are a: {a:,.6f}, m*: {m_star:,.6f}, sigma_3: {sigma_3:,.6f}')
        print(f'Log-Likelihood computed: {- log_likelihood:,.2f}')
    return - log_likelihood

def calibrate_vasicek(start_date: str = '2023-01-01', end_date: str = date.today().strftime('%Y-%m-%d'), calibration_rates: pd.DataFrame = None, verbosity: bool = False):
    """
    Calibrate the Vasicek model to interest rate data starting from a given date.
    Args:
        start_date (str): Start date for data download in 'YYYY-MM-DD'
        end_date (str): End date for data download in 'YYYY-MM-DD'
    """
    # Download calibration rates data
    if calibration_rates is None:
        print('No calibration rates given, downloading data from Yahoo Finance')
        calibration_rates = download_calibration_rates_data(start_date=start_date, end_date=end_date) / 100

    print('Dataframe length for calibration rates:', len(calibration_rates))
    # Prepare the data for calibration
    r_t = calibration_rates['r_t'].values[:-1]
    r_t_plus_1 = calibration_rates['r_t'].values[1:]
    # r_t = r_t / 100
    # r_t_plus_1 = r_t_plus_1 / 100

    
    # Initial parameter guesses
    initial_params = [np.float64(0.4), np.float64(r_t.mean()), np.float64(r_t.std())]   
    bounds = [
        (1e-6, None),   #bound for a
        (-0.05, 0.5),   #bound for m* 
        (1e-6, None)    #bound for sigma_3
        ]

    if verbosity:
        print(f'Initial parameters are: a: {initial_params[0]:,.6f}, m*: {initial_params[1]:,.6f}, sigma_3: {initial_params[2]:,.6f}')
    result = minimize(
        fun=neg_log_likelihood_vasicek_euler,
        x0=initial_params,
        args=(r_t, r_t_plus_1, 1/252, verbosity),
        bounds=bounds
    )
    
    if result.success:
        a, m_star, sigma_3 = result.x
        print("\n ---------------- Optimization Results for Vasicek ---------------- ")
        print("\ndr_t+1 = a * (m* - r_t) * dt + sigma_3 * dW_t")
        print(f"Estimated a: {a:.6f}")
        print(f"Estimated m* (%): {m_star*100:.6f}")
        print(f"Estimated sigma_3: {sigma_3:.6f}")
        print(f"(Neg) Log-Likelihood: {result.fun:.2f}")
        return a, m_star, sigma_3
    else:
        print(("### Calibration Failed"))
        print((f"- Message: **{result.message}**"))
        return None

# start_time = time.time()
# calibrate_vasicek(start_date=start_date, end_date=end_date, verbosity=True)
# end_time = time.time()
# print(f"\nCalibration completed in {end_time - start_time:.2f} seconds.") 