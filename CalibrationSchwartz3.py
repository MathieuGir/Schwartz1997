import pandas as pd
import numpy as np
import os
import time
from datetime import date
from scipy.optimize import minimize
from pykalman import KalmanFilter
from Vasicekcalibration import calibrate_vasicek

calibration_start_date = '2025-11-01'
calibration_end_date = '2025-11-13'

calibration_ticker = 'KC'  # Commodity ticker for calibration
vasicek_calibration_start_date = '2022-04-01'


likelihood_counter = 0
verbosity_cooldown = 5

def load_commodity_prices(file_path, date_col='date', measure_col='measure', measure_value='close', start_date=None, end_date=None):
    """
    Load the CSV file and filter rows where the measure column matches the specified value.
    
    Parameters:
    - file_path: str, path to the CSV file
    - date_col: str, name of the date column
    - measure_col: str, name of the measure column
    - measure_value: str, value to filter the measure column
    
    Returns:
    - pd.DataFrame: filtered DataFrame indexed by date and sorted
    """
    # Load the csv file
    data = pd.read_csv(file_path, sep=',')
    
    # Filter rows where the measure column matches the specified value, and drop measure column
    data = data[data[measure_col] == measure_value]
    
    #for each date and contract, keep only latest time entry
    data = data.sort_values(by=['date', 'time'], ascending=[True, True])
    data = data.drop_duplicates(subset=['date', 'contract'], keep='last')
    data = data.drop(columns=[measure_col, 'time'])
    

    # Rename the value column to price
    data.rename(columns={'value': 'price'}, inplace=True)
    
    # Convert date column to datetime format
    data[date_col] = pd.to_datetime(data[date_col], format='%Y-%m-%d')
    
    # Index by date and sort it
    data = data.set_index(date_col).sort_index()

    # Drop NAN values and values equal to 0
    zero_count = (data == 0).sum()
    if zero_count.any():
        print(f"Dropping rows with zero values:\n{zero_count[zero_count > 0]}")
    data = data[data['price'] != 0]
    na_count = data.isna().sum()
    if na_count.any():
        print(f"Dropping rows with NA values:\n{na_count[na_count > 0]}")
    data.dropna(inplace=True)
    return data

MONTH_MAP = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
}

def get_time_to_maturity(current_date, contract_name, verbosity = False):
    """
    Returns the number of BUSINESS DAYS until the futures contract expires.
    Expiry = 21st day of the delivery month.
    TO BE MODIFIED ACCORDING TO CONTRACT SPECIFICATIONS
    """
    # Convert to Timestamp
    current_date = pd.to_datetime(current_date).normalize()

    # Parse contract
    month_code = contract_name[0]
    year = int(contract_name[1:])
    month = MONTH_MAP[month_code]

    # Expiration = 21st day of that month
    expiration_date = (pd.Timestamp(year, month, 21)).normalize()
    
    business_days = np.busday_count(current_date.date(), expiration_date.date())

    if verbosity:
        print(f"Expiration date for contract {contract_name} is {expiration_date.date()}")
        print(f"Time to maturity from {current_date.date()} is {business_days} business days.")
    # Business-day difference (numpy expects YYYY-MM-DD strings)
    return business_days

def load_short_rate_data(file_path: str = 'data/DTB3.csv', start_date:str = '1980-01-01', end_date:str = date.today().strftime('%Y-%m-%d')) -> pd.DataFrame:
    """
    Returns the DTB3 short rate data (source: FRED) between a given start and end date.
    The rate is converted from percentage to decimal and renamed to 'r_t'.
    """
    short_rate = pd.read_csv(file_path, sep=',') 
    short_rate['observation_date'] = pd.to_datetime(short_rate['observation_date'], format='%Y-%m-%d')
    short_rate = short_rate.set_index('observation_date').sort_index()
    short_rate = short_rate.loc[start_date:end_date]
    short_rate['DTB3'] = short_rate['DTB3'] / 100  # Convert percentage to decimal
    short_rate = short_rate.rename(columns={'DTB3': 'r_t'})
    short_rate.index.name = 'date'
    short_rate.dropna(inplace=True)
    return short_rate

def load_calibration_data(commo_ticker: str, rate_file_path: str = 'data/DTB3.csv', start_date='1980-01-01', end_date='2025-01-01'):
    """
    Load commodity prices and short rate data for calibration.
    """
    commodity_data = load_commodity_prices(f'data/{commo_ticker}.csv', start_date=start_date, end_date=end_date)
    commodity_data['time_to_maturity'] = commodity_data.apply(lambda row: get_time_to_maturity(row.name, row['contract'], verbosity=False), axis=1)
    short_rate_data = load_short_rate_data(rate_file_path, start_date=start_date, end_date=end_date)
    
    # Merge datasets on date index
    calibration_data = commodity_data.join(short_rate_data, how='inner')
    
    return calibration_data


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
        calibration_rates = load_short_rate_data('data/DTB3.csv', start_date=vasicek_calibration_start_date, end_date=calibration_end_date)
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
        args=(log_futures_schwartz3, maturities_schwartz3, r_t_schwartz3, 1/252, True, verbosity_cooldown),
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


schwartz_3_estimation = calibrate_schwartz3(commodity_ticker=calibration_ticker, calibration_start_date=calibration_start_date, calibration_end_date=calibration_end_date, verbosity=True, verbosity_cooldown=10)
print(schwartz_3_estimation)



# Try to plot

# commodity_calibration = load_calibration_data(commo_ticker=calibration_ticker, start_date=calibration_start_date, end_date=calibration_end_date)
# print('Commodity calibration dataframe length:', len(commodity_calibration))
# commodity_calibration['log_futures'] = np.log(commodity_calibration['price'])
# log_futures = commodity_calibration['log_futures']
# maturities = commodity_calibration['time_to_maturity']
# r_t = commodity_calibration['r_t']

# kappa, a, m_star, alpha_hat, sigma_1, sigma_2, sigma_3, rho_12, rho_23 = schwartz_3_estimation
# dt = 1/252

# F = get_transition_matrix(kappa, a, dt)
# c = get_transition_offset(alpha_hat, m_star, sigma_1, kappa, a, dt)
# Q = get_transition_covariance(sigma_1, sigma_2, sigma_3, rho_12, rho_23, dt)

# # Prepare lists for each contract
# y_list = []     # List of T×1 observations
# Z_list = []     # List of 1×3 matrices
# d_list = []     # List of T×1 offsets

#     # Iterate over each contract
# for i in range(len(log_futures)):
#     # Scalar log futures price
#     y_t = float(log_futures.iloc[i])
#     y_list.append([y_t])
#     # Scalar maturity
#     tau_t = float(maturities.iloc[i])
#     # Observation matrix (1×3)
#     Z_t = np.array([
#         [1,
#          -A_tau(a, tau_t),
#          B_tau(kappa, tau_t)]
#          ])
#     Z_list.append(Z_t)
#     # Offset d_t (scalar)
#     d_t = float(
#         C_tau(kappa, alpha_hat, a, m_star,
#               sigma_1, sigma_2, sigma_3, rho_12,
#               tau_t)
#         )
#     d_list.append([d_t])

#     # Convert observations to array
#     y_obs = np.array(y_list)
#     # Observation covariance
#     H = np.array([[0.01]])

#     # Initial state
#     x0 = np.array([
#         float(log_futures.iloc[0]),   # S_0
#         0.1,                          # δ_0
#         float(r_t.iloc[0])            # r_0
#     ])
#     P0 = np.diag([0.1, 0.1, 0.1])

# kf = KalmanFilter(
#     transition_matrices=F,
#     transition_offsets=c,
#     transition_covariance=Q,
#     observation_matrices=Z_list,
#     observation_offsets=d_list,
#     observation_covariance=H,
#     initial_state_mean=x0,
#     initial_state_covariance=P0
# )

# # Run the Kalman filter
# kf.filter(y_obs)
# # Extract the model-implied values and compare them to the observed values

# model_implied_values = kf.smooth(y_obs)[0]  # Smooth the states
# # print(model_implied_values)


# # calibration_rates = load_short_rate_data('data/DTB3.csv', start_date=pd.to_datetime(calibration_start_date) - pd.Timedelta(days=14000), end_date=calibration_end_date)
# calibration_rates = load_short_rate_data('data/DTB3.csv', start_date=vasicek_calibration_start_date, end_date=calibration_end_date)
# vasicek_estimates = calibrate_vasicek(calibration_rates= calibration_rates,verbosity=False)
# a, m_star, sigma_3 = vasicek_estimates

# print(f"\n =============== VASICEK ESTIMATION DONE, PLUGGING INTO SCHWARTZ MLE ===============")

# initial_params = [
#     1,    # kappa
#     0.25,    # alpha_hat
#     a,    # a
#     m_star,    # m_star
#     0.3,    # sigma_1
#     0.3,    # sigma_2
#     sigma_3,   # sigma_3
#     0.3,    # rho_12
#     0     # rho_23
# ]

# # Bounds
# bounds = [
#     (1e-6, None),  # kappa
#     (None, None),  # alpha_hat
#     (a, a),  # a
#     (m_star, m_star),  # m_star
#     (1e-6, None),  # sigma_1
#     (1e-6, None),  # sigma_2
#     (sigma_3, sigma_3),  # sigma_3
#     (-0.9999, 0.9999),  # rho_12
#     (-0.9999, 0.9999)   # rho_23
# ]


# ### SCHWARTZ ESTIMATION

# commodity_calibration = load_calibration_data(commo_ticker=calibration_ticker, start_date=calibration_start_date, end_date=calibration_end_date)
# print('Commodity calibration dataframe length:', len(commodity_calibration))
# commodity_calibration['log_futures'] = np.log(commodity_calibration['price'])
# print('after log futures')

# maturities_schwartz3 = commodity_calibration['time_to_maturity']
# log_futures_schwartz3 = commodity_calibration['log_futures']
# r_t_schwartz3 = commodity_calibration['r_t']

# calibration_start_time = time.time()

# result = minimize(
#     negative_log_likelihood_schwartz3,
#     initial_params,
#     args=(log_futures_schwartz3, maturities_schwartz3, r_t_schwartz3, 1/252, True, verbosity_cooldown),
#     method='L-BFGS-B',
#     bounds=bounds
# )

# calibration_end_time = time.time()
# calibration_duration = calibration_end_time - calibration_start_time
# print(f"\nCalibration completed in {calibration_duration:.2f} seconds.")

# # List of parameter names (matching order in initial_params)
# param_names = [
#     "kappa",
#     "alpha_hat",
#     "a",
#     "m_star",
#     "sigma_1",
#     "sigma_2",
#     "sigma_3",
#     "rho_12",
#     "rho_23"
# ]

# # Print estimated parameters neatly
# print("\n=== Estimated Parameters ===")
# for name, value in zip(param_names, result.x):
#     print(f"{name:8s} = {value: .6f}")

# # Also print log-likelihood
# print(f"\nMaximized Log-Likelihood: { -result.fun :.6f}")

