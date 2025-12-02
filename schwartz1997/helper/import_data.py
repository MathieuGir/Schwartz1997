import pandas as pd
import numpy as np
from datetime import date
import yfinance as yf


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

def load_short_rate_data(file_path: str = 'data/raw_data/DTB3.csv', start_date:str = '1980-01-01', end_date:str = date.today().strftime('%Y-%m-%d')) -> pd.DataFrame:
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


def download_calibration_rates_data(rate_ticker = '^IRX', start_date = '2015-01-01', end_date = date.today().strftime('%Y-%m-%d')):
    """
    Download interest rates data from Yahoo Finance.

    Args:
        rate_ticker (str): Ticker symbol for the interest rate.
        start_date (str): Start date for data download in 'YYYY-MM-DD' format.
        end_date (str): End date for data download in 'YYYY-MM-DD' format.
    Returns:
        pd.DataFrame: DataFrame containing the interest rates data.
    """
    rates_data = yf.download(rate_ticker, start=start_date, end=end_date)['Close']
    #rename the column name to r_t
    rates_data.rename(columns={rate_ticker: 'r_t'}, inplace=True)
    return rates_data


def load_calibration_data(commo_ticker: str, rate_file_path: str = 'data/raw_data/DTB3.csv', start_date='1980-01-01', end_date='2025-01-01'):
    """
    Load commodity prices and short rate data for calibration.
    """
    commodity_data = load_commodity_prices(f'data/raw_data/{commo_ticker}.csv', start_date=start_date, end_date=end_date)
    commodity_data['time_to_maturity'] = commodity_data.apply(lambda row: get_time_to_maturity(row.name, row['contract'], verbosity=False), axis=1)
    short_rate_data = load_short_rate_data(rate_file_path, start_date=start_date, end_date=end_date)
    
    # Merge datasets on date index
    calibration_data = commodity_data.join(short_rate_data, how='inner')
    
    return calibration_data
