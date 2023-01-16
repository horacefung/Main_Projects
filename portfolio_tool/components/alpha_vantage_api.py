'''
Alpha Vantage API Wrapper
'''

import pandas as pd
from pandas.tseries.offsets import BMonthEnd  
import requests
from keys import secrets

AV_KEY = secrets['alpha_vantage']


# Alpha Vantage API wrappers
def security_price(ticker, endpoint):
    ''' API connection to alpha vantage. Allows various price queries, either historical time series
    at different increments or direct quote of current prices.

    Keyword Arguments
    ticker -- security ticker symbol
    endpoint -- which endpoint to hit e.g. "quote" for current price.
    '''

    if endpoint == 'quote':
        url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={AV_KEY}'
    elif endpoint == 'daily_time_series':
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval=5min&apikey={AV_KEY}'
    elif endpoint == 'weekly_time_series':
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol={ticker}&apikey={AV_KEY}'
    elif endpoint == 'monthly_time_series':
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol={ticker}&apikey={AV_KEY}'
    else:
        url = None
    
    if url is not None:
        r = requests.get(url)
        data = r.json()
    else:
        data = {}

    return data

def fred():
    ''' API connection to alpha vantage for FRED economic indicators
    '''
    return None

def fx_rates(curr_from, curr_to, dates=[]):
    ''' Get monthly exchange rate time series. We can add more features for different frequencies
    but right now it's not worth it. 

    Keyword Arguments:
    curr_from -- The current currency
    curr_ot -- The currency to exchange to
    '''
    curr_from = curr_from.upper()
    curr_to = curr_to.upper()
    offset = BMonthEnd() 

    if len(dates) == 0:
        # No specific range of dates, just get current
        url = f'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={curr_from}&to_currency={curr_to}&apikey={AV_KEY}'
        r = requests.get(url)
        data = r.json()
        rate = float(data['Realtime Currency Exchange Rate']['5. Exchange Rate'])
        return rate
    else:
        # Given specific dates
        url = f'https://www.alphavantage.co/query?function=FX_MONTHLY&from_symbol={curr_from}&to_symbol={curr_to}&apikey={AV_KEY}'
        r = requests.get(url)
        data = r.json()
        data = data['Time Series FX (Monthly)']
        fx_dates = list(data.keys())

        rates = {}
        for i in dates:
            for j in range(1, len(fx_dates)):
                if (i <= fx_dates[j-1]) & (i >= fx_dates[j]):
                    # Found a date in between
                    rates[i] = float(data[fx_dates[j-1]]['4. close'])

        return rates

if __name__ == '__main__':
    fx_rates('HKD', 'USD', dates=['2021-12-01', '2019-12-30'])