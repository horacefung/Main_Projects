'''
Alpha Vantage API Wrapper
'''

import requests
from keys import api_keys

AV_KEY = api_keys['alpha_vantage']


# Alpha Vantage API wrappers
def alpha_vantage_price(ticker, endpoint):
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

def alpha_vantage_fred():
    ''' API connection to alpha vantage for FRED economic indicators
    '''
    return None

