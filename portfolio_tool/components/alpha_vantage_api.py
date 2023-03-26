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
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={AV_KEY}'
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

def security_date_price(ticker, dates):
    data = security_price(ticker, endpoint='weekly_time_series')
    data = data['Weekly Adjusted Time Series']
    security_dates = list(data.keys())

    price = {}
    for i in dates:
        for j in range(1, len(security_dates)):
            if (i <= security_dates[j-1]) & (i >= security_dates[j]):
                # Found a date in between
                price[i] = float(data[security_dates[j-1]]['4. close'])
    return price

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


# --- Pandas Wrappers --- #
def get_currency_df(input_df):
    ''' Given a dataframe with currency and as_of_dates, hit alpha vantage API to 
    get the exchange rate to USD for that date. Return a dataframe with [as_of_date, currency,
    fx_rate].
    '''
    df = input_df.copy()
    assert ['currency', 'as_of_date'] in df.columns, 'missing column for get_currency()'

    # Currency conversion, everything to USD
    currencies = df['currency'].unique().tolist()
    fx_list = []
    for currency in currencies:
        fx_dates = df[df['currency']==currency]['as_of_date'].unique().tolist()
        if currency == 'usd':
            continue
        else:
            fx = fx_rates(currency, 'USD', dates=fx_dates)
            fx = pd.DataFrame({'as_of_date':fx.keys(), 'fx_rate':fx.values()})
            fx['currency'] = currency
        fx_list.append(fx)

    fx_df = pd.concat(fx_list, axis=0)

    return fx_df

def get_ticker_price_df(input_df):
    df = input_df.copy()
    # Hit API with ticker to autopopulate closing price
    tickers = df['ticker'].unique().tolist()
    tickers = [i for i in tickers if i != 'none']
    price_list = []
    for ticker in tickers:
        ticker_dates = df[df['ticker']==ticker]['as_of_date'].unique().tolist()
        price = security_date_price(ticker, dates=ticker_dates)
        price = pd.DataFrame({'as_of_date':price.keys(), 'live_price':price.values()})
        price['ticker'] = ticker
    price_list.append([price])

    price_df = pd.concat(price_list, axis=0)

    return price_df


if __name__ == '__main__':
    #fx_rates('HKD', 'USD', dates=['2021-12-01', '2019-12-30'])
    security_date_price('ESCA', dates=['2023-01-15', '2022-01-15'])
    