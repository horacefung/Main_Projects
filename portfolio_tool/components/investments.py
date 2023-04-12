'''
Investment component, class for handling investments (my own and pooled funds)

'''
# Standard imports
import pandas as pd
import datetime
from pytz import timezone
import sys
sys.path.append('./components')

# Project imports
from alpha_vantage_api import AlphaVantageAPI
import gcp_api
from keys import secrets


class InvestmentPortfolio(AlphaVantageAPI):
    def __init__(self, secrets):
        self.project_id = secrets['project_id']
        self.workbook = secrets['gsheet_workbook']
        self.dataset = secrets['dataset']
        self.sleep_tracker = 0

        # Tables
        self.fund_table = secrets['fund_table']
        self.invested_cap_table = secrets['invested_cap_table']
        self.other_investments_table = secrets['other_investments_table']

        #Hard code which accounts and users are allowed, prevents typos
        self.allowed_owners = secrets['allowed_owners']
        self.allowed_accounts = secrets['allowed_accounts']
        self.allowed_banks = secrets['allowed_banks']

        # Save the unique keys for convenience 
        self.fund_keys = ['as_of_date', 'fund_manager', 'fund_name', 'account_name', 'bank', 'ticker', 'currency']
        self.invested_cap_keys = ['as_of_date', 'account_owner', 'fund_name', 'account_name']
        self.other_keys = ['as_of_date', 'account_owner', 'account_name', 'bank', 'ticker', 'currency']
    

    @staticmethod
    def column_types(df, cols_dict):
        df = df[list(cols_dict.keys())]
        for i in cols_dict.keys():
            if cols_dict[i] in ['float']:
                df[i] = df[i].str.replace(',', '')
            df[i] = df[i].astype(cols_dict[i])
            # For strings, trim any whitespace
            if cols_dict[i] == 'str':
                df[i] = df[i].str.lower().str.strip()
        
        return df

    # --- Prepare functions --- #
    def prep_cap_table(self, cap_df):
        ''' Prep invested capital df
        '''
        cap_df = cap_df.dropna() #filter out blank rows

        # Enforce column selection
        cols = {'as_of_date':'string', 'account_owner':'string', 'fund_name':'string', 'account_name':'string',
                'invested_capital':'float', 'current_allocation_usd':'float', 'current_allocation_pct':'float'}
        cap_df = self.column_types(cap_df, cols)

        # Timestamp
        tz = timezone('EST')
        timestamp = datetime.datetime.now(tz)   
        timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')
        cap_df['ingestion_timestamp'] = timestamp

        return cap_df

    def prep_portfolio(self, port_df, portfolio_type):
        ''' Import portfolio information and update with API data + cleaning.

        Keyword Arguments:
        port_df -- Raw input dataframe of porfolios (including fund portfolios)
        '''
        port_df = port_df.dropna() #filter out blank rows
        assert portfolio_type in ['fund', 'other']

        # Enforce column selection
        if portfolio_type == 'other':
            cols = {'as_of_date':'str', 'account_owner':'str', 'account_name':'str',
                    'bank':'str', 'ticker':'str', 'currency':'str', 'weighted_avg_cost':'float',
                    'share_count':'float', 'total_cost':'float', 'price':'float'}
            
            port_df = self.column_types(port_df, cols)

            # Check accounts
            assert set(port_df['account_owner']) <= set(self.allowed_owners), "Non-certified owner detected"
            assert set(port_df['account_name']) <= set(self.allowed_accounts), "Non-certified name detected"
            assert set(port_df['bank']) <= set(self.allowed_banks), "Non-certified bank detected"

            # Check duplicates
            check = port_df.groupby(self.other_keys).count()
            assert len(check) == len(port_df), 'Non-unique key'

        elif portfolio_type == 'fund':
            cols = {'as_of_date':'str', 'fund_manager':'str', 'fund_name':'str','account_name':'str',
                    'bank':'str', 'ticker':'str', 'currency':'str', 'weighted_avg_cost':'float',
                    'share_count':'float', 'total_cost':'float'}

            port_df = self.column_types(port_df, cols)

            # Check accounts
            assert set(port_df['account_name']) <= set(self.allowed_accounts), "Non-certified name detected"
            assert set(port_df['bank']) <= set(self.allowed_banks), "Non-certified bank detected"

            # Check duplicates
            check = port_df.groupby(self.fund_keys).count()
            assert len(check) == len(port_df), 'Non-unique key'
        else:
            exit(0)

        # Timestamp
        tz = timezone('EST')
        timestamp = datetime.datetime.now(tz)   
        timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')
        port_df['ingestion_timestamp'] = timestamp

        # Currency conversion, everything to USD
        fx_df = self.get_currency_df(port_df)
        port_df = pd.merge(port_df, fx_df, how='left', on=['currency', 'as_of_date'])
        port_df.loc[port_df['currency']=='usd', 'fx_rate'] = 1

        # Hit API with ticker to autopopulate closing price
        price_df = self.get_ticker_price_df(port_df)
        port_df = pd.merge(port_df, price_df, how='left', on=['ticker', 'as_of_date'])

        port_df.loc[port_df['ticker']!='none', 'price'] = port_df['live_price']
        port_df = port_df.drop('live_price', axis=1)
        port_df = port_df.rename({'live_price':'price'}, axis=1)

        cols = list(port_df.columns)
        cols.remove('as_of_date')
        cols.remove('ingestion_timestamp')
        cols = ['as_of_date', 'ingestion_timestamp'] + cols # time always first two columns
        port_df = port_df[cols]

        return port_df
    
    # --- Update table functions --- #
    # There is no back-populate function for fund right now, my fund is new and there is no historical data
    def fund_pull(self, initiate=False):
        '''Fetch current cap and fund table from gsheets. Check if there are duplicates, if there are,
        output error.'''
        # Cap table
        cap_table = gcp_api.sheet_to_df(self.workbook, "Current Invested Capital", col_range="A:G")
        cap_table = self.prep_cap_table(cap_table)

        
        if initiate is False:
            check_key = self.invested_cap_keys + ['ingestion_timestamp']
            key_str = ', '.join(check_key)
            check_query = f'''SELECT DISTINCT {key_str} FROM {self.project_id}.{self.dataset}.{self.invested_cap_table}'''
            check_df = gcp_api.bq_to_df(check_query)
            check_df = pd.merge(check_df, cap_table[check_key], how='inner', on=check_key)
            assert len(check_df) == 0, print(check_df)

        self.cap_table = cap_table

        # Fund 
        fund = gcp_api.sheet_to_df(self.workbook, "Current Fund", col_range="A:J")
        fund = self.prep_portfolio(fund, portfolio_type='fund')

        if initiate is False:
            check_key = self.fund_keys + ['ingestion_timestamp']
            key_str = ', '.join(check_key)
            check_query = f'''SELECT DISTINCT {key_str} FROM {self.project_id}.{self.dataset}.{self.fund_table}'''
            check_df = gcp_api.bq_to_df(check_query)
            check_df = pd.merge(check_df, fund[check_key], how='inner', on=check_key)
            assert len(check_df) == 0, print(check_df)
        self.fund = fund

        assert fund['as_of_date'].loc[0] == cap_table['as_of_date'].loc[0], "Fund and cap table dates mismatched"

    def fund_entry(self, initiate=False):
        '''Upload cleaned fund tables to BQ'''

        if initiate == False:
            self.fund_pull()
            gcp_api.df_to_bq(self.cap_table, dataset=self.dataset, table=self.invested_cap_table, if_exists='append')
            gcp_api.df_to_bq(self.fund, dataset=self.dataset, table=self.fund_table, if_exists='append')
        else:
            self.fund_pull(initiate=True)
            gcp_api.df_to_bq(self.cap_table, dataset=self.dataset, table=self.invested_cap_table, if_exists='replace')
            gcp_api.df_to_bq(self.fund, dataset=self.dataset, table=self.fund_table, if_exists='replace')

    # Update other investments 
    # TODO: Fill investments history
    def initiate_other_investments(self):
        '''Keep it simple for now, take the raw gsheet of history'''
        port_hist = gcp_api.sheet_to_df(self.workbook, "Backpopulate Other Investments", col_range="A:J")
        port_hist = self.prep_portfolio(port_hist, portfolio_type='other')
        port_hist = port_hist.sort_values(by='as_of_date', ascending=True)
        self.port_hist = port_hist
        gcp_api.df_to_bq(port_hist, dataset=self.dataset, table=self.other_investments_table)

        return print("Initiated history")
    
    def other_investments_entry(self):
        ''' Fetch user inputs from other investments tab.'''
        other_invest = gcp_api.sheet_to_df(self.workbook, "Other Investments", col_range="A:J")
        other_invest = self.prep_portfolio(other_invest, portfolio_type='other')

        # Check there is no duplicates
        check_key = self.other_keys + ['ingestion_timestamp']
        key_str = ', '.join(check_key) 
        check_query = f'''SELECT DISTINCT {key_str} FROM {self.project_id}.{self.dataset}.{self.other_investments_table}'''
        check_df = gcp_api.bq_to_df(check_query)
        check_df = pd.merge(check_df, other_invest[check_key], how='inner', on=check_key)
        assert len(check_df) == 0, print(check_df)
        
        # Save to other investments table
        self.other_invest = other_invest
        gcp_api.df_to_bq(other_invest, dataset=self.dataset, table=self.other_investments_table, if_exists='append')
        
        return None


    def delete_entry(self):
        return None
    
    # Functions related to analytical views
    def account_cash():
        return None
        


if __name__ == '__main__':
    investments = InvestmentPortfolio(secrets)
    #investments.fund_entry(initiate=True)
    #investments.initiate_other_investments()
    investments.other_investments_entry()
    