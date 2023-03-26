'''
Investment component, class for handling investments (my own and pooled funds)

'''
# Standard imports
import pandas as pd
import datetime
from pytz import timezone

# Project imports
import alpha_vantage_api
import gcp_api
from keys import secrets

class InvestmentPortfolio:
    def __init__(self, secrets):
        self.project_id = secrets['project_id']
        self.workbook = secrets['gsheet_workbook']
        self.dataset = secrets['dataset']

        # Tables
        self.own_portfolio = secrets['cash_balance_table']
        self.fund_portfolio = secrets['cash_balance_table']
        self.invested_capital = secrets['cash_balance_table']

        #Hard code which accounts and users are allowed, prevents typos
        self.allowed_owners = secrets['allowed_owners']
        self.allowed_accounts = secrets['allowed_accounts']
        self.allowed_banks = secrets['allowed_banks']

        # Save the unique keys for convenience 
        self.portfolio_keys = ['as_of_date', 'account_owner', 'account_name', 'ticker', 'currency']
        self.fund_keys = ['as_of_date', 'account_owner', 'account_name', 'ticker', 'currency']
        self.invested_cap_keys = ['as_of_date', 'account_onwer', 'fund_name', 'account_name']
    

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

        return cap_df

    def prep_portfolio(self, port_df, is_fund):
        ''' Import portfolio information and update with API data + cleaning.

        Keyword Arguments:
        port_df -- Raw input dataframe of porfolios (including fund portfolios)
        '''
        port_df = port_df.dropna() #filter out blank rows

        # Enforce column selection
        if is_fund is False:
            cols = {'as_of_date':'str', 'account_owner':'str', 'account_name':'str',
                    'bank':'str', 'ticker':'str', 'currency':'str', 'weighted_avg_cost':'float',
                    'share_count':'float', 'total_cost':'float', 'price':'float'}
            
            port_df = self.column_types(port_df, cols)

            # Check accounts
            assert set(port_df['account_owner']) <= set(self.allowed_owners), "Non-certified owner detected"
            assert set(port_df['account_name']) <= set(self.allowed_accounts), "Non-certified name detected"
            assert set(port_df['bank']) <= set(self.allowed_banks), "Non-certified bank detected"

            # Check duplicates
            check = port_df.groupby(self.portfolio_keys).count()
            assert len(check) == len(port_df), 'Non-unique key'

        else:
            cols = {'as_of_date':'str', 'account_owner':'str', 'account_name':'str',
                    'bank':'str', 'ticker':'str', 'currency':'str', 'weighted_avg_cost':'float',
                    'share_count':'float', 'total_cost':'float', 'price':'float'}

            port_df = self.column_types(port_df, cols)

            # Check accounts
            assert set(port_df['account_owner']) <= set(self.allowed_owners), "Non-certified owner detected"
            assert set(port_df['account_name']) <= set(self.allowed_accounts), "Non-certified name detected"
            assert set(port_df['bank']) <= set(self.allowed_banks), "Non-certified bank detected"

            # Check duplicates
            check = port_df.groupby(self.fund_keys).count()
            assert len(check) == len(port_df), 'Non-unique key'

        # Timestamp
        tz = timezone('EST')
        timestamp = datetime.datetime.now(tz)   
        timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')
        port_df['ingestion_timestamp'] = timestamp

        # Currency conversion, everything to USD
        fx_df = alpha_vantage_api.get_currency_df(port_df)
        port_df = pd.merge(port_df, fx_df, how='left', on=['currency', 'as_of_date'])
        port_df.loc[port_df['currency']=='usd', 'fx_rate'] = 1
        port_df['amount_usd'] = port_df['amount'] * port_df['fx_rate']

        # Hit API with ticker to autopopulate closing price
        price_df = alpha_vantage_api.get_ticker_price_df(port_df)
        port_df = pd.merge(port_df, price_df, how='left', on=['ticker', 'as_of_date'])

        if is_fund is False:
            port_df.loc[port_df['ticker']!='none', 'price'] = port_df['live_price']
            port_df = port_df.drop('live_price', axis=1)
        else:
            port_df = port_df.rename({'live_price':'price'}, axis=1)

        cols = list(port_df.columns)
        cols.remove('as_of_date')
        cols.remove('ingestion_timestamp')
        cols = ['as_of_date', 'ingestion_timestamp'] + cols # time always first two columns
        port_df = port_df[cols]

        return port_df
    
    # --- Update table functions --- #

    # Fund
    def fund_pull(self):
        # Cap table
        cap_table = gcp_api.sheet_to_df(self.workbook, "Current Invested Capital", col_range="A:G")
        cap_table = self.prep_cap_table(cap_table)

        key_str = ', '.join(self.portfolio_keys)
        check_query = f'''SELECT DISTINCT {key_str} FROM {self.project_id}.{self.dataset}.{self.invested_cap_table}'''
        check_df = gcp_api.bq_to_df(check_query)
        check_df = pd.merge(check_df, cap_table[self.invested_cap_keys], how='inner', on=self.invested_cap_keys)
        assert len(check_df) == 0, print(check_df)
        self.cap_table = cap_table

        # Fund 
        fund = gcp_api.sheet_to_df(self.workbook, "Current Fund", col_range="A:J")
        fund = self.prep_portfolio(fund, is_fund=True)

        key_str = ', '.join(self.portfolio_keys)
        check_query = f'''SELECT DISTINCT {key_str} FROM {self.project_id}.{self.dataset}.{self.fund_table}'''
        check_df = gcp_api.bq_to_df(check_query)
        check_df = pd.merge(check_df, cap_table[self.fund_keys], how='inner', on=self.fund_keys)
        assert len(check_df) == 0, print(check_df)
        self.fund = fund

        assert fund['as_of_date'].loc[0] == cap_table['as_of_date'].loc[0], "Fund and cap table dates mismatched"

    def fund_update(self):
        '''Upload cleaned fund tables to BQ'''
        gcp_api.df_to_bq(self.cap_table, dataset=self.dataset, table=self.invested_cap_table, if_exists='append')
        gcp_api.df_to_bq(self.fund, dataset=self.dataset, table=self.fund_table, if_exists='append')

    # Update other investments 
    def other_investments_history(self):
        '''Keep it simple for now, take the raw gsheet of history'''
        port_hist = gcp_api.sheet_to_df(self.workbook, "Backpopulate Investments", col_range="A:F")
        port_hist = self.prep_portfolio(port_hist, is_fund=False)
        port_hist = port_hist.sort_values(by='as_of_date', ascending=True)
        self.port_hist = port_hist
        gcp_api.df_to_bq(port_hist, dataset=self.dataset, table=self.other_investments_table)

        return print("Initiated history")
    
    def other_investments_entry(self):
        ''' Fetch user inputs from other investments tab.'''
        other_invest = gcp_api.sheet_to_df(self.workbook, "Other Investments", col_range="A:J")
        other_invest = self.prep_portfolio(other_invest, is_fund=False)

        # Check there is no duplicates
        key_str = ', '.join(self.portfolio_keys)
        check_query = f'''SELECT DISTINCT {key_str} FROM {self.project_id}.{self.dataset}.{self.other_investments_table}'''
        check_df = gcp_api.bq_to_df(check_query)
        check_df = pd.merge(check_df, other_invest[self.portfolio_keys], how='inner', on=self.unique_key)
        assert len(check_df) == 0, print(check_df)
        
        # Save current cash balance
        self.other_invest = other_invest
        gcp_api.df_to_bq(other_invest, dataset=self.dataset, table=self.other_investments_table, if_exists='append')
        
        return print("Appended new balance to history")


    def delete_entry(self):
        return None
    
    # Functions related to analytical views
    def account_cash():
        return None


if __name__ == '__main__':
    None
    #cash_balance = CashBalance(secrets)
    #
    #cash_balance.initiate_history()
    #cash_balance.current_entry()