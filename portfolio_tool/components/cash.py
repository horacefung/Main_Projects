'''
Cash component, class for handling cash balance. 

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

class CashBalance(AlphaVantageAPI):
    def __init__(self, secrets):
        self.project_id = secrets['project_id']
        self.workbook = secrets['gsheet_workbook']
        self.dataset = secrets['dataset']
        self.cash_balance_table = secrets['cash_balance_table']
        self.sleep_tracker = 0

        #Hard code which accounts and users are allowed, prevents typos
        self.allowed_owners = secrets['allowed_owners']
        self.allowed_accounts = secrets['allowed_accounts']
        self.allowed_banks = secrets['allowed_banks']

        # Save the unique key for convenience 
        self.cash_key = ['as_of_date', 'account_owner', 'account_name', 'currency']
    
    # --- Prepare functions --- #
    def prep_cash(self, cash_df):
        ''' Enforce column types, clean up strings and
        add additional columns

        Keyword Arguments:
        cash_df -- Raw input dataframe of cash balance
        '''
        cash_df = cash_df.dropna() #filter out blank rows

        # Enforce column selection
        cols = {'as_of_date':'str', 'account_owner':'str', 'account_name':'str',
                 'bank':'str', 'currency':'str', 'amount':'float'}
        cash_df = cash_df[list(cols.keys())]

        for i in cols.keys():
            if cols[i] in ['float']:
                cash_df[i] = cash_df[i].str.replace(',', '')
            cash_df[i] = cash_df[i].astype(cols[i])
            # For strings, trim any whitespace
            if cols[i] == 'str':
                cash_df[i] = cash_df[i].str.lower().str.strip()
        
        # Check accounts
        assert set(cash_df['account_owner']) <= set(self.allowed_owners), "Non-certified owner detected"
        assert set(cash_df['account_name']) <= set(self.allowed_accounts), "Non-certified name detected"
        assert set(cash_df['bank']) <= set(self.allowed_banks), "Non-certified bank detected"

        # Check duplicates
        check = cash_df.groupby(self.cash_key).count()
        assert len(check) == len(cash_df), 'Non-unique key'

        # Timestamp
        tz = timezone('EST')
        timestamp = datetime.datetime.now(tz)   
        timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')
        cash_df['ingestion_timestamp'] = timestamp

        # Currency conversion, everything to USD
        fx_df = self.get_currency_df(cash_df)
        cash_df = pd.merge(cash_df, fx_df, how='left', on=['currency', 'as_of_date'])
        cash_df.loc[cash_df['currency']=='usd', 'fx_rate'] = 1
        cash_df['amount_usd'] = cash_df['amount'] * cash_df['fx_rate']

        cols = list(cash_df.columns)
        cols.remove('as_of_date')
        cols.remove('ingestion_timestamp')
        cols = ['as_of_date', 'ingestion_timestamp'] + cols # time always first two columns
        cash_df = cash_df[cols]

        return cash_df
    
    # --- Update table functions --- #
    def initiate_history(self):
        '''Keep it simple for now, take the raw gsheet of history'''
        history = gcp_api.sheet_to_df(self.workbook, "Backpopulate Cash", col_range="A:F")
        history = self.prep_cash(history)
        history = history.sort_values(by='as_of_date', ascending=True)
        self.history = history
        gcp_api.df_to_bq(history, dataset=self.dataset, table=self.cash_balance_table)

        return print("Initiated history")
    
    def cash_entry(self):
        ''' Fetch user inputs from current cash tab.'''
        curr_cash = gcp_api.sheet_to_df(self.workbook, "Current Cash", col_range="A:F")
        curr_cash = self.prep_cash(curr_cash)

        # Check there is no duplicates
        check_key = self.cash_key + ['ingestion_timestamp']
        key_str = ', '.join(check_key)
        check_query = f'''SELECT DISTINCT {key_str} FROM {self.project_id}.{self.dataset}.{self.cash_balance_table}'''
        check_df = gcp_api.bq_to_df(check_query)
        check_df = pd.merge(check_df, curr_cash[check_key], how='inner', on=check_key)
        assert len(check_df) == 0, print(check_df)
        
        # Save current cash balance
        self.curr_cash = curr_cash
        gcp_api.df_to_bq(curr_cash, dataset=self.dataset, table=self.cash_balance_table, if_exists='append')
        
        return print("Appended new balance to history")

    def delete_entry(self):
        return None
    
    # Functions related to analytical views
    def account_cash():
        return None


if __name__ == '__main__':
    cash_balance = CashBalance(secrets)
    cash_balance.initiate_history()
    #cash_balance.cash_entry()