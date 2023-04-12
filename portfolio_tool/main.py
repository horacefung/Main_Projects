'''
 Main interface with rest of portfolio tool components
'''

# Standard imports
import pandas as pd
import datetime
from pytz import timezone
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Project imports
from components.cash import CashBalance
from components.investments import InvestmentPortfolio
from components.alpha_vantage_api import AlphaVantageAPI
import components.gcp_api
from components.keys import secrets


class PortfolioManager(CashBalance, InvestmentPortfolio):
    def __init__(self):
        self.project_id = secrets['project_id']
        self.workbook = secrets['gsheet_workbook']
        self.dataset = secrets['dataset']
        self.sleep_tracker = 0

        # Tables
        self.fund_table = secrets['fund_table']
        self.invested_cap_table = secrets['invested_cap_table']
        self.other_investments_table = secrets['other_investments_table']
        self.cash_balance_table = secrets['cash_balance_table']

        #Hard code which accounts and users are allowed, prevents typos
        self.allowed_owners = secrets['allowed_owners']
        self.allowed_accounts = secrets['allowed_accounts']
        self.allowed_banks = secrets['allowed_banks']

        # Save the unique keys for convenience 
        self.fund_keys = ['as_of_date', 'fund_manager', 'fund_name', 'account_name', 'bank', 'ticker', 'currency']
        self.invested_cap_keys = ['as_of_date', 'account_owner', 'fund_name', 'account_name']
        self.other_keys = ['as_of_date', 'account_owner', 'account_name', 'bank', 'ticker', 'currency']
        self.cash_key = ['as_of_date', 'account_owner', 'account_name', 'currency']

    # --- Regular pipeline --- #
    def run_update(self):
        '''Once gsheet inputs are ready, run update to append new data entry
        for every table.'''

        # Update cash balance
        self.cash_entry()
        # Update fund
        self.fund_entry()
        # Update other investments
        self.other_investments_entry()

    # --- Analytical capabilities --- #

if __name__ == '__main__':
    portfolio = PortfolioManager()
    portfolio.run_update()



