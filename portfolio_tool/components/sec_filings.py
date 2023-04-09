'''
Get SEC data for company analysis

Each quarter, SEC publishes nicely flatten data of all filings. They take very little space and should be efficient to read-in
and translate into a BQ table. The idea is to generate 4 tables:

1. Company directory data, current only
2. Balance sheet data, append to form history
3. Income statement, append to form history
4. Cash flow, append to form history

'''
# Standard imports
import sys
import pandas as pd
import datetime
from pytz import timezone
sys.path.append('../components')

# Project imports
import alpha_vantage_api
import gcp_api
from keys import secrets

class SEC():
    def __init__(self, secrets):
        self.project_id = secrets['project_id']
        self.dataset = secrets['dataset']
    
    # -- Fetch balance sheet, income statement, statement of cash --- #
    def read_data(self):

        for i in ['num', 'pre', 'sub', 'tag']:
            df = pd.read_csv(f'./2022q4/{i}.txt', sep='\t')
            gcp_api.df_to_bq(df, dataset=self.dataset, table=f"sec_{i}", if_exists='replace')


if __name__ == '__main__':
    sec_con = SEC(secrets)
    sec_con.read_data()