'''
Gsheet API Wrapper
'''
# Standard imports
import pandas as pd

# GCP imports
import gspread
from google.oauth2 import service_account
import pandas_gbq   

# Project imports
from keys import google_credentials, secrets

# Globals
GC = gspread.service_account_from_dict(google_credentials)
PROJECT_ID = secrets['project_id']
CREDENTIALS = service_account.Credentials.from_service_account_info(google_credentials)

# --- Google Sheets Wrapper --- #
def sheet_to_df(workbook_name, sheet_name, col_range=None):
    sh = GC.open_by_key(workbook_name)
    if col_range is None:
        worksheet = sh.worksheet(sheet_name)
        list_of_dicts = worksheet.get_all_records()
        df = pd.DataFrame(list_of_dicts)
    else:
        list_of_dicts = sh.values_get(range=sheet_name+f"!{col_range}")['values']
        df = pd.DataFrame(list_of_dicts[1:])
        df.columns = list_of_dicts[0]
    
    return df

def df_to_sheet(df, workbook_name, sheet_name, col_range):
    sh = GC.open_by_key(workbook_name)
    worksheet = sh.worksheet(sheet_name)
    worksheet.update(col_range, [df.columns.values.tolist()] + df.values.tolist())
    return print(f"Updated {sheet_name}")

# --- BQ Wrapper --- #
def bq_to_df(query):
    df = pandas_gbq.read_gbq(query, project_id=PROJECT_ID)
    return df

def df_to_bq(df, dataset, table, if_exists='replace'):
    '''
    if_exists = fail, replace, append
    '''
    pandas_gbq.to_gbq(df, f'{dataset}.{table}', project_id=PROJECT_ID, 
                    if_exists=if_exists, credentials=CREDENTIALS)
    return print("Uploaded to BQ table {dataset}.{table}")


if __name__ == '__main__':
    #cash = sheet_to_df("Current Cash")
    df = pd.DataFrame({"my_string": ["a", "b", "c"],"my_int64": [1, 2, 3]})
    pandas_gbq.to_gbq(df, 'portfolio_tool.test_table', project_id=PROJECT_ID, credentials=CREDENTIALS)
