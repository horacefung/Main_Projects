'''
Gsheet API Wrapper
'''

import pandas as pd
import gspread
import requests
from keys import google_credentials, google_sheet

GC = gspread.service_account_from_dict(google_credentials)
SH = GC.open_by_key(google_sheet['workbook'])

# Google Sheets Wrapper
def sheet_to_df(sheet_name):
    worksheet = SH.worksheet(sheet_name)
    list_of_dicts = worksheet.get_all_records()
    df = pd.DataFrame(list_of_dicts)
    df = df.dropna()
    return df


if __name__ == '__main__':
    cash = sheet_to_df("Current Cash")
    breakpoint()
