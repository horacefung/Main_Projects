#load libraries
import pandas as pd
import numpy as np
#from urllib.request import urlopen # import urllib library
import json # import json
from sodapy import Socrata

import urllib
import requests

def test(one, **kwargs):
    print(one)
    check = input('Put into')

    return check

#create function that stores the JSON data pull from data.cdc.gov and converts it into a Pandas dataframe
def pull_cdc_data(id):
    '''
    input: dataset identifier
    output: Pandas dataframe of the JSON data 
    '''
    client = Socrata(
        domain = "data.cdc.gov",
        app_token = "bsYMLkKAQhIIVd7wzBzp5BiCR",
        timeout=10
    )
    return pd.DataFrame(client.get_all(id))


import urllib
import requests
def lat_lon_to_fips(lat, lon):
    #Sample latitude and longitudes
    #lat = 41.859127
    #lon = -87.719686

    #Encode parameters 
    params =  urllib.parse.urlencode({'latitude': lat, 'longitude':lon, 'format':'json'})
    #Contruct request URL
    url = 'https://geo.fcc.gov/api/census/block/find?' + params

    #Get response from API
    response = requests.get(url)

    #Parse json in response
    data = response.json()
    #Print FIPS code
    return data['County']['FIPS']




if __name__ == '__main__':
    #assign variable names to the CDC dataset identifiers
    covid_case_id = "kn79-hsxy" #https://data.cdc.gov/NCHS/Provisional-COVID-19-Death-Counts-in-the-United-St/kn79-hsxy
    vaccination_id = "8xkx-amqh" #https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh
    vaccine_location_id = "5jp2-pgaw" #https://data.cdc.gov/Vaccinations/Vaccines-gov-COVID-19-vaccinating-provider-locatio/5jp2-pgaw

    #covid_df = pull_cdc_data(covid_case_id)
    vaccination_df = pull_cdc_data(vaccination_id)
    #lat_lon_to_fips()
    breakpoint()