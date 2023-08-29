################################
##  Helper Functions
################################

import requests
from bs4 import BeautifulSoup
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

## Cleaning Data
def clean_reference(row):
    try:
        return row.strip()
    except:
        return row


## Scrape & Build holidays as sale dataframe
def Build_Sale_Data():
    """
    Scrape the Holidays of three decade from https://www.timeanddate.com/holidays/indonesia and clean and organize them to proceed further

    Do : Holidays + all the dates where month and day are same in a year + 24th March

    Input nothing:

    return : data frame 'ds' as date & 'holiday' as the event name
    """
    try:
        indonesia_data = []
        for yr in tqdm(range(2000,2030)):
            try:
                soup = BeautifulSoup(requests.get(f'https://www.timeanddate.com/holidays/indonesia/{yr}').text)
                ind_data = []
                for i in soup.find('table').find_all('tr'):
                    try:
                        ind_data.append([datetime.datetime.fromtimestamp(int(i['data-date'][:-3])).strftime('%Y-%m-%d'),i.find('a').text])
                    except:
                        continue
                ind_data = pd.DataFrame(ind_data)
                ind_data.columns = ['Hdate','Holiday']
                indonesia_data.append(ind_data)
            except:
                continue
                
        indonesia_data = pd.concat(indonesia_data).reset_index(drop=True)
        # indonesia_data.loc[indonesia_data.Holiday.str.contains('Idul')].Holiday.value_counts()
        indonesia_data.Hdate = pd.to_datetime(indonesia_data.Hdate,dayfirst=True)
        indonesia_data = indonesia_data.drop_duplicates('Hdate').reset_index(drop=True)
        indonesia_data['Yr'] = indonesia_data.Hdate.apply(lambda x : x.year)
        # indonesia_data.loc[indonesia_data.Yr==2021].drop_duplicates('Hdate')

        c = pd.DataFrame(pd.to_datetime([f'{j}-{i}-{i}' for i in range(1,13) for j in indonesia_data.Yr.value_counts().index]),columns=['Hdate'])
        c1 = pd.DataFrame(pd.to_datetime([f'{j}-03-24' for j in indonesia_data.Yr.value_counts().index]),columns=['Hdate'])
        sale_days = pd.concat([c,c1]).reset_index(drop=True)

        sale_data = pd.concat([indonesia_data,sale_days]).drop_duplicates('Hdate').reset_index(drop=True)
        sale_data['Special_days'] = 1
        sale_data.Holiday = sale_data.Holiday.fillna('special_day')
        sale_data = sale_data.rename(columns={'Hdate':'ds','Holiday':'holiday'})
        return sale_data
    except:
        c = pd.DataFrame(pd.to_datetime([f'{j}-{i}-{i}' for i in range(1,13) for j in indonesia_data.Yr.value_counts().index]),columns=['ds'])
        c1 = pd.DataFrame(pd.to_datetime([f'{j}-03-24' for j in indonesia_data.Yr.value_counts().index]),columns=['ds'])
        sale_days = pd.concat([c,c1]).reset_index(drop=True)
        sale_data['holiday'] = 'sale days'
        return sale_data
