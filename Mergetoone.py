import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob

new_data = pd.concat([pd.read_csv(i) for i in tqdm(glob('Data/New_Data_29-nov/**/*.csv'))]).drop_duplicates().reset_index(drop=True)
new_data.order_date = pd.to_datetime(new_data.order_date,dayfirst=True)

b2c = new_data.loc[new_data.team_name.isnull()][['order_date','product_name','order_platform','reference','qty_valid']]
b2b = new_data.loc[~new_data.team_name.isnull()][['order_date','product_ref','product_name','team_name','ordered_qty_valid']]
b2b = b2b.rename(columns={'product_ref':'reference','team_name':'order_platform','ordered_qty_valid':'qty_valid'})

b2c_category = {'Online':['sociolla','android','ios','lilla','sociocommerce_sociolla','sociocommerce_internal','lulla','chatbot','carasun'],
                'Offline':['offline_store','offline_store'],
                'Edit':['shopee','lazada','Shopee COSRX','tokopedia','zalora']}
b2c_category = {cod: k for k, cods in b2c_category.items() for cod in cods }
b2c['platform'] = b2c.order_platform.map(b2c_category)
# b2c.to_csv('new_b2c.csv',index=False)

b2b_category = {'MT':['B2B','MT','B2B MT','B2B Banjarmasin MT','B2B Solo MT','B2B Lampung MT'],
                'GT':['GT','B2B Banjarmasin GT','B2B Lampung GT','B2B Cirebon','B2B GT','B2B Semarang',
                      'B2B Bandung','B2B Tangerang','B2B Padang','B2B Pekanbaru','B2B Samarinda','B2B Denpasar',
                      'B2B Pontianak','B2B Palembang','B2B Aceh','B2B Medan','B2B Jaktim - Bekasi','B2B Malang',
                      'B2B Solo GT','B2B Yogyakarta','B2B Surabaya','B2B Bogor','B2B Makassar']}
b2b_category = {cod: k for k, cods in b2b_category.items() for cod in cods }
b2b['platform'] = b2b.order_platform.map(b2b_category)

completedb = pd.concat([b2b,b2c]).reset_index(drop=True)
completedb.to_csv('New_Complete Data.csv',index=False)