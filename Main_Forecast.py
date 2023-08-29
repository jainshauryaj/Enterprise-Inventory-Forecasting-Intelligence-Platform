#################  
### Main Code file
## Backend call : Models.py & Helper.py
##################


### Import Libraries
from tqdm import tqdm
# import requests
# from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
# from dateutil.parser import parse
# import statsmodels.api as sm
# import itertools
# from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# import pmdarima as pm
# import glob
# import datetime
# import time

# import kats.utils.time_series_parameter_tuning as tpt
from kats.consts import TimeSeriesData#, ModelEnum, SearchMethodEnum
# from ax.core.parameter import ChoiceParameter, FixedParameter, ParameterType
# from ax.models.random.sobol import SobolGenerator
# from ax.models.random.uniform import UniformGenerator
# from kats.detectors.seasonality import FFTDetector
# from kats.models.ensemble.ensemble import EnsembleParams, BaseModelParams
# from kats.models.ensemble.kats_ensemble import KatsEnsemble
from kats.models import holtwinters #(arima,holtwinters,linear_model,prophet,quadratic_model,sarima,theta)
# from kats.utils.backtesters import BackTesterSimple

from warnings import filterwarnings
filterwarnings("ignore")

from Models_ import *  # Contains all the Models related Inofrmation
from Helper import * # Contains all the Helper Functions required

import os
# os.chdir('/Users/shaurya/Desktop/multione/')

if __name__ == "__main__":

    org = pd.read_csv('Data/Entire_Channel_Data.csv')  # Load Data
    
    ## Forecasted // Output Files
    raw_forecasted_file = 'Raw_forecasted_File.csv'  # Raw level Forecasting
    output_forecasted = 'Data/Raw_Forecasted.csv' # Final Forecasted File
    Model_Parm_file = 'Data/Model_Params.csv' # Models Config File
    
    ### Also for M & W still some functions may all due no short length of data // Currently 'D' is working fine
    TimesSeries_Expansion = 'D' #  'D' = 'Daily', 'M' = Month,'W' = Week

    if TimesSeries_Expansion == 'D':
        Num_f_count = 365
    elif TimesSeries_Expansion == 'M':
        Num_f_count = 12
    else:
        Num_f_count = 52

    ### PReprocessing and analysis
    org = org.rename(columns={'qty_valid':'quantity'})
    org.order_date = pd.to_datetime(org.order_date, dayfirst=True)
    org.quantity = org.quantity.astype('int64')
    # org = org.drop_duplicates().reset_index(drop=True)
    df = org.copy()
        
    df['reference'] = df['reference'].apply(clean_reference)
    df['product_name'] = df['product_name'].apply(clean_reference)

    ### Removing Testing data
    remove_data = ['KILL-productesting123','testtt','shadetest1','testproduct123','KILL-testing-product-isi-bundle','Testing_0001','Testing03',
                   'Testing Product Item','ITM.Test-shll2','ITM.Test-shll','testing0002','Testing04','AAA','AAI8200','10-GLR','10-BLR','ss4444',
                   'sada1231231','testvirtualbundle1','item testing 12345','test produk vietnam','Product Testing 1','Item Testing Shl - replica 1',
                   'testing332211','testing112233','testingoffline']

    df = df.loc[~df.reference.isin(remove_data)]
    df = df.loc[~df.product_name.isin(remove_data)].reset_index(drop=True)
    # df.loc[df.reference.str.contains('est')].product_name.value_counts().index.tolist()

    print(f'reference having Null Values before removing Null order_date: {df.loc[df.reference.isnull()].shape[0]}')
    # df.head()

    df = df.loc[~df.order_date.isnull()].reset_index(drop=True)
    # print(f'\nRemoving Rows where Dates are missing --> DataFrame Shape --> {df.shape}')
    print(f'reference having Null Values after removing Null order_date: {df.loc[df.reference.isnull()].shape[0]}')


    ## Detect Missing
    n_found, Mone_found, one_found = [],[],[]
    ttmp = org.loc[org.product_name.isin(df.loc[df.reference.isnull()].product_name.value_counts().index.tolist())]

    for i in tqdm(df.loc[df.reference.isnull()].product_name.value_counts().index.tolist()):
        tmp = ttmp.loc[ttmp.product_name==i]
        if tmp.reference.value_counts().shape[0] < 1:
            n_found.append(i)
        elif tmp.reference.value_counts().shape[0] == 1:
            one_found.append(i)
        else:
            Mone_found.append(i)
            
    print(f'Reference Not found w.r.t Product Name : {len(n_found)}\nSingle Reference found w.r.t Product Name : {len(one_found)}\n\
    Multiple Reference found w.r.t Product Name : {len(Mone_found)}')

    # From Orginial Data Extracting Single entry found Data
    ## Fill Missing Values
    tmp = df.loc[df.product_name.isin(one_found)].groupby('product_name').first().reset_index()
    df_one_found = df.loc[df.product_name.isin(one_found)].reset_index(drop=True)
    df_one_found['reference'] = df_one_found.merge(tmp, how='left', on='product_name').reference_y.values # Filled With Uniquely Present

    df = pd.concat([df,df_one_found])
    df = df.loc[df.reference.notnull()]
    print(f'Missing value left in reference : {df.reference.isnull().sum()}')

    complete_df = df.loc[df.quantity > 0].drop_duplicates().reset_index(drop=True).copy()
    complete_df = complete_df.groupby(['platform','reference','order_date'])['quantity'].sum().reset_index()
    complete_df = complete_df.loc[complete_df.quantity > 0].drop_duplicates().reset_index(drop=True)

    print(f'Total SKUs => {complete_df.reference.value_counts().shape[0]} Active SKUs  => {complete_df.loc[complete_df.order_date >= complete_df.order_date.max()- pd.DateOffset(months=6)].reference.value_counts().shape[0]}')
    active_df = complete_df.loc[complete_df.reference.isin(complete_df.loc[complete_df.order_date >= complete_df.order_date.max()- pd.DateOffset(months=6)].reference.value_counts().index.tolist())]
    fbprophet_df = []
    sm_df = []
    for i in ['Online','Offline','GT','MT','Edit']:
        print('='*50)
        print(f'\t{i}')
        print('='*50)
        print(f"Total {i} SKUs => {complete_df.loc[complete_df.platform==i].reference.value_counts().shape[0]} {i} Active SKU => {active_df.loc[active_df.platform==i].reference.value_counts().shape[0]}")
        _tmp = active_df.loc[active_df.platform==i].reference.value_counts().index.tolist()
        _tmp = complete_df.loc[(complete_df.platform==i)&(complete_df.reference.isin(_tmp))]
        tmp = _tmp.reference.value_counts().reset_index().rename(columns={'index':'reference','reference':'cnt'})
        print(f'Large Datasets (fb-prophet) : {tmp.loc[tmp.cnt >= 365].shape[0]}')
        print(f'Medium & Small Dataset (Exponential // ARIMA // NAIVE) : {tmp.loc[tmp.cnt < 365].shape[0]}')
        print(f'Medium Dataset : {tmp.loc[(tmp.cnt > 10) & (tmp.cnt < 365)].shape[0]}')
        print(f'Small Dataset (NAIVE) : {tmp.loc[tmp.cnt <= 10].shape[0]}')
        print('\n')
        fbprophet_df.append(_tmp.loc[_tmp.reference.isin(tmp.loc[tmp.cnt >= 365].reference.tolist())])
        sm_df.append(_tmp.loc[_tmp.reference.isin(tmp.loc[tmp.cnt < 365].reference.tolist())])
    fbprophet_df = pd.concat(fbprophet_df).drop_duplicates().reset_index(drop=True)
    sm_df = pd.concat(sm_df).drop_duplicates().reset_index(drop=True)

    max_date = complete_df.order_date.max()
    print(f"\tMedium & Small\nTotal Models : {sm_df.groupby(['platform','reference']).count().reset_index().shape[0]} \n\tCategorize :\n {sm_df.groupby(['platform','reference']).count().reset_index().platform.value_counts()}")

    #### Merging both the Data adding Max date to all & Expanding with Daily Expansion
    full_data = pd.concat([sm_df,fbprophet_df]).drop_duplicates().reset_index(drop=True)
    # full_data = fbprophet_df.reset_index(drop=True)
    max_date_df = full_data.groupby(['platform','reference']).count().reset_index()
    max_date_df.order_date = max_date
    max_date_df.quantity = 0
    full_data = pd.concat([full_data,max_date_df]).reset_index(drop=True)

    new_db = []
    gbp = full_data.groupby(['platform','reference'])
    for i in tqdm(full_data.groupby(['platform','reference']).count().index):
        tmp = gbp.get_group(i).set_index('order_date')['quantity'].resample(TimesSeries_Expansion).sum().reset_index()
        tmp['platform'] = i[0]
        tmp['reference'] = i[1]
        new_db.append(tmp)
    new_db = pd.concat(new_db).reset_index(drop=True)

    last_org = new_db.copy()

    #### Creating Month DB & Storing the Max Value --- for Later Use
    if TimesSeries_Expansion != 'M':
        print('Calculating the Max Value')
        mon_new_db = []
        mon_gbp = last_org.groupby(['platform','reference'])
        for i in tqdm(mon_gbp.count().index):
            tmp = mon_gbp.get_group(i).set_index('order_date')['quantity'].resample('M').sum().reset_index()
            tmp['platform'] = i[0]
            tmp['reference'] = i[1]
            mon_new_db.append(tmp)
        mon_new_db = pd.concat(mon_new_db).reset_index(drop=True)
        mon_new_db = mon_new_db.groupby(['platform','reference'])['quantity'].max().reset_index()
        mon_new_db = mon_new_db.rename(columns={'quantity':'Max_Value'})
    else:
        mon_new_db = last_org.groupby(['platform','reference'])['quantity'].max().reset_index()
        mon_new_db = mon_new_db.rename(columns={'quantity':'Max_Value'})
    
    #############################################
    ## Data Ready Fore Modeling & Forecasting
    #############################################

    ### Getting all the Sale days & Holidays as the Sale Events in an entire year
    sale_data = Build_Sale_Data()

    new_gbp = new_db.groupby(['platform','reference'])
    output = []
    for sku in tqdm(new_gbp.count().index):
        sku_data = new_gbp.get_group(sku).sort_values('order_date',ascending=True).reset_index(drop=True)
        sku_data.quantity = sku_data.quantity + 1   ## Modify data to remove 0 and make them as 1
        tmp = sku_data[['order_date','quantity']].copy()
            
        ## Train // Evaluate // Generate Forecast with Ensemble Model
        if tmp.shape[0] > 3:
            tmp.columns = ['time','value']
            tmp = TimeSeriesData(tmp)
            try:
                seasonal_period = get_seasonal_periods(tmp)  ## Getting list of Seasonal Period to tune
                holtiwinter_param, holtiwinter_mape = tune_HW(tmp,seasonal_period)  ## Tune Holt Winter Model to get the right parm
                Ensemble_result = Ensemble_Model(holtiwinter_param, tmp, Num_f_count, sale_data)
                if Ensemble_result[1] <= 30:
                    forecast_df = Ensemble_result[0]
                    forecast_df.fcst = forecast_df.fcst - 1
                    forecast_df['reference'] = sku[1]
                    forecast_df['platform'] = sku[0]
                    forecast_df['err'] = np.round(Ensemble_result[1],2)
                    forecast_df['config'] = str(holtiwinter_param)
                    forecast_df['Model'] = 'Ensemble'
                    output.append(forecast_df)
                    continue
                else:
                    print(f'"Ensemble Model" Failed with Error Rate : {Ensemble_result[1]}')
            except:
                print('"Ensemble Model" Failed to train & Building Forecast !!!')

            ### Check Fore Holt Winter if Mape is less then forecast with it
            try:
                seasonal_period = get_seasonal_periods(tmp)  ## Getting list of Seasonal Period to tune
                holtiwinter_param, holtiwinter_mape = tune_HW(tmp,seasonal_period)  ## Tune Holt Winter Model to get the right parm
                if holtiwinter_mape <= 30:
                    params = holtwinters.HoltWintersParams(trend=holtiwinter_param['trend'],
                                                        seasonal=holtiwinter_param['seasonal'],
                                                        seasonal_periods = holtiwinter_param['seasonal_periods'])
                    m = holtwinters.HoltWintersModel(data=tmp, params=params)
                    m.fit()
                    forecast_df = m.predict(steps=Num_f_count, alpha = 0.1)
                    forecast_df.fcst = forecast_df.fcst - 1
                    forecast_df['reference'] = sku[1]
                    forecast_df['platform'] = sku[0]
                    forecast_df['err'] = np.round(holtiwinter_mape,2)
                    forecast_df['config'] = str(holtiwinter_param)
                    forecast_df['Model'] = 'Holt Winter'
                    output.append(forecast_df)
                    continue
                else:
                    print(f'"Holt Winter Model" Failed with Error Rate : {holtiwinter_mape}')
            except:
                print('"Holt Winter Model" Failed to train & Building Forecast !!!')

        else:
            print('Not fit for Kats library  : Data too small !!!')

        ## Check for ARIMA
        tmp = sku_data[['order_date','quantity']].copy()
        tmp = tmp.set_index('order_date')['quantity']
        try:
            ARIMA_result = ARIMA_Check(tmp,fcst_step=Num_f_count)
            if ARIMA_result[0] <= 30:
                output.append(Store_fdata(ARIMA_result,sku, 'ARIMA'))  ## Storing df of a Forecasted Data
                continue
            else:
                print(f'"ARIMA Model" Failed with Error Rate : {ARIMA_result[0]}')
        except:
            print('"ARIMA Model" Failed to train & Building Forecast !!!')


        ## Check for Holt Model
        try:
            Holt_result = Holt_Check(tmp,fcst_step=Num_f_count)
            if Holt_result[0] <= 30:
                output.append(Store_fdata(Holt_result,sku, 'Holt'))
                continue
            else:
                print(f'"HOLT Model" Failed with Error Rate : {Holt_result[0]}')
        except:
            print('"HOLT Model" Failed to train & Building Forecast !!!')


        ## Simple Exponential Model
        try:
            SES_result = ses_Check(tmp,fcst_step=Num_f_count)
            if SES_result[0] <= 30:
                output.append(Store_fdata(SES_result,sku, 'SES'))
                continue
            else:
                print(f'"Simple Exponential Model" Failed with Error Rate : {SES_result[0]}')
        except:
            print('"Simple Exponential Model" Failed to train & Building Forecast !!!')
        ### ImpleMent NAIVE  (Weighted Moving Average, Simple Moving Average or Simple Average)
        try:
            Naive_result = Naive_Models(tmp,12)
            output.append(Store_fdata(Naive_result,sku, 'Naive'))
            continue
        except:
            print('"NAIVE Model" Failed to train & Building Forecast !!!')

    final_result = pd.concat(output).drop_duplicates().reset_index(drop=True)
    final_result.to_csv(raw_forecasted_file,index=False)  ## storing Raw Forecasted

    ### Building Model Config File
    model_conf = final_result.groupby(['reference','platform','err','config','Model'])['fcst'].count().reset_index()
    model_conf = model_conf[['reference','platform','err','config','Model']]    
    model_conf.to_csv(Model_Parm_file,index=False)
    print(f'Final_SKU forecasted or Triggered : {model_conf.shape}')

    #### Build Final Forecast File
    final_gb = final_result.groupby(['reference','platform'])
    refined_forecast = []
    for i in tqdm(final_gb.count().index):
        tmp = final_gb.get_group(i)
        if TimesSeries_Expansion != 'M':
            tmp = tmp.set_index('time')['fcst'].resample('M').sum().reset_index()    
        # tmp.fcst = tmp.fcst.apply(lambda x : 0 if int(round(x)) <= 0 else int(x))
        tmp['platform'] = i[1]
        tmp['reference'] = i[0]
        refined_forecast.append(tmp)
    refined_forecast = pd.concat(refined_forecast).drop_duplicates().reset_index(drop=True)
    refined_forecast  = refined_forecast.rename(columns={'time':'order_date','fcst':'quantity'})
    refined_forecast['Max_Value'] = refined_forecast.merge(mon_new_db,on=['platform','reference'],how='left')['Max_Value'].values
    refined_forecast['Max_Value'] = refined_forecast['Max_Value'] * 5
    refined_forecast['Final_Q'] = refined_forecast[['quantity','Max_Value']].apply(lambda x : int(x[1]) if x[0] > x[1] else int(x[0]),axis=1)
    refined_forecast = refined_forecast[['reference','platform','order_date', 'Final_Q']].rename(columns={'Final_Q':'quantity'})
    
    ## Merging Perdiction with Original DF
    final_forecast = pd.concat([last_org,refined_forecast]).reset_index(drop=True)
    final_gbp = final_forecast.groupby(['reference','platform'])
    combined_forecast = []
    for i in tqdm(final_gbp.count().index):
        tmp = final_gbp.get_group(i)
        tmp = tmp.set_index('order_date')['quantity'].resample('M').sum().reset_index()
        tmp['platform'] = i[1]
        tmp['reference'] = i[0]
        combined_forecast.append(tmp)
    combined_forecast = pd.concat(combined_forecast).drop_duplicates().reset_index(drop=True)
    combined_forecast = combined_forecast.loc[(combined_forecast.order_date > max_date)].reset_index(drop=True)
    
    combined_forecast.to_csv(output_forecasted,index=False)
    
    print('Executed Sucessfully')