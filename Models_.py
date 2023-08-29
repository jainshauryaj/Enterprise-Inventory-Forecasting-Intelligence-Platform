##################################
## Models Function File
##################################


import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
# import time
# import datetime


import statsmodels.api as sm
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from statsmodels.tsa.arima.model import ARIMA

import kats.utils.time_series_parameter_tuning as tpt
from kats.consts import ModelEnum, SearchMethodEnum, TimeSeriesData
# from ax.core.parameter import ChoiceParameter, FixedParameter, ParameterType
# from ax.models.random.sobol import SobolGenerator
# from ax.models.random.uniform import UniformGenerator
from kats.detectors.seasonality import FFTDetector
from kats.models.ensemble.ensemble import EnsembleParams, BaseModelParams
from kats.models.ensemble.kats_ensemble import KatsEnsemble
from kats.models import (arima,holtwinters,linear_model,prophet,quadratic_model,sarima,theta)
# from kats.utils.backtesters import BackTesterSimple

from warnings import filterwarnings
filterwarnings("ignore")

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


## Get Seasonal Period
def get_seasonal_periods(check):
    """
    input : Pass the time series modifiered by timeseries function of Kats

    return : List of seasonal periods
    """

    seasonal_perdiods = sorted(
        list(
            set(
                [int(i)for i in FFTDetector(check).detector(mad_threshold=0.8)['seasonalities']]
            )))
    
    seasonal_perdiods = [i for i in seasonal_perdiods if (i < int(0.8*len(check))) & (i > 2) & (i < 100)]
    seasonal_perdiods.append(7)
    seasonal_perdiods.append(12)
    seasonal_perdiods.append(52)
    return list(set(seasonal_perdiods))


## Holt Winter Model
##############################
## Tuning Holt winter to getthe right seasonal period
###############################
def tune_HW(check,seasonal_perdiods):
    """
    Tuning the Holt Winter Model where we get the right Seasonal Periods, Trend & Seasonality
    # Build Parameter grid
    # Create parameter search function
    # Split the data into 80-20
    # Train & Evaluate with Custom evaluation function
    # We are using our own custome evaluation function as the default uses MAE as the evaluation metrics but here we are using the MAPE as the defaults error metrics.


    Input : Timeseries, List of seasonal Periods

    return : Dict of best Parameter, MAPE of Holtwinter model

    """
    parameters_grid_search = [
    {
        "name": "trend",
        "type": "choice",
        "values": ['mul','add'],
        "value_type": "str",
        "is_ordered": True,
    },
    {
        "name": "seasonal",
        "type": "choice",
        "values": ['mul','add'],
        "value_type": "str",
        "is_ordered": True,
    },
    {
        "name": "seasonal_periods",
        "type": "choice",
        "values": seasonal_perdiods,
        "value_type": "int",
        "is_ordered": True,
    },
    ]

    parameter_tuner_grid = tpt.SearchMethodFactory.create_search_method(objective_name="evaluation_metric",
                                                                        parameters=parameters_grid_search,
                                                                        selected_search_method=SearchMethodEnum.GRID_SEARCH,)
    # Divide into an 80/20 training-test split
    split = int(0.8*len(check))

    train_ts = check[0:split]
    test_ts = check[split:]
    
    # Fit an model and calculate the err for the test data
    def evaluation_function(params, er='mape'):
        hw_params = holtwinters.HoltWintersParams(trend = params['trend'],
                                                  seasonal = params['seasonal'],
                                                  seasonal_periods = params['seasonal_periods'])
        try:
            model = holtwinters.HoltWintersModel(train_ts, hw_params)
            model.fit()
            model_pred = model.predict(steps=len(test_ts))
            if er == 'mse':
                error = mean_squared_error(test_ts.value.values,model_pred['fcst'].values)
            elif er == 'mape':
                error = mean_absolute_percentage_error(test_ts.value.values,model_pred['fcst'].values)*100
            elif er == 'mae':
                error = np.mean(np.abs(model_pred['fcst'].values - test_ts.value.values))
        except:
            error = 100000000
        return error

    parameter_tuner_grid.generate_evaluate_new_parameter_values(evaluation_function=evaluation_function)

    # Retrieve parameter tuning results
    parameter_tuning_results_grid = (parameter_tuner_grid.list_parameter_value_scores())

    #parameter_tuning_results_grid
    print(parameter_tuning_results_grid.sort_values('mean',ascending=True).reset_index(drop=True).head(1).to_dict())
    tune_parm = parameter_tuning_results_grid.sort_values('mean',ascending=True).reset_index(drop=True).head(1).to_dict()['parameters'][0]
    tune_mape = parameter_tuning_results_grid.sort_values('mean',ascending=True).reset_index(drop=True).head(1).to_dict()['mean'][0]
    return tune_parm,tune_mape


# ## ARIMA Model  /// Currently Not working as the Kats has some issue under naming once start working will enable it
# ##############################
# ## Tuning ARIMA to get the right p,d,q
# ###############################

# def tune_Arima(check):
#     """
#     Tuning the ARIMA Model where we get the right p,d,q
#     # Build Parameter grid
#     # Create parameter search function
#     # Split the data into 80-20
#     # Train & Evaluate with Custom evaluation function
#     # We are using our own custome evaluation function as the default uses MAE as the evaluation metrics but here we are using the MAPE as the defaults error metrics.


#     Input : Timeseries, 

#     return : Dict of best Parameter, MAPE of Holtwinter model

#     """
#     parameters_grid_search = [
#         {
#             "name": "p",
#             "type": "choice",
#             "values": list(range(5)),
#             "value_type": "int",
#             "is_ordered": True,
#         },
#         {
#             "name": "d",
#             "type": "choice",
#             "values": list(range(3)),
#             "value_type": "int",
#             "is_ordered": True,
#         },
#         {
#             "name": "q",
#             "type": "choice",
#             "values": list(range(4)),
#             "value_type": "int",
#             "is_ordered": True,
#         },
#     ]

#     parameter_tuner_grid = tpt.SearchMethodFactory.create_search_method(objective_name="evaluation_metric",
#                                                                         parameters=parameters_grid_search,
#                                                                         selected_search_method=SearchMethodEnum.GRID_SEARCH,)
#     # Divide into an 80/20 training-test split
#     split = int(0.8*len(check))

#     train_ts = check[0:split]
#     test_ts = check[split:]

#     # Fit an model and calculate the err for the test data
#     def evaluation_function(params, er='mape'):
#         ar_params = arima.ARIMAParams(p = params['p'],d = params['d'],q = params['q'])
#         model = arima.ARIMAModel(train_ts, ar_params)
#         model.fit()
#         model_pred = model.predict(steps=len(test_ts))
#         print(model_pred['fcst'])
#         if er == 'mse':
#             error = mean_squared_error(test_ts.value.values,model_pred['fcst'].values)
#         elif er == 'mape':
#             error = mean_absolute_percentage_error(test_ts.value.values,model_pred['fcst'].values)*100
#         elif er == 'mae':
#             error = np.mean(np.abs(model_pred['fcst'].values - test_ts.value.values))
#         return error


#     parameter_tuner_grid.generate_evaluate_new_parameter_values(evaluation_function=evaluation_function)

#     # Retrieve parameter tuning results
#     parameter_tuning_results_grid = (parameter_tuner_grid.list_parameter_value_scores())

#     #parameter_tuning_results_grid
#     print(parameter_tuning_results_grid.sort_values('mean',ascending=True).reset_index(drop=True).head(1).to_dict())
#     tune_parm = parameter_tuning_results_grid.sort_values('mean',ascending=True).reset_index(drop=True).head(1).to_dict()['parameters'][0]
#     tune_mape = parameter_tuning_results_grid.sort_values('mean',ascending=True).reset_index(drop=True).head(1).to_dict()['mean'][0]
#     return tune_parm,tune_mape

############################
###  ARIMA (Above Function is not working due to some bug in the KATs lib)
############################
def Arima_order_tune(train_series,test_series):
    """
    input : Train & Test Series

    proc : Tune the ARIMA p,d,q parms to get the most optimized value for p,d,q w.r.t minumum mape

    return : best parms, MAPE
    """
    p = range(5)
    d = range(3)
    q = range(4)
    pdq = list(itertools.product(p,d,q))

    best_score, best_cfg = np.inf, None
    predictions = []
    for param in tqdm(pdq):
        try:
            model = ARIMA(train_series, order=param)
            result = model.fit()
            pred = result.predict(test_series.index.min(),test_series.index.max())
            print(param)
            mape = mean_absolute_percentage_error(test_series, pred)*100
            if mape < best_score:
                best_cfg = param
                best_score = mape
        except:
            continue
    return best_cfg,best_score

def ARIMA_Check(check,fcst_step):
    """
    input : Timeseries, Forecasted Steps

    process : Split into 80/20
              Call Tune Param to get the right parameter & MAPE of Model
              Check if MAPE is less than 30% 
                yes : Generate forecast with the Same Config
                No : Return with MAPE, & Conf  
    
    return : MAPE error, Param, Forecasted Data
    """
    train_series = check[:int(len(check)*0.8)]
    test_series = check[int(len(check)*0.8):]

    best_cfg,best_score = Arima_order_tune(train_series,test_series)

    if best_score <= 30:
        model = ARIMA(endog=check,order=best_cfg,enforce_stationarity=False,enforce_invertibility=False)
        res = model.fit()
        fcst = res.forecast(fcst_step).reset_index()#.rename(columns={'index':'time','predicted_mean':'fcst'})
        fcst.columns = ['time','fcst']
        return [best_score,best_cfg,fcst]
    else:
        #simply return
        return [best_score,best_cfg,None]


## Ensemble Model  
###############################
def Ensemble_Model(tune_parm,check,step, sale_data):
    """
    Current Build the ensemble method using following models:
        1. SARIMA
        2. Holt Winter (Tuned)
        3. fbprophet
        4. Linear
        5. Quadratic
        6. Theta

    # Build Parameter grid
    # Define Ensemble Configuration
    # Train
    # Forecast & Execute backtest
    # Average the BAcktest executer result to the Average MAPE of the Ensemble model


    Input : Timeseries, Number of steps to be forecasted, sale_data 

    return : Forecasted Data, MAPE error, Backtest executer results, Model

    """
    model_params = EnsembleParams(
                [
                    # BaseModelParams("arima", arima.ARIMAParams(p=1, d=1, q=1)),
                    BaseModelParams("sarima",sarima.SARIMAParams(p=2,d=1,q=1,trend="ct",
                                                                 seasonal_order=(1,0,1,tune_parm['seasonal_periods']),
                                                                 enforce_invertibility=False,
                                                                 enforce_stationarity=False),),
                    BaseModelParams("holtwinters", holtwinters.HoltWintersParams(trend=tune_parm['trend'],
                                                                                 seasonal=tune_parm['seasonal'],
                                                                                 seasonal_periods=tune_parm['seasonal_periods'])),
                    BaseModelParams("prophet", prophet.ProphetParams(holidays = sale_data[['holiday','ds']],seasonality_mode='multiplicative')),  # requires fbprophet be installed
                    BaseModelParams("linear", linear_model.LinearModelParams()),
                    BaseModelParams("quadratic", quadratic_model.QuadraticModelParams()),
                    BaseModelParams("theta", theta.ThetaParams(m=tune_parm['seasonal_periods'])),
                ]
            )
    # create `KatsEnsembleParam` with detailed configurations 
    KatsEnsembleParam = {"models": model_params,
                         "aggregation": "median",
                         "seasonality_length":tune_parm['seasonal_periods'],
                         "decomposition_method": "multiplicative"}
    # create `KatsEnsemble` model
    m = KatsEnsemble(data = check,params=KatsEnsembleParam)

    # fit and predict
    m.fit()

    # predict for the next 30 steps
    fcst = m.predict(steps=step)
    
    # # aggregate individual model results
    m.aggregate()

    # # plot to visualize
    # m.plot()
    forecast_Data = fcst.fcst_df
    print(forecast_Data.shape)
    back_test_result = m.backTestExecutor()
    ERR = sum(list(back_test_result.values()))/len(back_test_result)*100
    return [forecast_Data,ERR,back_test_result,m]


########################################
### Holt Model
########################################
def holt_wp(y_to_train,y_to_test):
    """
    
    input : Train & Test Series 

        process : Tune Slopping level, Slopping Scale, Linear or Exponential Trend
                  Calcualte MAPE

    return : best Score & Configuration

    """
    best_score = np.inf
    conf = [] 
    sl = ss = [round(x * 0.1,2) for x in range(0, 10)]
    comb = list(itertools.product(sl,ss))
    
    # Linear Trend
    for val in comb:
        try:
            fit = Holt(y_to_train).fit(val[0], val[1], optimized=False)
            fcast = fit.forecast(len(y_to_test))
            mape = mean_absolute_percentage_error(y_to_test, fcast)*100
            if mape < best_score:
                best_score = mape
                conf = [val[0],val[1],'Linear']
        except:
            continue

    # Exponential Trend
    for val in comb:
        try:
            fit = Holt(y_to_train, exponential=True).fit(val[0], val[1], optimized=False)
            fcast = fit.forecast(len(y_to_test))
            mape = mean_absolute_percentage_error(y_to_test, fcast)*100
            if mape < best_score:
                best_score = mape
                conf = [val[0],val[1],'Exponential']
#                 trend = ["".join(i) for i in fit.summary().tables[0].as_csv().split()][20][1:-1] + '- Exponential'
        except:
            continue
    
    try:
        fit = Holt(y_to_train).fit()
        fcast = fit.forecast(len(y_to_test))
        mape = mean_absolute_percentage_error(y_to_test, fcast)*100
        if mape < best_score:
            best_score = mape
            conf = [fit.params['smoothing_level'],fit.params['smoothing_trend'],'Linear']
    except:
        pass
    return best_score,conf


def Holt_Check(check,fcst_step):
    """
    input : Timeseries, Forecasted Steps

    process : Split into 80/20
              Call Tune Param to get the right parameter & MAPE of Model
              Check if MAPE is less than 30% 
                yes : Generate forecast with the Same Config
                No : Return with MAPE, & Conf  
    
    return : MAPE error, Param, Forecasted Data
    """
    train_series = check[:int(len(check)*0.8)]
    test_series = check[int(len(check)*0.8):]

    best_score, best_cfg = holt_wp(train_series,test_series)

    if best_score <= 30:
        if best_cfg[2] == 'Exponential':
            model = Holt(check,exponential=True).fit(best_cfg[0],best_cfg[1],optimized=False)
        else:
            model = Holt(check).fit(best_cfg[0],best_cfg[1],optimized=False)
        
        fcst = model.forecast(fcst_step).reset_index()
        fcst.columns = ['time','fcst']
        return [best_score,best_cfg,fcst]
    else:
        #simply return
        return [best_score,best_cfg,None]


########################################
### Simple Exponential Smoothing Model
########################################
def ses_wp(y_to_train,y_to_test):
    """
    
    input : Train & Test Series 

        process : Tune Slopping level
                  Calcualte MAPE

    return : best Score & Configuration

    """
    best_score = np.inf
    best_sl = None
    for sl in [round(x * 0.1,2) for x in range(0, 10)]:
        try:
            fit = SimpleExpSmoothing(y_to_train).fit(smoothing_level=sl, optimized=False)
            fcast = fit.forecast(len(y_to_test))
            mape = mean_absolute_percentage_error(y_to_test, fcast)*100
            if mape < best_score:
                best_score = mape
                best_sl = sl
        except:
            continue

    ## Auto checking as well
    try:
        fit2 = SimpleExpSmoothing(y_to_train).fit()
        fcast2 = fit2.forecast(len(y_to_test))
        mape = mean_absolute_percentage_error(y_to_test, fcast)*100
        if mape < best_score:
            best_score = mape
            best_sl = fit.params['smoothing_level']
    except:
        pass
    
    return best_score,[best_sl]

def ses_Check(check,fcst_step):
    """
    input : Timeseries, Forecasted Steps

    process : Split into 80/20
              Call Tune Param to get the right parameter & MAPE of Model
              Check if MAPE is less than 30% 
                yes : Generate forecast with the Same Config
                No : Return with MAPE, & Conf  
    
    return : MAPE error, Param, Forecasted Data
    """    
    train_series = check[:int(len(check)*0.8)]
    test_series = check[int(len(check)*0.8):]
    best_score, best_cfg = ses_wp(train_series,test_series)
    if best_score <= 30:
        model = SimpleExpSmoothing(check).fit(smoothing_level=best_cfg, optimized=False)        
        fcst = model.forecast(fcst_step).reset_index()
        fcst.columns = ['time','fcst']
        return [best_score,best_cfg,fcst]
    else:
        #simply return
        return [best_score,best_cfg,None]


#############################
## SImple Moving Average Models === > Simple Moving Average // Weighted Moving Average /// Naive
##############################
def sma_optimize(timeseries):
    """
    Tuning the rolling window size for Simple Moving Average

    imput : Monthly time series
    return : optimal_n : as the rolling window size, MSE
    
    """
    optimal_n = None
    best_mse = None
    timeseries = timeseries/1.0
    mean_results_for_all_possible_n_values = np.zeros(int(len(timeseries) / 2 - 2))
    for n in range(3, int(len(timeseries) / 2 + 1)):
        mean_for_n = np.zeros(len(timeseries) - n)
        for i in range(0, len(timeseries) - n):
            mean_for_n[i] = np.round(mean_squared_error([timeseries[i + n]], [np.mean(timeseries[i:i+n])]),2)
#             mean_for_n[i] = np.power(np.mean(timeseries[i:i+n]) - timeseries[i + n], 2)
        mean_results_for_all_possible_n_values[n - 3] = np.mean(mean_for_n)
    optimal_n = np.argmin(mean_results_for_all_possible_n_values) + 3
    best_mse = np.min(mean_results_for_all_possible_n_values)
#     print(f"ALL MSE =>  {mean_results_for_all_possible_n_values}\n\nBest MSE =>{best_mse}\nOptimal window_size => {optimal_n}")
#     print(f"Best MSE =>{best_mse}\nOptimal window_size => {optimal_n}")
    return optimal_n,best_mse

def wma_optimize(timeseries):
    """
    Tuning the rolling window size for SIMPLE moving Average

    imput : Monthly time series
    return : optimal_n : as the rolling window size, MSE
    
    """
    optimal_n = None
    best_mse = None
    timeseries = timeseries/1.0
    mean_results_for_all_possible_n_values = np.zeros(int(len(timeseries) / 2 - 2))
    for n in range(3, int(len(timeseries) / 2 + 1)):
        mean_for_n = np.zeros(len(timeseries) - n)
        for i in range(0, len(timeseries) - n):
            weight = 1
            divider = 0
            result = 0
            for _data in timeseries[i:i+n]: ## Assigning Weights
                result += _data * weight
                divider += weight
                weight += 1
            obs = result / divider ## Next Value
#             mean_for_n[i] = np.power(obs - timeseries[i + n], 2)
            mean_for_n[i] = np.round(mean_squared_error([timeseries[i + n]], [obs]),2)
        mean_results_for_all_possible_n_values[n - 3] = np.mean(mean_for_n)
    optimal_n = np.argmin(mean_results_for_all_possible_n_values) + 3
    best_mse = np.min(mean_results_for_all_possible_n_values)
#     print(f"Best MSE =>{best_mse}\nOptimal window_size => {optimal_n}")
    return optimal_n,best_mse

def naive_forecast(model,timeseries,optimal_n,fcount):
    """
    Single Point Forecast to n

    imput : Model NAme : 'ma','wma', Monthly Time Series, Optimal value of rolling window, Forecasted count required
    
    return : forecasted timeseries added to original
    
    """

    ts = timeseries.copy()
    for forecastedcount in tqdm(range(fcount)):
        if model == 'wma':
            weight = 1
            divider = 0
            result = 0
            for _data in ts[len(ts) - optimal_n: len(ts)]:
                result += _data * weight
                divider += weight
                weight += 1
            ts[ts.index.max() + pd.DateOffset(months=1)] = np.round(result / divider)
        elif model == 'ma':
            ts[ts.index.max() + pd.DateOffset(months=1)] = np.mean(ts[len(ts) - optimal_n:len(ts)])
    return ts


def Naive_Models(timeseries,fcst_cnt):
    """
    Entire NAive Forecast wrapper

    imput : Time Series, Forecasted count required
    
    return : [MSE, [Model, Parameter], Forecasted Count]
    
    """    
    timeseries = timeseries.resample('M').sum()
    max_val = timeseries.index.max()
#     naive_config = [None,None]
    if len(timeseries) > 5:
        ma_mp = sma_optimize(timeseries)
        wma_mp = wma_optimize(timeseries)
        if ma_mp[1] < wma_mp[1]:
            fcst = naive_forecast(model='ma',timeseries=timeseries,optimal_n=ma_mp[0],fcount=fcst_cnt)
            fcst = fcst.loc[fcst.index > max_val].reset_index()
            fcst.columns = ['time','fcst']
            return [ma_mp[1],['ma',ma_mp[0]],fcst]
        else:
            fcst = naive_forecast(model='wma',timeseries=timeseries,optimal_n=wma_mp[0],fcount=fcst_cnt)
            fcst = fcst.loc[fcst.index > max_val].reset_index()
            fcst.columns = ['time','fcst']
            return [wma_mp[1],['wma',wma_mp[0]],fcst]
    else:
        fcst = pd.DataFrame(columns=['time','fcst'])
        return [-1,['Naive',-1],fcst]



def Store_fdata(Model_result,_sku, Model_Name):
    forecast_df = Model_result[2]
    forecast_df.fcst = forecast_df.fcst - 1
    forecast_df['reference'] = _sku[1]
    forecast_df['platform'] = _sku[0]
    forecast_df['err'] = np.round(Model_result[0],2)
    forecast_df['config'] = str(Model_result[1])
    forecast_df['Model'] = Model_Name
    return forecast_df