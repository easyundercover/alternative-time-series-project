#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly

#Import data
data_train_a = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-a.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
data_test_a = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-a.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
data_train_b = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-b.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
data_test_b = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-b.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)

#Reset index for dataset A
data_train_a.reset_index(inplace=True)

#Rename cols
data_train_a.rename(columns={'datetime': 'ds', 'cpu': 'y'}, inplace=True)

#Fit Prophet model and predict
m = Prophet()
m.fit(data_train_a)
future = m.make_future_dataframe(periods=1)
forecast = m.predict(future)
m = Prophet(weekly_seasonality=False)
m.add_seasonality(name='hourly', period=60, fourier_order=5)
forecast = m.fit(data_train_a).predict(future)

#Reload datasets to fit an ARIMA model
data_train_a = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-a.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
data_test_a = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-a.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
data_train_a.index = pd.to_datetime(data_train_a.index)
data_train_b.index = pd.to_datetime(data_train_b.index)

#Fit model
from pmdarima.arima import auto_arima
stepwise_model = auto_arima(data_train_a, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
stepwise_model.fit(data_train_a)
future_forecast = stepwise_model.predict(n_periods=60)
future_forecast = pd.DataFrame(future_forecast,index = data_test_a.index,columns=['Prediction'])


#Reset index for dataset B
data_train_b.reset_index(inplace = True)

#Rename cols
data_train_b.rename(columns={'datetime': 'ds', 'cpu': 'y'}, inplace=True)

#Fit Prophet model
n = Prophet()
n.fit(data_train_b)
forecast = n.predict(future)
n = Prophet(weekly_seasonality=False)
n.add_seasonality(name='hourly', period=60, fourier_order=5)
forecast = n.fit(data_train_b).predict(future)
future = n.make_future_dataframe(periods=300, freq='1min')
fcst = n.predict(future)

#Reload datasets to fit an ARIMA model
data_train_b = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-b.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
data_test_b = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-b.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
data_test_b.index = pd.to_datetime(data_test_b.index)
data_train_b.index = pd.to_datetime(data_train_b.index)

#Fit model and predict
from pmdarima.arima import auto_arima
stepwise_model_b = auto_arima(data_train_b, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
stepwise_model_b.fit(data_train_b)          
future_forecast = stepwise_model_b.predict(n_periods=60)    
future_forecast = pd.DataFrame(future_forecast,index = data_test_b.index,columns=['Prediction'])

#Save model
import pickle
filename = '../models/stepwise_model.pkl'
pickle.dump(stepwise_model, open('../models/stepwise_model.pkl', 'wb'))