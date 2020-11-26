import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import xgboost
import math
import pickle
df = pd.read_csv('C:/Users/91707/OneDrive - LNMIIT/plants_decay_eval/propulsion.csv')


regressor_compressor = xgboost.XGBRegressor()
pickle_in = open('C:/Users/91707/OneDrive - LNMIIT/plants_decay_eval/finalized_model_xgbregressor_compressor.pkl','rb')
regressor_compressor = pickle.load(pickle_in)


regressor_turbine = xgboost.XGBRegressor()
pickle_in = open('C:/Users/91707/OneDrive - LNMIIT/plants_decay_eval/finalized_model_xgbregressor_turbine.pkl','rb')
regressor_turbine = pickle.load(pickle_in)


df_train = df.iloc[:math.ceil(len(df)*0.8),:]
df_test = df.iloc[math.ceil(len(df)*0.8):,:]
y_train_compressor = df_train[['GT Compressor decay state coefficient.']]
y_test_compressor = df_test[['GT Compressor decay state coefficient.']]
y_train_turbine = df_train[['GT Turbine decay state coefficient.']]
y_test_turbine = df_test[['GT Turbine decay state coefficient.']]
df_train = df_train.drop(['GT Compressor decay state coefficient.','GT Turbine decay state coefficient.'],axis = 1)
df_test = df_test.drop(['GT Compressor decay state coefficient.','GT Turbine decay state coefficient.'],axis = 1)
df_train


df_train = df_train.to_numpy()
df_test = df_test.to_numpy()


y_pred_compressor = regressor_compressor.predict(df_test) 
y_pred_turbine = regressor_turbine.predict(df_test) 


y_pred_compressor = regressor_compressor.predict(df_test) 
y_pred_turbine = regressor_turbine.predict(df_test) 


print(math.sqrt(mean_squared_error(y_pred_turbine,y_test_turbine.to_numpy())),' = rmse of turbine of the splitted test data')


print(math.sqrt(mean_squared_error(y_pred_compressor,y_test_compressor.to_numpy())), ' = rmse of  compressor splitted test data')