import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import os
print(os.getcwd())

base = '/Users/macos/Desktop/demoday/demoday/ML_Model'
reg = xgb.XGBRegressor()
model = pickle.load(open('/model1.pkl', "rb"))
print(model.get_booster().feature_names)
# model = pickle.load(open(f'{base}/model1.pkl', "rb"))
#model = pickle.load(open(os.path.join(os.getcwd(), "model1.pkl"), "rb"))

#model = reg.load_model(f'{base}/model2.json')

def create_f(df):
    ''' Create a DataFrame with .....'''     ### à travailler 
    df = df.copy()
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week
    
    return df


def create_period(start_date, end_date):
    '''Create a Pandas dataframe for time interval from date_start to date_end'''
    
    new_period  = pd.date_range(start_date +' 12:00:00+00:00', end_date+' 12:00:00+00:00', freq='10min')
    new_period  = pd.DataFrame(index=new_period)
    new_period = create_f(new_period)
    return new_period 
