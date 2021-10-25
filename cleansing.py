import pandas as pd
import numpy as np

def tab_clean(df):
    df['Total Cloud Cover [%]'] = np.where(df['Total Cloud Cover [%]'] < -1 , np.nan,df['Total Cloud Cover [%]'])
    df['Total Cloud Cover [%]'] = np.where(df['Total Cloud Cover [%]'] > 100, 100, df['Total Cloud Cover [%]'])
    df['Total Cloud Cover [%]'] = df['Total Cloud Cover [%]'].interpolate(method='linear', limit_direction = 'both')
    return df