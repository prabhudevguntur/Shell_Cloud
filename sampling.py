import pandas as pd
import numpy as np
import os
from cleansing import tab_clean
from scipy.stats.stats import pearsonr

def sampler(train_tabular, test_tabular):

    print('Sampling similar data to test')
    test_tabular = tab_clean(test_tabular)

    compare_cols = ['Global CMP22 (vent/cor) [W/m^2]', 'Direct sNIP [W/m^2]',
       'Azimuth Angle [degrees]', 'Tower Dry Bulb Temp [deg C]',
       'Tower Wet Bulb Temp [deg C]', 'Tower Dew Point Temp [deg C]',
       'Tower RH [%]', 'Total Cloud Cover [%]', 'Peak Wind Speed @ 6ft [m/s]',
       'Avg Wind Direction @ 6ft [deg from N]', 'Station Pressure [mBar]',
       'Precipitation (Accumulated) [mm]', 'Snow Depth [cm]', 'Moisture',
       'Albedo (CMP11)']

    test_df = test_tabular[compare_cols]
    train_df = train_tabular[compare_cols]

    nrows = test_df.shape[0]
    overlap = 0

    df_list = []

    for i in range(0, len(train_df) - overlap): # range(0, len(train_df) - overlap, nrows - overlap) for non lstm

        # if  (train_sample.shape[0] < nrows) or (train_sample.index[-1]+120 > len(train_df)):
        #     continue
        try:
            train_sample = train_df.iloc[i: i + nrows]
            correlation = pearsonr(train_sample['Total Cloud Cover [%]'].values,test_df['Total Cloud Cover [%]'].values)
            if correlation[0] > 0.8:
                df_list.append(train_tabular.iloc[i: i + nrows + 120])
        except:
            continue

    return df_list