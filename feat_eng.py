import numpy as np

def feat_engine(train_df_list):

    n_steps_in = 360
    X_cols = ['Global CMP22 (vent/cor) [W/m^2]', 'Direct sNIP [W/m^2]',
       'Azimuth Angle [degrees]', 'Tower Dry Bulb Temp [deg C]',
       'Tower Wet Bulb Temp [deg C]', 'Tower Dew Point Temp [deg C]',
       'Tower RH [%]', 'Peak Wind Speed @ 6ft [m/s]',
       'Avg Wind Direction @ 6ft [deg from N]', 'Station Pressure [mBar]',
       'Precipitation (Accumulated) [mm]', 'Snow Depth [cm]', 'Moisture',
       'Albedo (CMP11)','Total Cloud Cover [%]']

    Y_cols = ['Total Cloud Cover [%]']
    X, y = list(), list()

    y_ind_list = [n_steps_in+30, n_steps_in+60, n_steps_in+90, n_steps_in+120]
    for df in train_df_list:
        df = df.reset_index(drop=True)
        # gather input and output parts of the pattern
        seq_x, seq_y = df.loc[0:n_steps_in, X_cols], df.loc[y_ind_list, Y_cols]
        X.append(seq_x.values)
        y.append(np.concatenate(seq_y.values))
    return np.array(X), np.array(y)