import numpy as np

def predictor(model, test_tabular):
    X_cols = ['Global CMP22 (vent/cor) [W/m^2]', 'Direct sNIP [W/m^2]',
              'Azimuth Angle [degrees]', 'Tower Dry Bulb Temp [deg C]',
              'Tower Wet Bulb Temp [deg C]', 'Tower Dew Point Temp [deg C]',
              'Tower RH [%]', 'Peak Wind Speed @ 6ft [m/s]',
              'Avg Wind Direction @ 6ft [deg from N]', 'Station Pressure [mBar]',
              'Precipitation (Accumulated) [mm]', 'Snow Depth [cm]', 'Moisture',
              'Albedo (CMP11)', 'Total Cloud Cover [%]']

    X_test = np.array(test_tabular[X_cols].values)

    n_steps_in = X_test.shape[0]
    n_features = X_test.shape[1]

    X_test = X_test.reshape((1, n_steps_in, n_features))
    predictions = model.predict(X_test, verbose=1)
    return predictions