from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def Model(X_train, Y_train):
    n_steps_in = X_train.shape[1]
    n_features = X_train.shape[2]
    n_steps_out = Y_train.shape[1]

    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(optimizer=Adam(learning_rate=0.1, clipnorm =1), loss='mae')
    # fit model
    model.fit(X_train, Y_train, batch_size= 32, epochs=10, verbose=1, validation_split= 0.1)
    return model