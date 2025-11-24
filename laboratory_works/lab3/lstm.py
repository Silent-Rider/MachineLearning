import numpy as np
import pandas as pd
from keras import Input, Model
from keras.src.layers import Dense, LSTM
from pandas import DataFrame

np.random.seed(24)

def generate_data(n:float = 1000) -> DataFrame:
    x = np.random.normal(3, 10, (n, 1))
    e = np.random.normal(0, 0.3, (n, 1))

    data = np.hstack([
        x**2 + e,
        np.sin(x / 2) + e,
        np.cos(2 * x) + e,
        x - 3 + e,
        -x + e,
        np.abs(x) + e,
        (x**3) / 4 + e
    ])

    # F - сокращение от feature (признак)
    data_frame = pd.DataFrame(
        data,
        columns=[f'F{(i + 1)}' for i in range(len(data[0]))]
    )
    return data_frame


def create_general_model():
    input = Input(shape=6)

    #Encoder
    enc = Dense(12, activation='relu')(input)
    enc = Dense(3, activation='relu', name='encoder')(enc)

    #Decoder
    dec = Dense(12, activation='relu')(enc)
    dec = Dense(6, activation='linear', name='decoder')(dec)

    #Regressor
    regr = LSTM(12, activation='relu')(dec)
    regr = Dense(1, activation='linear')(regr)

    general_model = Model(input=input, outputs=[dec, regr])
    general_model.compile(loss='mse', optimizer='adam')
    return general_model


df = generate_data()
df.to_csv('original_dataset.csv', index=False)

X = df.drop(columns=['F5']).values
Y = df['F5'].values.reshape(1000, 1)
split_index = int(len(X) * 0.8)

x_train = X[:split_index]
x_val = X[split_index:]
y_train = Y[:split_index]
y_val = Y[split_index:]

model = create_general_model()
model.fit(x_train, y_train, epochs=100, batch_size=32)





