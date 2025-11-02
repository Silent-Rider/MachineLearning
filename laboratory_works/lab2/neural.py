import numpy as np
from keras import Sequential, layers
from numpy import ndarray

from helper_utils import draw_plots


def generate_dataset() -> tuple:
    data = []
    labels = []
    for a in [0, 1]:
        for b in [0, 1]:
            for c in [0, 1]:
                x = (a, b, c)
                data.append(x)
                y = int((a or b) != (not(b and c)))
                labels.append(y)

    print("\tx\t\t|\ty")
    for i in range(len(data)):
        print(str(data[i]) + "\t|\t" + str(labels[i]))
    return np.asarray(data, 'int8'), np.asarray(labels, 'int8')

def keras_neural(x, y):
    model = Sequential()
    model.add(layers.Dense(4, activation='relu', input_shape=(x.shape[1], )))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    hist_obj = model.fit(x, y, epochs=50)

    weights = model.weights
    print(weights)

x, y = generate_dataset()
keras_neural(x, y)


def manual_model(x, weights):
    pass

def numpy_model(x, weights):
    pass



