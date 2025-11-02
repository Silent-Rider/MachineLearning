import math

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


def manual_relu(x):
    x = x.copy()
    for i in range(len(x)):
        x[i] = max(x[i], 0)
    return x


def manual_sigmoid(x):
    return 1/(1 + (math.e ** -x))


def keras_model(x) -> Sequential:
    model = Sequential()
    model.add(layers.Dense(4, activation='relu', input_shape=(x.shape[1], )))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def manual_model(x, w1, b1, w2, b2):
    result = []
    for k in range(len(x)):
        data = x[k]
        #1 Dense слой
        hidden = []
        for i in range(len(w1)):
            neuron = 0.0
            for j in range(len(w1[0])):
                neuron += w1[i][j] * data[j]
            neuron += b1[i]
            hidden.append(neuron)
        hidden = manual_relu(hidden)

        # 2 Dense слой
        output = 0.0
        for i in range(len(w2)):
            output += w2[i] * hidden[i]
        output += b2

        result.append(manual_sigmoid(float(output)))
    return result


def numpy_model(x, w1, b1, w2, b2):
    hidden = x @ w1 + b1
    hidden = np.maximum(0.0, hidden)

    output = hidden @ w2 + b2
    output = 1.0 / (1.0 + np.exp(-output))
    return output.flatten()


def extract_weights(model:Sequential) -> tuple:
    w1, b1 = model.layers[0].get_weights()
    w2, b2 = model.layers[1].get_weights()
    print("Веса нейронов первого слоя: \n", w1.T)  # транспонирую, потому что в Keras веса хранятся в матрице вида (число входных признаков, число нейронов)
    print("Смещения нейронов первого слоя: \n", b1)
    print("Веса нейрона второго слоя: \n", w2[:, 0])
    print("Смещение нейрона второго слоя: \n", b2)
    return (w1, b1), (w2, b2)


x_data, y_labels = generate_dataset()
keras_model = keras_model(x_data)

# До обучения
print("До обучения модели")
(weight1, bias1), (weight2, bias2) = extract_weights(keras_model)

manual_model = manual_model(x_data, weight1.T, bias1, weight2[:, 0], bias2)
numpy_model = numpy_model(x_data, weight1, bias1, weight2, bias2)

print(keras_model.predict(x_data)[:, 0])
print(manual_model)
print(numpy_model)

#hist_obj = keras_model.fit(x_data, y_labels, epochs=50)

