from keras import models, layers
from keras.src.datasets import reuters
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
from helper_utils import vectorize_sequences
from helper_utils import draw_plots

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# def to_one_hot(labels, dimension=46):
#     results = np.zeros((len(labels), dimension))
#     for i, label in enumerate(labels):
#         results[i, label] = 1
#     return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# one_hot_train_labels = to_one_hot(train_labels)
# one_hot_test_labels = to_categorical(test_labels)
one_hot_train_labels = np.asarray(train_labels, dtype='float32')
one_hot_test_labels = np.asarray(test_labels, dtype='float32')

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              # loss='categorical_crossentropy',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=9,
    batch_size=512,
    validation_data=(x_val, y_val),
)

history_dict = history.history
draw_plots(history_dict)

print(model.evaluate(x_test, one_hot_test_labels))

predictions = model.predict(x_test)
print(predictions[0].shape)
#(46,) #размер выходного вектора

print(np.sum(predictions[0]))
#1.0 #сумма вероятностей, из-за погрешности не ровно 1

print(np.argmax(predictions[0]))
#3 #к какому классу отнесено наблюдение

print(np.max(predictions[0]))
#0.8662524 #с какой вероятностью отнесено к классу

