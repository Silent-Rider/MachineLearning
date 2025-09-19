from keras import models, layers
from keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32', copy=False)
y_test = np.asarray(test_labels).astype('float32', copy=False)

tokenizer = Tokenizer()
texts = ["Не люблю такое",
         "Мне нравится фильм",
         "Мне не нравится фильм",
         "Я ненавижу этот фильм",
         "Я обожаю этот фильм",
         "Фильм лучший",
         "Мне очень нравится этот фильм",
         "Я люблю этот фильм",
         "Фильм ужасен",
         "Хорошо",
         "Фильм плох",
         "Фильм шикарный",
         "Ужасно плохой фильм",
         "Интересный фильм",
         "Интересный",
         "Интересный сценарий",
         "Нравится подобный жанр",
         "Ненавижу подобное",
         "Фильм хорош"]

tokenizer.fit_on_texts(texts)
train_data = vectorize_sequences(tokenizer.texts_to_sequences(texts))
train_labels = np.asarray([0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1]).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

x_val = train_data[:1]           # 1 пример на валидацию
partial_x_train = train_data[1:] # 18 примера на обучение
y_val = train_labels[:1]
partial_y_train = train_labels[1:]

history = model.fit(partial_x_train,
    partial_y_train,
    epochs=200,
    batch_size=len(train_data),
    validation_data=(x_val, y_val))

print("\n--- ТЕСТИРУЕМ МОДЕЛЬ ---")
while True:
    review = input("Введите отзыв (или 'выход' для завершения): ")
    if review.lower() == 'выход':
        break
    seq = tokenizer.texts_to_sequences([review])
    vec = vectorize_sequences(seq, dimension=10000)
    pred = model.predict(vec)[0][0]
    print(f"Оценка: {pred:.4f} → {'😊 Позитивный' if pred > 0.5 else '😠 Негативный'}\n")





