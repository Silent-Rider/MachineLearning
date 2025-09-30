from keras import models, layers
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from helper_utils import vectorize_sequences

def generate_data(token) -> tuple[np.ndarray, np.ndarray]:
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

    token.fit_on_texts(texts)
    data = vectorize_sequences(token.texts_to_sequences(texts), 100)
    labels = np.asarray([0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1]).astype('float32')
    return data, labels

tokenizer = Tokenizer()
train_data, train_labels = generate_data(tokenizer)

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(100,)))
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
    batch_size=3,
    validation_data=(x_val, y_val))

while True:
    review = input("Введите отзыв (или 'выход' для завершения): ")
    if review.lower() == 'выход':
        break
    seq = tokenizer.texts_to_sequences([review])
    vec = vectorize_sequences(seq, dimension=100)
    pred = model.predict(vec)[0][0]
    print(f"Оценка: {pred:.4f} → {'😊 Позитивный' if pred > 0.5 else '😠 Негативный'}\n")





