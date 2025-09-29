import random
from keras import models, layers
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from helper_utils import vectorize_sequences


def readFile(filepath, text_labels:dict):
    sentences: list[str] = []
    labels: list[int] = []
    current_genre = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("[") and line.endswith("]"):
                genre_name = line[1:-1]
                if genre_name in text_labels:
                    current_genre = genre_name
                else:
                    raise ValueError(f"Неизвестный жанр: {genre_name}")
            else:
                if current_genre is None:
                    raise ValueError("Предложение до объявления жанра!")
                line = line.replace(" он ", " ")
                line = line.replace(" я ", " ")
                line = line.replace(" она ", " ")
                line = line.replace(" они ", " ")
                line = line.replace(" его ", " ")
                line = line.replace(" меня ", " ")
                line = line.replace(" её ", " ")
                line = line.replace(" ее ", " ")
                line = line.replace(" их ", " ")
                line = line.replace(" ему ", " ")
                line = line.replace(" мне ", " ")
                line = line.replace(" ей ", " ")
                line = line.replace(" им ", " ")
                line = line.replace(" о нем ", " ")
                line = line.replace(" обо мне ", " ")
                line = line.replace(" о нём ", " ")
                line = line.replace(" о ней ", " ")

                sentences.append(line)
                labels.append(text_labels[current_genre])
    return sentences, labels


def generate_data(filepath) -> tuple[list[str], list[int]]:
    genre_to_label:dict = {
        "ужасы": 0,
        "комедия": 1,
        "фантастика": 2,
        "драма": 3,
        "боевик": 4
    }
    sentences, labels = readFile(filepath, genre_to_label)

    combined = list(zip(sentences, labels))
    random.seed(42)
    random.shuffle(combined)
    sentences, labels = zip(*combined)

    return sentences, labels


tokenizer = Tokenizer()
(raw_data, raw_labels) = generate_data("movie_genres.txt")
tokenizer.fit_on_texts(raw_data)
train_data = vectorize_sequences(tokenizer.texts_to_sequences(raw_data), 1000)
train_labels = np.asarray(raw_labels).astype('float32')

x_val = train_data[:10]
x_train = train_data[10:]
y_val = train_labels[:10]
y_train = train_labels[10:]

model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(1000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    x_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_data=(x_val, y_val)
)

while True:
    review = input("Введите отзыв к фильму (или 'выход' для завершения): ")
    if review.lower() == 'выход':
        break
    seq = tokenizer.texts_to_sequences([review])
    vec = vectorize_sequences(seq, dimension=1000)

    prediction = model.predict(vec)
    genre_id = np.argmax(prediction[0])

    genre:str = ''
    if 0 <= genre_id < 5:
        match genre_id:
            case 0: genre = '😱 Ужасы'
            case 1: genre = '😂 Комедия'
            case 2: genre = '🚀 Фантастика'
            case 3: genre = '💔 Драма'
            case 4: genre = '💥 Боевик'

    print(f"Жанр: {genre} Уверенность в ответе: {np.max(prediction[0]):.4f}\n")