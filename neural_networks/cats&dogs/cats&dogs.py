from keras import models, layers
from keras.src.legacy.preprocessing.image import ImageDataGenerator, DirectoryIterator

from helper_utils import draw_plots

data_dir:str = 'dataset'
img_height, img_width = 200, 200

datagen = ImageDataGenerator(
    rescale=1/255,               # Нормализация: пиксели [0,255] → [0,1]
    rotation_range=20,            # Поворот на ±20 градусов
    width_shift_range=0.2,        # Сдвиг по горизонтали на 20% ширины
    height_shift_range=0.2,       # Сдвиг по вертикали на 20% высоты
    shear_range=0.2,              # "Срез" изображения (как перекос)
    zoom_range=0.2,               # Зум ±20%
    horizontal_flip=True,         # Зеркальное отражение по горизонтали
    validation_split=0.2          # 20% данных выделить под валидацию
)

train_generator = datagen.flow_from_directory(
    directory=data_dir,          # корневая папка
    target_size=(img_height, img_width),       # все фото ресайзятся до этого размера
    batch_size=16,                # сколько фото в одном батче
    class_mode='binary',          # бинарная классификация → метки [0, 1]
    subset='training'             # брать из той части, что НЕ валидация
)

val_generator = datagen.flow_from_directory(
    directory=data_dir,
    target_size=(img_height, img_width),
    batch_size=16,
    class_mode='binary',
    subset='validation'           # брать 20% (как указано выше)
)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator
)

draw_plots(history)