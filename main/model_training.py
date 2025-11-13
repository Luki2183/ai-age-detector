from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os
import numpy as np
import image_processor

def create_age_estimation_model(input_shape=(200, 200, 3)):
    model = Sequential([
        # Pierwsza warstwa konwolucyjna
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),

        # Druga warstwa konwolucyjna
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Trzecia warstwa konwolucyjna
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Czwarta warstwa konwolucyjna
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Spłaszczenie
        Flatten(),

        # Warstwy w pełni połączone
        Dense(512, activation='relu'),
        Dropout(0.5),

        Dense(256, activation='relu'),
        Dropout(0.3),

        # Warstwa wyjściowa - REGRESJA (pojedyncza wartość wieku)
        Dense(1, activation='linear')
    ])

    return model

model = create_age_estimation_model()

model.compile(
    optimizer='adam',
    loss='mean_absolute_error',
    metrics=['mae']
)
model.summary()

# Przykładowe dane treningowe
# X_train - zdjęcia (numpy array o kształcie (n_samples, 200, 200, 3))
# Y_train - wiek (numpy array o kształcie (n_samples,))


def load_utkface_data(data_dir):
    images = []
    ages = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg'):
            age = int(filename.split('_')[0])  # Pobierz wiek z nazwy pliku
            img_path = os.path.join(data_dir, filename)
            img = image_processor.preprocess_image_or_frame(img_path, train_data=True)
            images.append(img)
            ages.append(age)
    return np.array(images), np.array(ages)

X_train, y_train = load_utkface_data('resources/data/UTKFace')

# todo
# validation_data, przygotować dane do walidacji odpowiednio dla przedziałów
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]
)

model.save('resources/models/age_estimation_model_intervals.keras')