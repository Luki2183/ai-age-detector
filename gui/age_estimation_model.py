
# Import niezbędnych bibliotek
import tensorflow as tf
from tensorflow.python import keras
from keras import layers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import cv2
import os
import numpy as np

# ===============================================
# CZĘŚĆ 1: Przygotowanie danych
# ===============================================

# Funkcja do ładowania i przetwarzania zdjęć
def load_and_preprocess_image(image_path, target_size=(200, 200)):
    # Wczytaj zdjęcie
    img = cv2.imread(image_path)

    # Konwersja do skali szarości (opcjonalne)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Zmiana rozmiaru
    img = cv2.resize(img, target_size)

    # Normalizacja (0-1)
    img = img / 255.0

    return img

# ===============================================
# CZĘŚĆ 2: Budowa modelu CNN
# ===============================================

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

# Stworzenie modelu
model = create_age_estimation_model()

# ===============================================
# CZĘŚĆ 3: Kompilacja modelu
# ===============================================

# Dla regresji używamy Mean Absolute Error (MAE) jako funkcji straty
model.compile(
    optimizer='adam',
    loss='mean_absolute_error',
    metrics=['mae']
)

# Podsumowanie architektury
model.summary()

# ===============================================
# CZĘŚĆ 4: Trening modelu (przykład)
# ===============================================

# Przykładowe dane treningowe (musisz załadować własne dane)
# X_train - zdjęcia (numpy array o kształcie (n_samples, 200, 200, 3))
# y_train - wiek (numpy array o kształcie (n_samples,))


def load_utkface_data(data_dir, img_size=200):
    images = []
    ages = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg'):
            age = int(filename.split('_')[0])  # Pobierz wiek z nazwy pliku
            img_path = os.path.join(data_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_size, img_size))
            img = img / 255.0
            images.append(img)
            ages.append(age)
    return np.array(images), np.array(ages)

# Wczytaj całą bibliotekę (np. UTKFace)
X_train, y_train = load_utkface_data('data/UTKFace', img_size=200)


history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        # keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
)

model.save('models/age_estimation_model50e.keras')