import tensorflow as tf
import keras as kr
from tensorflow.python import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import cv2
import os
import numpy as np

model = kr.models.load_model('models/age_estimation_model.keras')

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

def predict_age(model, image_path):
    # Wczytaj i przetwórz zdjęcie
    img = load_and_preprocess_image(image_path)

    # Dodaj wymiar batch
    img = np.expand_dims(img, axis=0)

    # Predykcja
    predicted_age = model.predict(img)[0][0]

    return predicted_age

# Przykładowe użycie:

data_dir = 'data/tomy'

for filename in os.listdir(data_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(data_dir, filename)
            age = predict_age(model, img_path)
            print(f'Przewidywany wiek: {age:.1f} lat zdjęcia {filename}')

