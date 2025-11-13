import image_processor
import numpy as np

def predict_age(model, image_path):
    # Wczytaj i przetwórz zdjęcie
    img = image_processor.preprocess_image_or_frame(image_path)

    # Dodanie kolumny z powodu błedu
    img = np.expand_dims(img, axis=0)

    # Predykcja
    predicted_age = model.predict(img)[0][0]

    return predicted_age