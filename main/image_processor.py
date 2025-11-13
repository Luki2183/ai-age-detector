import cv2

def preprocess_image_or_frame(data, target_size=(200, 200), train_data=False):
    
    # Sprawdza potrzebe wczytania pliku
    if isinstance(data, str):
        img = cv2.imread(data)

    # Zmiana na skale szarości, tylko do trenowania
    if train_data:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Zmiana rozmiaru
    img = cv2.resize(img, target_size)

    # Normalizacja (0-1)
    img = img / 255.0

    return img