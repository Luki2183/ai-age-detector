import cv2

def preprocess_image_or_frame(data, target_size=(200, 200), train_data=False):
    """
    Funkcja używana do przygotowania zdjęcia, zmiany rozmiaru, normalizacja itp.
    :param data: ścieżka do pliku lub MatLike obiekt zwrócony przez cv2.imread()
    
    :param target_size: tuple z rozmiarem do którego będziemy zmieniać zdjęcie
    :param train_data: wartość logiczna służąca przygotowaniu danych do trenowania(zmiana na skale szarości)

    Returns:
        MatLike: gotowy obiekt do trenowania/predykcji
    """

    # Sprawdza potrzebe wczytania pliku
    if isinstance(data, str):
        img = cv2.imread(data)
    else:
        img = data

    # Zmiana na skale szarości, tylko do trenowania
    if train_data:
        img = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

    # Zmiana rozmiaru
    img = cv2.resize(img, target_size)

    # Normalizacja (0-1)
    img = img / 255.0

    return img