
# ===============================================
# KOMPLETNY SYSTEM: DETEKCJA TWARZY + ESTYMACJA WIEKU
# ===============================================

import cv2
import numpy as np
from tensorflow.python.keras.layers import load_model

# ===============================================
# CZĘŚĆ 1: Detekcja twarzy używając OpenCV
# ===============================================

def detect_faces(image_path, face_cascade_path='haarcascade_frontalface_default.xml'):
    # Wczytaj kaskadę Haara do detekcji twarzy
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + face_cascade_path
    )

    # Wczytaj obraz
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Wykryj twarze
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    return img, faces

# ===============================================
# CZĘŚĆ 2: Przygotowanie twarzy do predykcji
# ===============================================

def preprocess_face(face_img, target_size=(200, 200)):
    # Zmień rozmiar
    face_resized = cv2.resize(face_img, target_size)

    # Normalizacja
    face_normalized = face_resized / 255.0

    # Dodaj wymiar batch
    face_batch = np.expand_dims(face_normalized, axis=0)

    return face_batch

# ===============================================
# CZĘŚĆ 3: System predykcji wieku
# ===============================================

class AgeEstimationSystem:
    def __init__(self, model_path=None):
        if model_path:
            self.model = load_model(model_path)
        else:
            # Stwórz nowy model (jeśli nie ma wytrenowanego)
            self.model = self.create_model()

    def create_model(self):
        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
            MaxPooling2D((2, 2)),

            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),

            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),

            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='linear')
        ])

        model.compile(
            optimizer='adam',
            loss='mean_absolute_error',
            metrics=['mae']
        )

        return model

    def predict_age(self, face_image):
        # Przygotuj twarz
        face_processed = preprocess_face(face_image)

        # Predykcja
        age = self.model.predict(face_processed, verbose=0)[0][0]

        return int(age)

    def process_image(self, image_path, output_path=None):
        # Wykryj twarze
        img, faces = detect_faces(image_path)

        results = []

        # Dla każdej wykrytej twarzy
        for (x, y, w, h) in faces:
            # Wytnij twarz
            face = img[y:y+h, x:x+w]

            # Przewiduj wiek
            age = self.predict_age(face)
            results.append({'bbox': (x, y, w, h), 'age': age})

            # Rysuj ramkę i wiek na obrazie
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Dodaj tekst z wiekiem
            text = f'Wiek: {age}'
            cv2.putText(
                img, text, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0), 2
            )

        # Zapisz obraz wynikowy
        if output_path:
            cv2.imwrite(output_path, img)

        return img, results

# ===============================================
# CZĘŚĆ 4: Przykładowe użycie
# ===============================================

# Inicjalizacja systemu
# system = AgeEstimationSystem(model_path='age_model.h5')

# Przetwarzanie pojedynczego obrazu
# result_img, predictions = system.process_image(
#     'input_image.jpg',
#     output_path='output_image.jpg'
# )

# Wyświetl wyniki
# for i, pred in enumerate(predictions):
#     print(f'Twarz {i+1}: Przewidywany wiek = {pred["age"]} lat')

# ===============================================
# CZĘŚĆ 5: Przetwarzanie wideo (opcjonalne)
# ===============================================

def process_video(video_path, model_path, output_path='output_video.mp4'):
    # Załaduj model
    system = AgeEstimationSystem(model_path)

    # Otwórz wideo
    cap = cv2.VideoCapture(video_path)

    # Pobierz właściwości wideo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Stwórz writer dla wideo wyjściowego
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Wykryj twarze w klatce
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        # Przewiduj wiek dla każdej twarzy
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            age = system.predict_age(face)

            # Rysuj wyniki
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame, f'Wiek: {age}', (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )

        # Zapisz klatkę
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Użycie:
# process_video('input_video.mp4', 'age_model.h5', 'output_video.mp4')
