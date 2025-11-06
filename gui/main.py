import cv2
import keras as kr
import time
import numpy as np

model = kr.models.load_model('models/age_estimation_model.keras')

def load_and_preprocess_image(frame, target_size=(200, 200)):
    # Zmiana rozmiaru
    img = cv2.resize(frame, target_size)

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

# Open the default camera
cam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    for (x, y, w, h) in faces:
    #     face = frame[y:y+h, x:x+w]
    #     cv2.imshow('Detected Face', face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        age = predict_age(model, frame)
        label = "Age: " + str(int(age))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, label, (x, y - 10), font, 0.8, (255, 0, 0), 2)
    
    cv2.imshow('Webcam', frame)
    
    # Less fps
    # time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()