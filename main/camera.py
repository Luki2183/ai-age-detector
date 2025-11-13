import cv2
import keras as kr
import time
import age_prediction

model = kr.models.load_model('resources/models/age_estimation_model.keras')

cam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        age = age_prediction.predict_age(model, frame)
        label = "Wiek: " + str(int(age))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, label, (x, y - 10), font, 0.8, (255, 0, 0), 2)
    
    cv2.imshow('Kamera', frame)
    
    # Mniejsza ilość fps
    # time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()