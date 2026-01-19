import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import age_prediction
import keras as kr
import time

"""
Main class used to predict age.
"""

model = kr.models.load_model('resources/models/age_estimation_model.keras')

camera_loop_id = None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def assign_age_and_convert_frame(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        # Required value assignment associated with UnboundLocalError
        # do NOT delete
        ages_list = []

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            age = -1
            age = age_prediction.predict_age(model, frame)
            ages_list.append(int(age))
            label = "Age: " + str(int(age))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, label, (x, y - 10), font, 0.8, (255, 0, 0), 2)
        return Image.fromarray(frame), ages_list


def convert_ages_list_to_str(ages_list: list):
    length_of_ages_list = len(ages_list)
    if length_of_ages_list < 1:
        return f'Age: -1'
    elif length_of_ages_list == 1:
        return f'Age: {ages_list.pop()}'
    else:
        return f'Ages: {str(ages_list)[1:-1:]}'


def open_camera():
    def update_frame():
        global current_img, camera_loop_id
        ret, frame = cam.read()
        
        img, ages_list = assign_age_and_convert_frame(frame)
        
        # Lower fps
        time.sleep(0.1)

        current_img = img
        current_age = convert_ages_list_to_str(ages_list)
        age_panel.config(text=current_age)
        age_panel.text = current_age
        resize_image()
        camera_loop_id = img_panel.after(10, update_frame)

    release_cam()

    global cam
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot access camera")
        return
    update_frame()


def open_picture():
    global current_img, age_panel

    release_cam()

    filepath = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    if not filepath:
        return
    
    frame = cv2.imread(filepath)

    img, ages_list = assign_age_and_convert_frame(frame)

    current_img = img
    current_age = convert_ages_list_to_str(ages_list)
    age_panel.config(text=current_age)
    age_panel.text = current_age
    root.update_idletasks()
    resize_image()


def release_cam():
    global cam, camera_loop_id
    if camera_loop_id:
        img_panel.after_cancel(camera_loop_id)
        camera_loop_id = None
    
    try:
        cam.release()
    except:
        pass


root = tk.Tk()
root.title("Age prediction")
root.geometry("600x500")


current_img = None


def resize_image(event=None):
    if current_img is None:
        return
    w, h = img_panel.winfo_width(), img_panel.winfo_height()
    resized = current_img.resize((max(1, w-50), max(1, h-50)))
    img_tk = ImageTk.PhotoImage(resized)
    img_panel.config(image=img_tk)
    img_panel.image = img_tk


btn_cam = tk.Button(root, text="Open Camera", command=open_camera)
btn_cam.pack(pady=10)


btn_pic = tk.Button(root, text="Open Picture", command=open_picture)
btn_pic.pack(pady=10)


age_panel = tk.Label(root, text = 'Age: ')
age_panel.pack(pady=10)


img_panel = tk.Label(root)
img_panel.pack(expand=True, fill="both")


root.bind("<Configure>", resize_image)


root.mainloop()