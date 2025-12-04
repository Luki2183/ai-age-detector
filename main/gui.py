import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2

camera_loop_id = None

def open_camera():
    def update_frame():
        global current_img, camera_loop_id
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_img = Image.fromarray(frame)
            resize_image()
        camera_loop_id = panel.after(10, update_frame)
        
    release_cam()

    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera")
        return
    update_frame()


def open_picture():
    global current_img

    release_cam()

    filepath = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    if not filepath:
        return
    current_img = Image.open(filepath)
    root.update_idletasks()
    resize_image()

def release_cam():
    global cap, camera_loop_id
    if camera_loop_id:
        panel.after_cancel(camera_loop_id)
        camera_loop_id = None
    
    try:
        cap.release()
    except:
        pass

root = tk.Tk()
root.title("Age prediction")
root.geometry("600x500")


current_img = None


def resize_image(event=None):
    if current_img is None:
        return
    w, h = panel.winfo_width(), panel.winfo_height()
    resized = current_img.resize((max(1, w-50), max(1, h-50)))
    img_tk = ImageTk.PhotoImage(resized)
    panel.config(image=img_tk)
    panel.image = img_tk


btn_cam = tk.Button(root, text="Open Camera", command=open_camera)
btn_cam.pack(pady=10)


btn_pic = tk.Button(root, text="Open Picture", command=open_picture)
btn_pic.pack(pady=10)


panel = tk.Label(root)
panel.pack(expand=True, fill="both")


root.bind("<Configure>", resize_image)


root.mainloop()