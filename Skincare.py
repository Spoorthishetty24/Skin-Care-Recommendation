import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Button, filedialog
from keras.models import load_model


def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around the face
        face = image[y:y + h, x:x + w]
        return face, image
    return None, image


def analyze_skin(face_image):
    model = load_model('models/skin_analysis_model.h5')
    face_resized = cv2.resize(face_image, (128, 128))
    face_resized = face_resized / 255.0
    face_resized = np.expand_dims(face_resized, axis=0)

    predictions = model.predict(face_resized)
    skin_types = ['Oily', 'Dry', 'Normal', 'Combination']
    return skin_types[np.argmax(predictions)]


def recommend_products(skin_type):
    recommendations = {
        "Oily": ["Oil-free moisturizer", "Salicylic acid cleanser"],
        "Dry": ["Hydrating cream", "Hyaluronic acid serum"],
        "Normal": ["Gentle cleanser", "SPF 50 sunscreen"],
        "Combination": ["Balancing toner", "Lightweight moisturizer"]
    }
    return recommendations.get(skin_type, [])


def upload_and_process():
    file_path = filedialog.askopenfilename()
    image = cv2.imread(file_path)
    face, processed_image = detect_face(image)
    if face is not None:
        skin_type = analyze_skin(face)
        products = recommend_products(skin_type)
        cv2.imshow("Detected Face", processed_image)  # Show detected face with a box
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        result_label.config(text=f"Detected Skin Type: {skin_type}\nRecommended Products: {', '.join(products)}")
    else:
        result_label.config(text="No face detected. Please try another image.")


def capture_from_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face, processed_frame = detect_face(frame)
        cv2.imshow("Live Camera - Detecting Face", processed_frame)

        if face is not None:
            cap.release()
            cv2.destroyAllWindows()
            skin_type = analyze_skin(face)
            products = recommend_products(skin_type)
            result_label.config(text=f"Detected Skin Type: {skin_type}\nRecommended Products: {', '.join(products)}")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


# GUI Setup
root = tk.Tk()
root.title("Skin Care Recommender")

Label(root, text="Upload Your Face Image or Use Live Camera").pack()
Button(root, text="Upload Image", command=upload_and_process).pack()
Button(root, text="Use Live Camera", command=capture_from_camera).pack()
result_label = Label(root, text="")
result_label.pack()

root.mainloop()
