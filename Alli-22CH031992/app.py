# app.py
from flask import Response
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import sqlite3
from datetime import datetime
import cv2
import urllib.request

app = Flask(__name__)
camera = cv2.VideoCapture(0)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "emotion_detection_model.h5"
model = load_model(MODEL_PATH)

# Load the model
model = load_model(MODEL_PATH)

# Initialize DB
def init_db():
    conn = sqlite3.connect("database.db")
    conn.execute('''CREATE TABLE IF NOT EXISTS uploads
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     filename TEXT,
                     emotion TEXT,
                     upload_time TEXT)''')
    conn.close()

init_db()

# Emotion labels (FER2013)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Resize for model
            resized = cv2.resize(gray, (48, 48))
            img = resized.reshape(1, 48, 48, 1) / 255.0

            # Predict emotion
            prediction = model.predict(img, verbose=0)
            emotion = EMOTIONS[np.argmax(prediction)]

            # Overlay text
            cv2.putText(frame, emotion, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 3, cv2.LINE_AA)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Stream frame to browser
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/admin')
def admin():
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    cur.execute("SELECT filename, emotion, upload_time FROM uploads ORDER BY id DESC")
    records = cur.fetchall()
    conn.close()
    return render_template('admin.html', records=records)


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file:
        return render_template('index.html', emotion="No image uploaded.")

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Preprocess
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.reshape(1, 48, 48, 1) / 255.0

    prediction = model.predict(img)
    emotion = EMOTIONS[np.argmax(prediction)]

    # Save record
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    cur.execute("INSERT INTO uploads (filename, emotion, upload_time) VALUES (?, ?, ?)",
                (file.filename, emotion, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

    return render_template('index.html', filename=file.filename, emotion=emotion)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
