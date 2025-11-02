from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
MODEL_PATH = "emotion_detection_model.h5"
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_webcam', methods=['POST'])
def predict_webcam():
    """Receive webcam frame in base64, predict emotion, and return JSON"""
    data_url = request.json['image']
    img_data = data_url.split(',')[1]
    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    img = cv2.resize(frame, (48, 48))
    img = img.reshape(1, 48, 48, 1) / 255.0
    
    prediction = model.predict(img, verbose=0)
    emotion = EMOTIONS[np.argmax(prediction)]
    return jsonify({'emotion': emotion})

@app.route('/predict', methods=['POST'])
def predict():
    """Handle uploaded image file"""
    file = request.files.get('image')
    if not file:
        return render_template('index.html', emotion="No image uploaded.")
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Preprocess image
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.reshape(1, 48, 48, 1) / 255.0
    
    prediction = model.predict(img, verbose=0)
    emotion = EMOTIONS[np.argmax(prediction)]

    # Save record in DB
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    cur.execute("INSERT INTO uploads (filename, emotion, upload_time) VALUES (?, ?, ?)",
                (file.filename, emotion, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

    return render_template('index.html', filename=file.filename, emotion=emotion)

@app.route('/admin')
def admin():
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()
    cur.execute("SELECT filename, emotion, upload_time FROM uploads ORDER BY id DESC")
    records = cur.fetchall()
    conn.close()
    return render_template('admin.html', records=records)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
