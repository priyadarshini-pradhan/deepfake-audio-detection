import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from keras.models import load_model

# -----------------------------
# Flask App Configuration
# -----------------------------
app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Load Trained Model
# -----------------------------
MODEL_PATH = "Models/deepfake_cnn.keras"

model = load_model(MODEL_PATH, compile=False)

print("Model Loaded Successfully!")
print("Output Shape:", model.output_shape)


# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "audio" not in request.files:
        return "No audio file uploaded."

    file = request.files["audio"]

    if file.filename == "":
        return "No file selected."

    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # -----------------------------
        # Read Audio
        # -----------------------------
        audio, sr = sf.read(filepath)

        # Convert stereo to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        audio = audio.astype(np.float32)

        # Keep only first 5 seconds
        max_length = sr * 5
        if len(audio) > max_length:
            audio = audio[:max_length]

        # -----------------------------
        # Save Waveform
        # -----------------------------
        plt.figure(figsize=(10, 3))
        plt.plot(audio)
        plt.title("Audio Waveform")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")

        waveform_path = os.path.join("static", "waveform.png")
        plt.savefig(waveform_path)
        plt.close()

        # -----------------------------
        # Feature Extraction
        # -----------------------------
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        )

        mfcc = np.mean(mfcc.T, axis=0)
        mfcc = mfcc.reshape(1, 40, 1)

        # -----------------------------
        # Prediction
        # -----------------------------
        prediction = model.predict(mfcc, verbose=0)

        # For Softmax Output (2 classes)
        label = np.argmax(prediction)
        confidence = round(float(np.max(prediction)) * 100, 2)

        if label == 0:
            result = "REAL AUDIO"
        else:
            result = "FAKE AUDIO"

        # -----------------------------
        # Display
        # -----------------------------
        audio_file = url_for(
            "static",
            filename="uploads/" + filename
        )

        return render_template(
            "index.html",
            prediction=result,
            confidence=confidence,
            audio_file=audio_file,
            waveform_image="waveform.png",
        )

    except Exception as e:
        return f"Error: {str(e)}"


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )