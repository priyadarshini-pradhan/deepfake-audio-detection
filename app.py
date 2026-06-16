import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
# Ensure upload folder exists
os.makedirs("static/uploads", exist_ok=True)

# Load trained model
model = load_model("Models/deepfake_cnn.h5")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['audio']

    # Save file inside static folder
    filename = secure_filename(file.filename)

    filepath = os.path.join("static/uploads", filename)

    file.save(filepath)

    # Feature Extraction
    audio, sr = librosa.load(filepath, sr=16000, duration=5)
    plt.figure(figsize=(10,3))
    plt.plot(audio)
    plt.title("Audio Waveform")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")

    waveform_path = os.path.join("static", "waveform.png")

    plt.savefig(waveform_path)
    plt.close()

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    mfcc = mfcc.reshape(1, 40, 1)

    # Prediction
    prediction = model.predict(mfcc)

    label = np.argmax(prediction)
    confidence = round(float(np.max(prediction)) * 100, 2)

    result = "REAL AUDIO" if label == 0 else "FAKE AUDIO"

    # Create URL for audio playback
    audio_file = url_for('static', filename='uploads/' + filename)

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        audio_file=audio_file,
        waveform_image="waveform.png"
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )