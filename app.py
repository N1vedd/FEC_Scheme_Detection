from flask import Flask, render_template, request, redirect
import numpy as np
from joblib import load
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
model = load_model('model.h5')

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        file = request.files['file']

        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            predictions = process_file(file_path)
            return redirect(f'/results/{file.filename}')

    return render_template('home.html')

def process_file(file_path):
    with open(file_path, 'r', errors='ignore') as file:
        demodulated_signals = file.readlines()

    predictions = []
    for signal in demodulated_signals:
        signal = signal.strip()
        prediction = predict_signal(signal)
        predictions.append({'demodulated_signal': signal, 'fec_scheme': prediction})

    return predictions

def predict_signal(signal):
    # Split the signal string into individual values
    signal_values = signal.split(',')

    # Check if all values in the signal are non-empty
    if all(signal_value.strip() for signal_value in signal_values):
        # Convert each value to float
        x = np.array([float(signal_value) for signal_value in signal_values])
        x = np.asarray(x).astype('float32')

        # Normalize the data
        x = (x - np.min(x)) / (np.max(x) - np.min(x))

        # Pad the sequence to match the model input shape
        x = np.pad(x, (0, 1800 - len(x)), mode='constant')
        
        # Reshape to match the model's input shape
        x = np.reshape(x, (1, -1))
        
        # Make predictions
        preds = model.predict(x)
        predicted_label = np.argmax(preds)
        
        # Map the predicted label to the FEC scheme
        inverse_mapping = {0: 'BCH', 1: 'CONVOLUTIONAL', 2: 'LDPC', 3: 'Turbo'}
        predicted_fec_scheme = inverse_mapping[predicted_label]

        return predicted_fec_scheme

    else:
        # Handle the case where the input signal is not valid
        return 'Invalid Signal'

@app.route('/results/<filename>')
def results(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    predictions = process_file(file_path)
    return render_template('results.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
