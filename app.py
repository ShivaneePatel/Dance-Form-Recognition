import os
import cv2
from keras.models import load_model
import numpy as np
model = load_model('dance_classification_model.h5')
dataset_path = 'dataset/train'
class_names = os.listdir(dataset_path)

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def predict_dance_style(video_path, model_path):
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Initialize variables
    frame_count = 0
    predictions = []
    
    # Loop over frames
    while cap.isOpened():
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame
        frame = cv2.resize(frame, (128, 128))
        frame = img_to_array(frame)
        frame = preprocess_input(frame)
        frame = np.expand_dims(frame, axis=0)
        
        # Make prediction
        pred = model.predict(frame)[0]
        predicted_class = np.argmax(pred)
        if pred[predicted_class] < 0.5:
            predictions.append(len(class_names))
        else:
            predictions.append(predicted_class)
        
        # Increment frame count
        frame_count += 1
    
    # Close video file
    cap.release()
    
    # Get the mode prediction
    mode_prediction = max(predictions, key=predictions.count)
    if mode_prediction == len(class_names):
        return "No dance style matches"
    else:
        return class_names[mode_prediction]


"""Flask Server Backend"""

from flask import Flask, request, jsonify,render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/predict', methods=['POST'])
def predict():
    videoFile = request.files['video']
    filename = secure_filename(videoFile.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    videoFile.save(filepath)
    
    predicted_style = predict_dance_style(filepath, 'dance_classification_model.h5')
    
    return render_template('index.html', prediction=predicted_style)

if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=True)