from flask import Flask, request, render_template, redirect, url_for
from keras.models import load_model
import numpy as np
import cv2
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os

app = Flask(__name__)

# Define paths
DISCRIMINATOR_MODEL_PATH = 'discriminator_model.h5'
GENERATOR_MODEL_PATH = 'generator_model.h5'
PROFILE_CSV_PATH = 'fake_profiles.csv'
UPLOAD_FOLDER = 'uploads/'

# Create the uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the saved models
discriminator = load_model(DISCRIMINATOR_MODEL_PATH)
generator = load_model(GENERATOR_MODEL_PATH)

# Helper functions
def load_and_preprocess_image(image_path, img_shape):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.resize(img, (img_shape[1], img_shape[0]))
    img = (img / 127.5) - 1.0  # Normalize image to [-1, 1]
    return np.expand_dims(img, axis=0)  # Add batch dimension

def load_profile_info(csv_file, image_name):
    profiles = pd.read_csv(csv_file)
    profile_data = profiles[profiles['image_path'].str.contains(image_name.split('.')[0])]
    return profile_data.drop(columns=['image_path']).values

def classify_image(discriminator, image, profile_encoded):
    if profile_encoded is None:
        print("Cannot classify image without profile information.")
        return None
    profile_encoded = np.reshape(profile_encoded, (1, -1))  # Flatten the array
    expected_size = discriminator.input_shape[1][1]
    profile_encoded = np.pad(profile_encoded, [(0, 0), (0, expected_size - profile_encoded.shape[1])])
    prediction = discriminator.predict([image, profile_encoded])
    return prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        profile_data = request.form['profile_data']

        # Save the uploaded image
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)

        # Preprocess the image
        img_shape = (64, 64, 3)
        sample_image = load_and_preprocess_image(image_path, img_shape)

        # Load and preprocess the profile data
        sample_profile_info = load_profile_info(PROFILE_CSV_PATH, image.filename)
        if sample_profile_info.size == 0:
            return "No profile information found. Check if the image name is present in the CSV file."

        encoder = OneHotEncoder()
        sample_profile_encoded = encoder.fit_transform(sample_profile_info).toarray()

        # Classify the image
        prediction = classify_image(discriminator, sample_image, sample_profile_encoded)
        result = "Fake" if prediction > 0.5 else "Real"

        return render_template('result.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
