import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Define absolute paths to the models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISCRIMINATOR_MODEL_PATH = os.path.join(BASE_DIR, 'discriminator_model.h5')
GENERATOR_MODEL_PATH = os.path.join(BASE_DIR, 'generator_model.h5')

# Define a custom layer to handle the unrecognized arguments
class CustomConv2DTranspose(tf.keras.layers.Conv2DTranspose):
    def __init__(self, *args, groups=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.groups = groups

    def get_config(self):
        config = super().get_config()
        config.update({"groups": self.groups})
        return config

# Register the custom object
tf.keras.utils.get_custom_objects().update({'CustomConv2DTranspose': CustomConv2DTranspose})

# Function to load models with error handling
def load_models(generator_path, discriminator_path):
    try:
        # Load generator model
        custom_objects = {'Conv2DTranspose': CustomConv2DTranspose}
        generator = load_model(generator_path, custom_objects=custom_objects)
        print(f"Successfully loaded generator model from {generator_path}")
    except Exception as e:
        print(f"Error loading generator model: {e}")
        generator = None

    try:
        # Load discriminator model
        discriminator = load_model(discriminator_path)
        print(f"Successfully loaded discriminator model from {discriminator_path}")
    except Exception as e:
        print(f"Error loading discriminator model: {e}")
        discriminator = None

    return generator, discriminator

generator, discriminator = load_models(GENERATOR_MODEL_PATH, DISCRIMINATOR_MODEL_PATH)

if generator is None or discriminator is None:
    print("Failed to load models. Exiting...")
    exit(1)

# Home route
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        try:
            input_data = request.form['input_data']
            input_array = np.array(eval(input_data)).reshape((1, 100))  # Adjust reshape according to your input shape
            
            generated_data = generator.predict(input_array)
            result = {'generated_data': generated_data.tolist()}
            
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})
    return render_template('generate.html')

@app.route('/discriminate', methods=['GET', 'POST'])
def discriminate():
    if request.method == 'POST':
        try:
            file = request.files['file']
            img = Image.open(io.BytesIO(file.read()))
            img = img.resize((64, 64))
            img_array = np.array(img) / 255.0  # Normalize the image data to 0-1 range
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            discrimination_result = discriminator.predict(img_array)
            result = {'discrimination_result': discrimination_result.tolist()}
            
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})
    return render_template('discriminate.html')

if __name__ == '__main__':
    app.run(debug=True)
