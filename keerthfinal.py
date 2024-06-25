# -*- coding: utf-8 -*-
"""KeerthFinal.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jWzer5Yc0RFP75S2RjeynhnUXrP4jnK7
"""


import os
import csv
import random
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from faker import Faker
from lorem_text import lorem
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, Flatten, Dense, Dropout
from keras.layers import BatchNormalization, Reshape, Conv2DTranspose
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from google.colab import drive

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
img_rows = 64
img_cols = 64
channels = 3
img_shape = (img_rows, img_cols, channels)
noise_dim = 100

def load_images_and_profiles(image_folder, csv_file, img_shape):
    profiles = pd.read_csv(csv_file)
    images = []
    labels = []
    profile_infos = []

    for _, row in profiles.iterrows():
        img_path = row['image_path']
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load image: {img_path}")
            continue
        img = cv2.resize(img, (img_shape[1], img_shape[0]))
        images.append(img)
        profile_infos.append(row.drop(['image_path']).values)
        labels.append(0 if 'real' in img_path else 1)

    images = np.array(images) / 127.5 - 1.0
    labels = np.array(labels)
    profile_infos = np.array(profile_infos)
    return images, labels, profile_infos

# Paths to your folders and CSV files
fake_csv_file = '/content/drive/MyDrive/Fake Profile Dectection/fake_profiles.csv'
real_csv_file = '/content/drive/MyDrive/Fake Profile Dectection/real_profiles.csv'
fake_image_folder = '/content/drive/MyDrive/Fake Profile Dectection/fake'
real_image_folder = '/content/drive/MyDrive/Fake Profile Dectection/real'
# Build the Discriminator
def build_discriminator(profile_input_shape):
    img_input = tf.keras.Input(shape=img_shape)
    profile_input = tf.keras.Input(shape=profile_input_shape)

    x = Conv2D(64, kernel_size=3, strides=2, padding="same")(img_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Conv2D(256, kernel_size=3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Conv2D(512, kernel_size=3, strides=1, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)

    profile_features = Dense(x.shape[1])(profile_input)  # Match the dimensionality of the flattened image data
    profile_features = LeakyReLU(alpha=0.2)(profile_features)
    profile_features = Dropout(0.25)(profile_features)

    combined_features = tf.keras.layers.Concatenate()([x, profile_features])
    combined_features = Dense(512)(combined_features)
    combined_features = LeakyReLU(alpha=0.2)(combined_features)
    combined_features = Dropout(0.25)(combined_features)

    output = Dense(1, activation='sigmoid')(combined_features)

    model = tf.keras.Model([img_input, profile_input], output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    return model

# Build the Generator
def build_generator():
    model = Sequential([
        Dense(128 * 16 * 16, activation="relu", input_dim=noise_dim),
        Reshape((16, 16, 128)),
        BatchNormalization(momentum=0.8),
        Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),
        Conv2DTranspose(3, kernel_size=4, strides=1, padding="same", activation='tanh')
    ])
    return model

# Mount Google Drive
drive.mount('/content/drive')
data_path = '/content/drive/My Drive/Fake Profile Dectection'
# Load datasets
real_images, real_labels, real_profiles = load_images_and_profiles(real_image_folder, real_csv_file, img_shape)
fake_images, fake_labels, fake_profiles = load_images_and_profiles(fake_image_folder, fake_csv_file, img_shape)

# Combine datasets
images = np.concatenate((real_images, fake_images))
labels = np.concatenate((real_labels, fake_labels))
profiles = np.concatenate((real_profiles, fake_profiles))

# Convert profile data to numerical representation (example using one-hot encoding)
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
profiles_encoded = encoder.fit_transform(profiles).toarray()

# Split into training and testing sets
x_train, x_test, y_train, y_test, profile_train_encoded, profile_test_encoded = train_test_split(
    images, labels, profiles_encoded, test_size=0.2, random_state=42
)

discriminator = build_discriminator(profile_train_encoded.shape[1:])
generator = build_generator()

# Combine models for GAN
discriminator.trainable = False
z = tf.keras.Input(shape=(noise_dim,))
profile_input = tf.keras.Input(shape=profile_train_encoded.shape[1:]) # Use the encoded profile shape
img = generator(z)
validity = discriminator([img, profile_input]) # Pass profile input to discriminator
combined = tf.keras.Model([z, profile_input], validity) # Include profile input in combined model
combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Training the GAN
# Training the GAN
def train_gan(generator, discriminator, combined, x_train, profile_train, epochs=2000, batch_size=32, save_interval=1000):
    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_imgs = x_train[idx]
        real_profiles = profile_train[idx]

        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch([real_imgs, real_profiles], valid)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, real_profiles], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        # Pass both noise and corresponding profile data to the combined model
        g_loss = combined.train_on_batch([noise, real_profiles], valid) # Pass profile data here

        # Print progress
        print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {d_loss[1] * 100}] [G loss: {g_loss}]")

        # Save generated image samples at intervals
        if epoch % save_interval == 0:
            # Define the save_imgs function here
            def save_imgs(generator, epoch):
                r, c = 5, 5
                noise = np.random.normal(0, 1, (r * c, noise_dim))
                gen_imgs = generator.predict(noise)

                # Rescale images 0 - 1
                gen_imgs = 0.5 * gen_imgs + 0.5

                fig, axs = plt.subplots(r, c)
                cnt = 0
                for i in range(r):
                    for j in range(c):
                        axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                        axs[i,j].axis('off')
                        cnt += 1
                fig.savefig("/content/drive/MyDrive/Fake Profile Dectection/fake/fake_%d.png" % epoch)
                plt.close()

            save_imgs(generator, epoch) # Call the newly defined function

        # Save model weights at intervals
        if epoch % save_interval == 0:
            discriminator.save_weights('/content/drive/MyDrive/Fake Profile Dectection/discriminator_weights.h5')
            generator.save_weights('/content/drive/MyDrive/Fake Profile Dectection/generator_weights.h5')

train_gan(generator, discriminator, combined, x_train, profile_train_encoded, epochs=200, batch_size=32, save_interval=1000)

# Save the models after training
generator.save('/content/drive/MyDrive/Fake Profile Dectection/generator_model.h5')
discriminator.save('/content/drive/MyDrive/Fake Profile Dectection/discriminator_model.h5')

from keras.models import load_model

# Load the saved models
generator = load_model('/content/drive/MyDrive/Fake Profile Dectection/generator_model.h5')
discriminator = load_model('/content/drive/MyDrive/Fake Profile Dectection/discriminator_model.h5')

discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

def generate_fake_images(generator, profile_data, noise_dim=100):
    noise = np.random.normal(0, 1, (profile_data.shape[0], noise_dim))
    gen_imgs = generator.predict(noise)
    return gen_imgs

# Assuming profile_test_encoded is already defined and contains the test profile data
fake_images = generate_fake_images(generator, profile_test_encoded)

def predict_profiles(discriminator, images, profile_data):
    predictions = discriminator.predict([images, profile_data])
    return predictions

# Predict real and fake profiles
real_predictions = predict_profiles(discriminator, x_test, profile_test_encoded)
fake_predictions = predict_profiles(discriminator, fake_images, profile_test_encoded)

from sklearn.metrics import accuracy_score, confusion_matrix

# Binarize predictions
real_predictions_bin = (real_predictions > 0.5).astype(int)
fake_predictions_bin = (fake_predictions > 0.5).astype(int)

# True labels
true_labels_real = np.zeros(x_test.shape[0])
true_labels_fake = np.ones(fake_images.shape[0])

# Combine predictions and labels
combined_predictions = np.concatenate((real_predictions_bin, fake_predictions_bin))
combined_labels = np.concatenate((true_labels_real, true_labels_fake))

# Evaluate
accuracy = accuracy_score(combined_labels, combined_predictions)
conf_matrix = confusion_matrix(combined_labels, combined_predictions)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Save models
discriminator.save('/content/drive/MyDrive/Fake Profile Dectection/discriminator_model.h5')
generator.save('/content/drive/MyDrive/Fake Profile Dectection/generator_model.h5')

# Load models
from keras.models import load_model
discriminator = load_model('/content/drive/MyDrive/Fake Profile Dectection/discriminator_model.h5')
generator = load_model('/content/drive/MyDrive/Fake Profile Dectection/generator_model.h5')

import cv2
import numpy as np

def load_and_preprocess_image(image_path, img_shape):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.resize(img, (img_shape[1], img_shape[0]))
    img = (img / 127.5) - 1.0  # Normalize image to [-1, 1]
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Example usage
sample_image_path = '/content/drive/MyDrive/Fake Profile Dectection/real/real_00080.jpg'  # Replace with actual path
img_shape = (64, 64, 3)  # Shape should match your model's input shape
sample_image = load_and_preprocess_image(sample_image_path, img_shape)

from keras.models import load_model

# Load the saved models
discriminator = load_model('/content/drive/MyDrive/Fake Profile Dectection/discriminator_model.h5')
generator = load_model('/content/drive/MyDrive/Fake Profile Dectection/generator_model.h5')

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_and_preprocess_image(image_path, img_shape):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.resize(img, (img_shape[1], img_shape[0]))
    img = (img / 127.5) - 1.0  # Normalize image to [-1, 1]
    return np.expand_dims(img, axis=0)  # Add batch dimension

def load_profile_info(csv_file, image_name):
    profiles = pd.read_csv(csv_file)
    # Check if image name (without extension) exists in 'image_path' column
    profile_data = profiles[profiles['image_path'].str.contains(image_name.split('.')[0])]
    return profile_data.drop(columns=['image_path']).values

# Example usage - Adjust the paths and image name accordingly
sample_image_path = '/content/drive/MyDrive/Fake Profile Dectection/fake/easy_224_1100.jpg'
img_shape = (64, 64, 3)
sample_image = load_and_preprocess_image(sample_image_path, img_shape)

# Extract the image name from the path (without extension)
sample_image_name = sample_image_path.split('/')[-1].split('.')[0]
sample_profile_info = load_profile_info('/content/drive/MyDrive/Fake Profile Dectection/real_profiles.csv', sample_image_name)

if sample_profile_info.size == 0:
    print(f"No profile information found for {sample_image_name}. Check if the image name is present in the CSV file.")
else:
    # Convert profile data to numerical representation
    encoder = OneHotEncoder()  # Initialize OneHotEncoder here
    sample_profile_encoded = encoder.fit_transform(sample_profile_info).toarray()

    # Check if the shape of the encoded profile matches what the discriminator expects
    expected_input_shape = discriminator.input_shape[1][1]  # Get the expected shape from the discriminator model
    if sample_profile_encoded.shape[1] != expected_input_shape:
        print(f"Warning: Profile data shape ({sample_profile_encoded.shape[1]}) does not match expected input shape ({expected_input_shape})")
        # Handle the mismatch here, e.g., by resizing the encoded profile or adjusting your model

    print(sample_profile_encoded)

def classify_image(discriminator, image, profile_encoded):
    if profile_encoded is None:
        print("Cannot classify image without profile information.")
        return None

    # Reshape profile_encoded to match the expected input shape of the discriminator
    profile_encoded = np.reshape(profile_encoded, (1, -1))  # Flatten the array
    # Pad the profile_encoded array with zeros to match the expected size
    expected_size = discriminator.input_shape[1][1]
    profile_encoded = np.pad(profile_encoded, [(0, 0), (0, expected_size - profile_encoded.shape[1])])

    prediction = discriminator.predict([image, profile_encoded])
    return prediction

# Predict the class of the sample image
prediction = classify_image(discriminator, sample_image, sample_profile_encoded)
if prediction is not None:
    print("Prediction (0: Real, 1: Fake):", prediction)

