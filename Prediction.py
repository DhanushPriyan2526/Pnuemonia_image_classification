import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocess import preprocess_image
import matplotlib.pyplot as plt

# Load the pre-trained model
model_path = 'model/cnn_model.h5'
model = load_model(model_path)

def predict_image(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    return prediction

def display_prediction(image_path, prediction):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(f'Prediction: {"Pneumonia" if prediction > 0.5 else "Normal"}')
    plt.show()

# Directory paths
test_dir = 'chest_xray/test'

# Image data generator for test data
test_datagen = ImageDataGenerator(rescale=0.255)

# Load test data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Optionally, display predictions for individual images
# image_dirs = ['NORMAL', 'PNEUMONIA']
# for image_dir in image_dirs:
#     full_dir = os.path.join(test_dir, image_dir)
#     for image_name in os.listdir(full_dir):
#         image_path = os.path.join(full_dir, image_name)
#         prediction = predict_image(image_path)
#         display_prediction(image_path, prediction)
