import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the pre-trained model
model_path = 'model/cnn_model.h5'
try:
    model = load_model(model_path)
    logging.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    st.error(f"Error loading model: {e}")

# Define image preprocessing function
IMG_SIZE = (128, 128)

def preprocess_image(image):
    try:
        image = image.resize(IMG_SIZE)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0  # Normalize to [0, 1]
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        st.error(f"Error preprocessing image: {e}")

def predict_image(image):
    try:
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        return prediction
    except Exception as e:
        logging.error(f"Error predicting image: {e}")
        st.error(f"Error predicting image: {e}")

# Streamlit app
st.title("Pneumonia Image Classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    prediction = predict_image(image)
    if prediction is not None:
        if prediction > 0.5:
            st.write("The image is classified as: **Pneumonia**")
        else:
            st.write("The image is classified as: **Normal**")
