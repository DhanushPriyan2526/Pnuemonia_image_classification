# Pnuemonia_image_classification
This project involves building and deploying a convolutional neural network (CNN) model for classifying chest X-ray images into two categories: Normal and Pneumonia. The model is deployed as a web application using Streamlit.

The goal of this project is to classify chest X-ray images as either Normal or Pneumonia using a convolutional neural network (CNN). The model is trained on the Chest X-ray Images (Pneumonia) dataset from Kaggle and is deployed as an interactive web application using Streamlit.

Dataset
The dataset used for training and testing the model can be found on Kaggle: Chest X-ray Images (Pneumonia).

Model
The CNN model is built using TensorFlow and Keras. It consists of several convolutional layers followed by max-pooling layers and dense layers. The model is trained to achieve high accuracy in distinguishing between normal and pneumonia-infected chest X-rays.

Prerequisites
Before you begin, ensure you have met the following requirements:
Python 3.6+
pip package manager

Installation
Clone the repository:
git clone https://github.com/yourusername/pneumonia-classification.git
cd pneumonia-classification

Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required dependencies:
pip install -r requirements.txt
