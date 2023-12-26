# Pneumonia-Prediction

## Overview
This project implements a Pneumonia Detection System using a Convolutional Neural Network (CNN). The system analyzes chest X-ray images to classify whether a person has pneumonia or not. The project includes a Streamlit web application for easy interaction.

## Access Deployed Code
The Pneumonia Detection System is deployed and can be accessed using the following link: [Pneumonia Prediction Streamlit App](https://pneumonia-prediction-b7zuf4tf0em.streamlit.app/)

## Installation
To run the Pneumonia Detection System locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/Pneumonia-Detection-System.git
   ```
2. **Install the required Python packages:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit app:**
   ```bash
   streamlit run bio.py
   ```
This will launch the web application locally.

##Usage
1. Navigate to the provided URL to run the Streamlit app.
2. Choose a chest X-ray image file (in PNG, JPG, or JPEG format) for analysis.
3. Click the "Classify" button to let the model predict whether pneumonia is detected or not.

##Model Details
The CNN model used in this project is trained to classify chest X-ray images. The model details, including layers and architecture, can be found in the model summary section of the Streamlit app.
