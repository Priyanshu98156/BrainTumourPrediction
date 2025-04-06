import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import time

# === Dark Mode Toggle ===
dark_mode = st.checkbox("🌙 Dark Mode")
if dark_mode:
    st.markdown("""
        <style>
        body {
            background-color: #121212;
            color: #FFFFFF;
        }
        </style>
    """, unsafe_allow_html=True)

# === MedVision Header ===
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50; font-size: 60px;'>
        MedVision
    </h1>
            
    <p style='text-align: center; color: gray; font-size: 18px;'>
            AI-powered Brain Tumor Detection Tool
    </p>
        """, unsafe_allow_html=True)

# === Load Model ===
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("brain_tumor_model.h5")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# === Preprocess Image ===
def preprocessing_image(uploaded_file):
    image = np.array(uploaded_file)
    if len(image.shape) == 2:
        gray_image = image
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.resize(gray_image, (128, 128))
    gray_image = gray_image / 255.0
    return np.expand_dims(gray_image, axis=(0, -1))

# === Upload & Predict ===
uploaded_file = st.file_uploader("Upload an MRI image", type=['jpg', 'png', 'jpeg'])

if uploaded_file and model:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)
    processed_image = preprocessing_image(image)

    with st.spinner("🔍 Analyzing MRI..."):
        time.sleep(3)
        prediction = model.predict(processed_image)[0][0]

    # === Show Result ===
    st.write("### Prediction Result:")
    if prediction > 0.5:
        st.error("🧠 Tumor Detected")
    else:
        st.success("✅ No Tumor Detected")

    # === Confidence Progress Bar ===
    st.write(f"Prediction Confidence: {prediction:.2f}")
    st.progress(int(prediction * 100))
    # st.write(f"Confidence: {prediction:.2f}")