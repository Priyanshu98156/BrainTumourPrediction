import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
#load model
def load_model():
    try:
        model = tf.keras.models.load_model("brain_tumor_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    
def preprocessing_image(uploaded_file):
    # Convert PIL Image to NumPy array
    image = np.array(uploaded_file)  

   #ensure 3 channels (RGB) before converting to grayscale
    if len(image.shape) == 2:
        gray_image = image
    else:
        gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    gray_image = cv2.resize(gray_image,(128,128))

    gray_image = gray_image/255.0

    #Expand dimensions for model input (1,128,128,-1)
    gray_image = np.expand_dims(gray_image,axis=(0,-1))

    return gray_image




model = load_model()


uploaded_file = st.file_uploader("Upload an MRI image",  type = ['jpg','png', 'jpeg'])


if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image,caption="Uploaded Image",width = 300)
    processed_image = preprocessing_image(image)
    prediction = model.predict(processed_image)[0][0]

    if model:  # Ensure model is loaded
        prediction = model.predict(processed_image)[0][0]

        if prediction > 0.5:
            st.error("*** Tumor Detected ***")
        else:
            st.success("**No Tumor Detected**")

        st.write(f"Confidence: {prediction:.2f}")