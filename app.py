import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

#load model
def load_model():
    try:
        model = tf.keras.models.load_model("brain_tumor_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    
    
def preprocessing_image(image):
    image = image.resize((128,128))
    image = np.array(image)/255.0
    image = np.expand_dims(image,axis = 0 )    
    
    return image
model = load_model()


uploaded_file = st.file_uploader("Upload an MRI image",  type = ['jpg','png', 'jpeg'])


if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image,caption="Uploaded Image",use_column_width= True)
    processed_image = preprocessing_image(image)
    prediction = model.predict(processed_image)[0][0]


    if prediction >  0.5:
        st.error("*** Tumor Detected***")
    else:
        st.success("**No Tumor Detected")

    st.write(f"Confidence :{prediction:.2f}")