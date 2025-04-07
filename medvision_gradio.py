import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import time

# === Load Model ===
model = tf.keras.models.load_model("brain_tumor_model.h5")

# === Prediction Function ===
def predict(image):
    image = image.convert("RGB")
    img = np.array(image)
    
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_img = cv2.resize(gray_img, (128, 128))
    gray_img = gray_img / 255.0
    gray_img = np.expand_dims(gray_img, axis=(0, -1))

    # Simulate loading time
    time.sleep(3)

    prediction = model.predict(gray_img)[0][0]
    label = "🧠 Tumor Detected" if prediction > 0.5 else "✅ No Tumor Detected"
    confidence = f"{prediction:.2f}"

    return label, confidence

# === Gradio UI ===
title = "🧬 MedVision"
description = "AI-powered Brain Tumor Detection Tool. Upload an MRI image to get started."

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload MRI Image"),
    outputs=[
        gr.Label(label="Prediction"),
        gr.Textbox(label="Confidence Score")
    ],
    title=title,
    description=description,
    theme="default",  # or "soft", "huggingface", "dark"
    allow_flagging="never"
)

# === Launch App ===
interface.launch()
