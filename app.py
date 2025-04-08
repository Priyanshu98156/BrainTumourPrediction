import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import time
from io import BytesIO
import base64
from streamlit_option_menu import option_menu

# === Background Video Styling ===
st.markdown(
    """
    <style>
    .video-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        z-index: -1;
        opacity: 0.8;
    }

    .stApp {
        background: transparent;
    }

    .result-box {
        background-color: rgba(100, 255, 250, 0.2);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffffff;
        color: #ffffff;
        font-weight: bold;
        font-size: 18px;
        animation: fadeIn 1.5s ease-in-out;
    }

    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    </style>

    <video id="bgVideo" autoplay muted loop class="video-bg">
        <source src="https://github.com/Priyanshu98156/videos/raw/main/nodes.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>

    <script>
    var vid = document.getElementById("bgVideo");
    vid.playbackRate = 0.4;
    </script>
    """,
    unsafe_allow_html=True
)
st.markdown("""
    <style>
    div[data-testid="stSidebar"] a:hover {
        color: #1cfff2 !important;
    }
    </style>
""", unsafe_allow_html=True)
# === Style File Uploader ===
st.markdown("""
    <style>
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.0) !important;
        border: none !important;
        padding: 15px;
        color: white !important;
        text-align: center;
    }

    [data-testid="stFileUploader"] label {
        color: white !important;
        font-weight: bold;
        font-size: 16px;
        text-align: center;
    }

    /* Hide uploaded file name and size */
    [data-testid="uploaded-file-name"] {
        display: none !important;
    }
    </style>
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

# === Preprocess Function ===
def preprocessing_image(uploaded_file):
    image = np.array(uploaded_file)
    if len(image.shape) == 2:
        gray_image = image
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.resize(gray_image, (128, 128))
    gray_image = gray_image / 255.0
    return np.expand_dims(gray_image, axis=(0, -1))

# === Sidebar Navigation ===
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Guide", "How This Works", "About Us"],
        icons=["house", "book", "gear", "person-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#1e1e1e"},
            "icon": {"color": "#black", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "color": "#ffffff",
                "--hover-color": "#13cfc2"
            },
            "nav-link-selected": {
                "background-color": "#16ccc1",
                "color": "black",
                "font-weight": "bold",
                "icon-color": "white"
            }
        }
    )

# === Pages ===

# --- HOME PAGE ---
if selected == "Home":
    st.markdown("""
        <h1 style='text-align: center; color: #1cfff2; font-size: 60px; font-weight:bold;'>MedVision</h1>
        <p style='text-align: center; color: #ffffff; font-size: 18px;'>AI-powered Brain Tumor Detection Tool</p>
    """, unsafe_allow_html=True)

    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Upload an MRI image", type=['jpg', 'png', 'jpeg'])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file and model:
        image = Image.open(uploaded_file)

        if image.mode != "RGB":
            image = image.convert("RGB")
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        encoded = base64.b64encode(img_bytes).decode()

        st.markdown(
            f"""
            <div style="text-align: center; margin-top: -40px;">
                <img src="data:image/png;base64,{encoded}" 
                    style="width: 200px; 
                            border-radius: 20px; 
                            border: 0px solid #ffffff;
                            transition: transform 0.3s ease;"
                    onmouseover="this.style.transform='scale(1.05)'"
                    onmouseout="this.style.transform='scale(1)'"
                />
                <p style="color: white; margin-top: 10px;">📷 Uploaded Image</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        processed_image = preprocessing_image(image)

        with st.spinner("🔍 Analyzing MRI..."):
            time.sleep(3)
            prediction = model.predict(processed_image)[0][0]

        # === Show Result ===
        st.write("### Prediction Result:")

        if prediction > 0.5:
            st.markdown("""
            <div class="result-box">
                🧠 Tumor Detected
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-box">
                ✅ No Tumor Detected
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
            <hr style="border: 1px solid #1cfff2; margin-top: 30px; margin-bottom: 10px;">
            <p style="
                text-align: justify;
                color: white;
                font-size: 16px;
                background-color: rgba(255, 255, 255, 0.05);
                padding: 15px;
                border-radius: 10px;
                border-left: 4px solid #1cfff2;
                box-shadow: 0 0 10px rgba(28, 255, 242, 0.2);
            ">
            ⚠️ <b>Important:</b> This result is generated by a Trained Data Model and is intended for informational purposes only.
            Only a qualified radiologist or medical professional can accurately interpret MRI scans and diagnose conditions such as brain tumors.<br>
            If you have any concerns about this scan, we strongly recommend consulting a licensed healthcare provider for a comprehensive evaluation.
            </p>
            """, unsafe_allow_html=True)

# --- GUIDE PAGE ---
elif selected == "Guide":
    st.title("📘 User Guide")
    st.markdown("""
    #### Here's how to use MedVision:
    - **Step 1:** Go to the Home page  
    - **Step 2:** Upload a clear MRI scan (.jpg, .jpeg, .png)  
    - **Step 3:** Wait while the model processes it  
    - **Step 4:** See results and check for tumor detection  

    This tool is designed for early detection and quick screening.
    """)
    st.image("https://github.com/Priyanshu98156/images/raw/main/lotus.png", width=200)

# --- HOW THIS WORKS PAGE ---
elif selected == "How This Works":
    st.title("⚙️ How This Works")
    st.markdown("""
    This app uses a **Convolutional Neural Network (CNN)** trained on thousands of MRI brain scans.

    **Workflow:**
    - Convert image to grayscale
    - Resize to 128x128
    - Normalize pixel values
    - Predict using a trained TensorFlow model

    If prediction > 0.5 → **Tumor Detected**  
    Else → **No Tumor**

    Powered by **TensorFlow, OpenCV, and Streamlit**.
    """)
    st.image("https://miro.medium.com/v2/resize:fit:1200/format:webp/1*MaP1B-8W7H8yETrg94Q_Wg.png", use_column_width=True)

# --- ABOUT US PAGE ---
elif selected == "About Us":
    st.title("👤 About Us")
    st.markdown("""
    **MedVision** is a student-driven AI project aimed at helping radiologists and medical professionals.

    - 👨‍💻 Developer: Priyanshu  
    - 🧪 Focus: AI in Healthcare  
    - 💻 Tools Used: Python, TensorFlow, OpenCV, Streamlit  
    - 🌟 Goal: Assist in fast, reliable tumor screening  
    - 📬 Contact: your.email@example.com  
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=150)
