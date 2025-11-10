import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile, os

# 🌱 Page Setup
st.set_page_config(page_title="🌾 Rice Leaf Disease Detector", page_icon="🍃", layout="centered")

# 🌟 Custom Transparent Style
st.markdown("""
<style>
/* Full Background Image */
[data-testid="stAppViewContainer"] {
    background-image: url("https://www.shutterstock.com/image-photo/rice-blast-diseases-on-leaf-260nw-2422855389.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #fff;
}

/* Transparent Overlay */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background-color: rgba(0, 0, 0, 0.45);
    z-index: -1;
}

/* Title */
h1 {
    text-align: center;
    background: linear-gradient(90deg, #FFD700, #FFB347);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #f2f2f2;
    font-size: 1.15rem;
    margin-bottom: 1.8rem;
}

/* File Upload */
div[data-testid="stFileUploader"] > label {
    color: #f5deb3;
    font-weight: bold;
    font-size: 1.1rem;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(90deg, #FFD700, #FFA500);
    color: #222;
    border-radius: 10px;
    border: none;
    font-size: 1rem;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    transition: all 0.3s ease-in-out;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #FFA500, #FF8C00);
    transform: scale(1.05);
}

/* Results text */
.stSuccess, .stWarning {
    font-size: 1.1rem !important;
    font-weight: 500 !important;
}

/* Hide footer */
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 🌾 App Title and Description
st.markdown("<h1>🌾 Rice Leaf Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload a rice leaf image to identify the disease using AI 🍂</p>", unsafe_allow_html=True)

# Load YOLO Model
model_path = "rice_leaf_yolo_model.pt"
model = YOLO(model_path)

# Image Upload
uploaded_file = st.file_uploader("📤 Upload a rice leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    temp_dir = tempfile.mkdtemp()
    img_path = os.path.join(temp_dir, uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(Image.open(img_path), caption="📸 Uploaded Image", use_container_width=True)
    st.write("🔍 Detecting disease... Please wait...")

    # Predict
    results = model.predict(source=img_path, conf=0.5)
    res_img_path = os.path.join(results[0].save_dir, os.path.basename(img_path))
    st.image(res_img_path, caption="🩺 Detection Result", use_container_width=True)

    labels = [model.names[int(cls)] for cls in results[0].boxes.cls]
    if labels:
        st.success(f"✅ Detected Disease: **{', '.join(set(labels))}**")
    else:
        st.warning("No visible disease detected.")


