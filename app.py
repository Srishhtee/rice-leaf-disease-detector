import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile, os, glob, time

# Page Setup
st.set_page_config(page_title="Rice Leaf Disease Detector", page_icon="", layout="centered")

# Custom Transparent Style
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://t3.ftcdn.net/jpg/05/06/01/58/360_F_506015806_u0a26CkS65J3iGJRljc4bCfJeNOks4Nj.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #222;  /* 🟢 darker text overall */
}

/* Slight overlay for soft contrast */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background-color: rgba(255, 255, 255, 0.3);
    z-index: -1;
}

/* 🏷 Title */
h1 {
    text-align: center;
    background: linear-gradient(90deg, #8B4513, #654321); /* darker gold-brown */
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}

/* 📜 Subtitle */
.subtitle {
    text-align: center;
    color: #2f2f2f; /* dark gray for readability */
    font-size: 1.15rem;
    margin-bottom: 1.8rem;
    font-weight: 600;
}

/* 📁 File uploader label */
div[data-testid="stFileUploader"] > label {
    color: #222; /* darker text */
    font-weight: bold;
    font-size: 1.1rem;
}

/* 🔘 Buttons */
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

.stSuccess, .stWarning {
    font-size: 1.1rem !important;
    font-weight: 500 !important;
}

/* Hide Streamlit footer */
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# App Title and Description
st.markdown("<h1>Rice Leaf Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload a rice leaf image to identify the disease using AI</p>", unsafe_allow_html=True)

# Load YOLO model
model_path = "rice_leaf_yolo_model.pt"
model = YOLO(model_path)

# Upload image
uploaded_file = st.file_uploader("Upload a rice leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    temp_dir = tempfile.mkdtemp()
    img_path = os.path.join(temp_dir, uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(Image.open(img_path), caption="📸 Uploaded Image", use_container_width=True)
    st.write("Detecting disease... Please wait...")

    # Predict
    results = model.predict(source=img_path, conf=0.5, save=True)

    # Find latest prediction folder dynamically
    result_dirs = glob.glob("runs/detect/predict*")
    if result_dirs:
        latest_dir = max(result_dirs, key=os.path.getctime)
    else:
        latest_dir = "runs/detect/predict"

    # Wait until YOLO finishes saving output
    time.sleep(1)
    res_img_path = os.path.join(latest_dir, os.path.basename(uploaded_file.name))

    # Check if file actually exists
    if os.path.exists(res_img_path):
        st.image(res_img_path, caption="🩺 Detection Result", use_container_width=True)
    else:
        st.error("Result image not found. Try uploading again!")

    # Get detected class labels
    labels = [model.names[int(cls)] for cls in results[0].boxes.cls]
    if labels:
        st.success(f"Detected Disease: **{', '.join(set(labels))}**")
    else:
        st.warning("No visible disease detected.")
