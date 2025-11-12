import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile, os, glob, time, requests

# ------------------ PAGE SETUP ------------------
st.set_page_config(page_title="Rice Leaf Disease Detector", page_icon="🍃", layout="centered")

# ------------------ STYLING ------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://t3.ftcdn.net/jpg/05/06/01/58/360_F_506015806_u0a26CkS65J3iGJRljc4bCfJeNOks4Nj.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #222;
}
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background-color: rgba(255, 255, 255, 0.3);
    z-index: -1;
}
h1 {
    text-align: center;
    background: linear-gradient(90deg, #8B4513, #654321);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.subtitle {
    text-align: center;
    color: #2f2f2f;
    font-size: 1.15rem;
    margin-bottom: 1.8rem;
    font-weight: 600;
}
div[data-testid="stFileUploader"] > label {
    color: #222;
    font-weight: bold;
    font-size: 1.1rem;
}
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
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.markdown("<h1>Rice Leaf Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload or capture a rice leaf image to identify disease using AI</p>", unsafe_allow_html=True)

# ------------------ DOWNLOAD MODEL FROM DRIVE ------------------
def download_from_drive(file_id, dest_path):
    """Download model file from Google Drive if not present"""
    if os.path.exists(dest_path):
        return

    st.info("📥 Downloading model from Google Drive... please wait (takes time on first run).")

    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    response.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    st.success("✅ Model downloaded successfully!")

# Your Google Drive file ID
DRIVE_FILE_ID = "1smsUoQLoijhyfqMzjVzJvIVy6wTcYu-7"
MODEL_PATH = "rice_leaf_yolo_model.pt"

download_from_drive(DRIVE_FILE_ID, MODEL_PATH)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(MODEL_PATH)

# ------------------ IMAGE UPLOAD ------------------
uploaded_file = st.file_uploader("📂 Upload a rice leaf image...", type=["jpg", "jpeg", "png"])

# ------------------ CAMERA INPUT ------------------
st.markdown("### Or take a live photo (for Android users):")
camera_file = st.camera_input("📸 Capture a photo")

# Prioritize camera photo if both are given
image_source = camera_file if camera_file else uploaded_file

if image_source:
    temp_dir = tempfile.mkdtemp()
    img_path = os.path.join(temp_dir, image_source.name)
    with open(img_path, "wb") as f:
        f.write(image_source.read())

    st.image(Image.open(img_path), caption="Uploaded Image", use_container_width=True)
    st.write("🧠 Detecting disease... please wait...")

    # Run prediction
    results = model.predict(source=img_path, conf=0.5, save=True, verbose=False)

    # Find latest prediction folder dynamically
    result_dirs = glob.glob("runs/detect/predict*")
    latest_dir = max(result_dirs, key=os.path.getctime) if result_dirs else "runs/detect/predict"

    time.sleep(1)  # small delay for saving output
    res_img_path = os.path.join(latest_dir, os.path.basename(image_source.name))

    if os.path.exists(res_img_path):
        st.image(res_img_path, caption="Detection Result", use_container_width=True)
    else:
        st.error("⚠️ Result image not found. Try again!")

    # Detected class labels
    if results and len(results[0].boxes) > 0:
        labels = [model.names[int(cls)] for cls in results[0].boxes.cls]
        st.success(f"🌾 Detected Disease: **{', '.join(set(labels))}**")
    else:
        st.warning("No visible disease detected.")
