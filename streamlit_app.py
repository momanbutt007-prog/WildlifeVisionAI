
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "wildlife_classifier_final.keras"
CLASSES_PATH = BASE_DIR / "models" / "class_names.json"

IMAGE_SIZE = 384
DEFAULT_CLASSES = ["buffalo", "elephant", "rhino", "zebra"]


st.set_page_config(
    page_title="Wildlife Vision AI",
    page_icon="🦓",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(255, 107, 0, .14), transparent 28%),
        radial-gradient(circle at 95% 20%, rgba(255, 150, 50, .09), transparent 25%),
        #070707;
    color: #f5f5f5;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111111, #080808);
    border-right: 1px solid rgba(255,255,255,.08);
}

.hero {
    padding: 52px 55px;
    border-radius: 30px;
    margin-bottom: 28px;
    background:
        linear-gradient(135deg, rgba(255,105,0,.13), transparent 45%),
        linear-gradient(145deg, #181818, #0c0c0c);
    border: 1px solid rgba(255,105,0,.20);
    box-shadow: 0 25px 70px rgba(0,0,0,.45);
}

.hero .eyebrow {
    color: #ff7518;
    letter-spacing: 3px;
    font-weight: 800;
    font-size: 12px;
}

.hero h1 {
    font-size: clamp(40px, 6vw, 76px);
    line-height: .98;
    letter-spacing: -4px;
    margin: 12px 0;
}

.hero h1 span {
    color: #ff6b00;
}

.hero p {
    color: #999;
    max-width: 720px;
    font-size: 17px;
    line-height: 1.7;
}

.metric {
    background: #111;
    border: 1px solid rgba(255,255,255,.08);
    padding: 18px;
    border-radius: 18px;
}

.metric .number {
    color: #ff7418;
    font-size: 28px;
    font-weight: 900;
}

.metric .label {
    color: #777;
    font-size: 12px;
    margin-top: 4px;
}

.prediction-card {
    padding: 30px;
    border-radius: 24px;
    background: linear-gradient(145deg,#181818,#0d0d0d);
    border: 1px solid rgba(255,105,0,.22);
    text-align: center;
}

.prediction-label {
    color: #777;
    font-size: 11px;
    letter-spacing: 3px;
    font-weight: 800;
}

.prediction-name {
    color: #ff7418;
    font-size: 42px;
    font-weight: 900;
    margin: 8px 0;
}

.confidence {
    color: #ddd;
    font-size: 17px;
}

div[data-testid="stFileUploader"] {
    border-radius: 22px;
}

.stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(255,105,0,.35);
    background: linear-gradient(135deg,#ff6500,#ff963f);
    color: #080808;
    font-weight: 900;
}

.animal-card {
    padding: 22px;
    background: #111;
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 20px;
    text-align: center;
}

.animal-card .emoji {
    font-size: 45px;
}

.animal-card b {
    display: block;
    margin-top: 10px;
}

.footer {
    text-align: center;
    color: #555;
    padding: 35px 0 10px;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model missing. Put wildlife_classifier_final.keras inside models/."
        )

    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data
def load_classes():
    if CLASSES_PATH.exists():
        try:
            data = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list) and len(data) == 4:
                return data
        except Exception:
            pass

    return DEFAULT_CLASSES


def predict(image):
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    probabilities = load_model().predict(array, verbose=0)[0]

    if not np.isclose(np.sum(probabilities), 1.0, atol=0.02):
        exp = np.exp(probabilities - np.max(probabilities))
        probabilities = exp / np.sum(exp)

    classes = load_classes()
    order = np.argsort(probabilities)[::-1]

    return [
        (classes[int(i)], float(probabilities[i] * 100))
        for i in order
    ]


classes = load_classes()

with st.sidebar:
    st.markdown("## 🦓 Wildlife Vision")
    st.caption("Deep Learning Classification System")

    st.divider()

    st.markdown("### Model")
    st.success("● Model Ready")

    st.markdown("""
    **Architecture:** EfficientNetV2-S  
    **Input:** 384 × 384  
    **Classes:** 4  
    **Validation Accuracy:** 98.01%
    """)

    st.divider()

    st.markdown("### Supported Wildlife")
    for animal in classes:
        st.write(f"• {animal.title()}")

    st.divider()
    st.caption("Built with TensorFlow + Streamlit")


st.markdown("""
<div class="hero">
<div class="eyebrow">AI-POWERED WILDLIFE RECOGNITION</div>
<h1>See the wild.<br><span>Understand it.</span></h1>
<p>
Upload a wildlife photograph and let a trained deep-learning model
identify the animal in seconds.
</p>
</div>
""", unsafe_allow_html=True)


m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown('<div class="metric"><div class="number">98.01%</div><div class="label">VALIDATION ACCURACY</div></div>', unsafe_allow_html=True)

with m2:
    st.markdown('<div class="metric"><div class="number">98.02%</div><div class="label">F1 SCORE</div></div>', unsafe_allow_html=True)

with m3:
    st.markdown('<div class="metric"><div class="number">98.11%</div><div class="label">PRECISION</div></div>', unsafe_allow_html=True)

with m4:
    st.markdown('<div class="metric"><div class="number">0.189</div><div class="label">VALIDATION LOSS</div></div>', unsafe_allow_html=True)


st.write("")


left, right = st.columns([1.05, .95], gap="large")

with left:
    st.markdown("### 📤 Upload Wildlife Image")
    uploaded = st.file_uploader(
        "Drag and drop an image here",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="Recommended: clear wildlife photographs."
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded wildlife image", use_container_width=True)

with right:
    st.markdown("### 🔍 AI Prediction")

    if uploaded:
        with st.spinner("Analyzing image..."):
            predictions = predict(image)

        top_name, top_score = predictions[0]

        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-label">IDENTIFIED WILDLIFE</div>
            <div class="prediction-name">{top_name.title()}</div>
            <div class="confidence">Confidence: <b>{top_score:.2f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.markdown("#### Prediction Breakdown")

        for name, score in predictions:
            st.write(f"**{name.title()}** — {score:.2f}%")
            st.progress(min(score / 100, 1.0))
    else:
        st.info("Upload an image to receive an AI prediction.")


st.divider()

st.markdown("### 🐾 Recognized Wildlife")

cols = st.columns(4)
emoji_map = {
    "buffalo": "🦬",
    "elephant": "🐘",
    "rhino": "🦏",
    "zebra": "🦓"
}

for col, animal in zip(cols, classes):
    emoji = emoji_map.get(animal.lower(), "🐾")
    with col:
        st.markdown(
            f'<div class="animal-card"><div class="emoji">{emoji}</div><b>{animal.title()}</b></div>',
            unsafe_allow_html=True
        )


st.markdown(
    '<div class="footer">Wildlife Vision AI • EfficientNetV2-S • TensorFlow • Streamlit</div>',
    unsafe_allow_html=True
)
