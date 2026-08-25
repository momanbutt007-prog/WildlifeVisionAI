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


# ============================================================
# THEME — Savanna palette: charcoal-umber bg, gold + terracotta + olive accents
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800;900&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4 { font-family: 'Poppins', sans-serif !important; }

/* ---------- App background: charcoal-umber savanna dusk ---------- */
.stApp {
    background:
        radial-gradient(circle at 8% -5%, rgba(212, 160, 23, 0.12), transparent 32%),
        radial-gradient(circle at 95% 15%, rgba(191, 87, 44, 0.10), transparent 30%),
        radial-gradient(circle at 50% 105%, rgba(107, 122, 66, 0.10), transparent 55%),
        linear-gradient(160deg, #100d09 0%, #17130d 45%, #0d0b07 100%);
    color: #f4ecdc;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 25% 0%, rgba(212,160,23,0.09), transparent 45%),
        radial-gradient(circle at 90% 45%, rgba(191,87,44,0.08), transparent 50%),
        linear-gradient(180deg, #14110b 0%, #0a0805 100%);
    border-right: 1px solid rgba(244,236,220,0.08);
}

section[data-testid="stSidebar"] * { color: #f4ecdc !important; }

.sb-logo-wrap { text-align: center; padding: 0.6rem 0 0.4rem 0; }

.sb-logo-badge {
    width: 60px;
    height: 60px;
    margin: 0 auto 0.55rem auto;
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.85rem;
    background: linear-gradient(135deg, #bf572c, #d4a017 55%, #6b7a42);
    box-shadow: 0 10px 26px rgba(212,160,23,0.30);
}

.sb-title {
    font-size: 1.22rem;
    font-weight: 800;
    font-family: 'Poppins', sans-serif;
    margin-bottom: 0.12rem;
    color: #f4ecdc;
}

.sb-subtitle {
    font-size: 0.7rem;
    color: rgba(244,236,220,0.55) !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.sb-section-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 0.92rem;
    margin: 0.6rem 0 0.6rem 0;
    color: #f4ecdc !important;
}

.sb-section-label .icon-chip-sm {
    width: 25px;
    height: 25px;
    min-width: 25px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.82rem;
    background: linear-gradient(135deg, rgba(212,160,23,0.30), rgba(191,87,44,0.28));
    border: 1px solid rgba(244,236,220,0.14);
}

.sb-card {
    background: rgba(244,236,220,0.045);
    border: 1px solid rgba(244,236,220,0.12);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.28);
}

.sb-card p {
    margin: 0.25rem 0;
    font-size: 0.86rem;
    color: rgba(244,236,220,0.85) !important;
}

.sb-card b { color: #e8bd52 !important; }

.sb-status-chip {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    background: rgba(107,122,66,0.16);
    border: 1px solid rgba(107,122,66,0.4);
    border-radius: 12px;
    padding: 0.6rem 0.85rem;
    margin-bottom: 0.6rem;
    font-size: 0.86rem;
    font-weight: 600;
    color: #b7cf8a !important;
}

.sb-animal-pill {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    background: rgba(244,236,220,0.04);
    border: 1px solid rgba(244,236,220,0.10);
    border-radius: 10px;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.4rem;
    font-size: 0.88rem;
}

/* ---------- Hero ---------- */
.hero {
    padding: 52px 55px;
    border-radius: 30px;
    margin-bottom: 28px;
    background:
        linear-gradient(135deg, rgba(212,160,23,0.14), transparent 45%),
        linear-gradient(145deg, #1c160e, #100d08);
    border: 1px solid rgba(212,160,23,0.22);
    box-shadow: 0 25px 70px rgba(0,0,0,.45);
    position: relative;
    overflow: hidden;
}

.hero::after {
    content: "";
    position: absolute;
    top: -60px;
    right: -60px;
    width: 220px;
    height: 220px;
    background: rgba(212,160,23,0.06);
    border-radius: 50%;
}

.hero .eyebrow {
    color: #e8bd52;
    letter-spacing: 3px;
    font-weight: 800;
    font-size: 12px;
}

.hero h1 {
    font-size: clamp(40px, 6vw, 76px);
    line-height: .98;
    letter-spacing: -4px;
    margin: 12px 0;
    color: #f4ecdc;
}

.hero h1 span {
    color: #bf572c;
}

.hero p {
    color: #cabb9d;
    max-width: 720px;
    font-size: 17px;
    line-height: 1.7;
}

/* ---------- Metric cards ---------- */
.metric {
    background: rgba(244,236,220,0.035);
    border: 1px solid rgba(244,236,220,0.10);
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 10px 26px rgba(0,0,0,0.3);
}

.metric .number {
    color: #e8bd52;
    font-size: 28px;
    font-weight: 900;
    font-family: 'Poppins', sans-serif;
}

.metric .label {
    color: #9c8e73;
    font-size: 12px;
    margin-top: 4px;
    letter-spacing: 0.5px;
}

/* ---------- Prediction card ---------- */
.prediction-card {
    padding: 30px;
    border-radius: 24px;
    background: linear-gradient(145deg, #1c160e, #100d08);
    border: 1px solid rgba(212,160,23,0.28);
    text-align: center;
    box-shadow: 0 20px 45px rgba(0,0,0,0.35);
}

.prediction-label {
    color: #9c8e73;
    font-size: 11px;
    letter-spacing: 3px;
    font-weight: 800;
}

.prediction-name {
    color: #e8bd52;
    font-size: 42px;
    font-weight: 900;
    margin: 8px 0;
    font-family: 'Poppins', sans-serif;
}

.confidence {
    color: #e5dcc6;
    font-size: 17px;
}

/* ---------- File uploader ---------- */
div[data-testid="stFileUploader"] {
    border-radius: 22px;
}

div[data-testid="stFileUploaderDropzone"] {
    background:
        radial-gradient(circle at 20% 15%, rgba(212,160,23,0.09), transparent 55%),
        radial-gradient(circle at 80% 85%, rgba(191,87,44,0.08), transparent 55%),
        rgba(244,236,220,0.03);
    border: 2px dashed rgba(212,160,23,0.45);
    border-radius: 18px;
}

div[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(212,160,23,0.75);
}

div[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #bf572c, #d4a017) !important;
    color: #100d08 !important;
    border: none !important;
    font-weight: 800 !important;
    border-radius: 10px !important;
}

/* ---------- Buttons ---------- */
.stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(212,160,23,.35);
    background: linear-gradient(135deg, #bf572c, #d4a017);
    color: #100d08;
    font-weight: 900;
    box-shadow: 0 8px 20px rgba(212,160,23,0.28);
}

/* ---------- Progress bar ---------- */
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #bf572c, #d4a017, #6b7a42) !important;
}

/* ---------- Animal cards ---------- */
.animal-card {
    padding: 22px;
    background: rgba(244,236,220,0.035);
    border: 1px solid rgba(244,236,220,0.10);
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 10px 24px rgba(0,0,0,0.28);
}

.animal-card .emoji {
    font-size: 45px;
}

.animal-card b {
    display: block;
    margin-top: 10px;
    color: #f4ecdc;
}

hr { border-color: rgba(244,236,220,0.12) !important; }

.footer {
    text-align: center;
    color: #8a7d63;
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

emoji_map = {
    "buffalo": "🦬",
    "elephant": "🐘",
    "rhino": "🦏",
    "zebra": "🦓"
}

with st.sidebar:

    st.markdown(
        """
        <div class="sb-logo-wrap">
            <div class="sb-logo-badge">🦓</div>
            <div class="sb-title">Wildlife Vision</div>
            <div class="sb-subtitle">Deep Learning Classification</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="sb-section-label"><span class="icon-chip-sm">🤖</span> Model</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sb-status-chip">● Model Ready</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sb-card">
            <p>🧠 <b>Architecture:</b> EfficientNetV2-S</p>
            <p>📐 <b>Input:</b> 384 × 384</p>
            <p>🎯 <b>Classes:</b> 4</p>
            <p>📊 <b>Validation Accuracy:</b> 98.01%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="sb-section-label"><span class="icon-chip-sm">🐾</span> Supported Wildlife</div>',
        unsafe_allow_html=True,
    )

    animals_html = "".join(
        f'<div class="sb-animal-pill">{emoji_map.get(animal.lower(), "🐾")} {animal.title()}</div>'
        for animal in classes
    )
    st.markdown(animals_html, unsafe_allow_html=True)

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