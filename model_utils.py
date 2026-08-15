import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import numpy as np
import tensorflow as tf

from config import MODEL_PATH, CLASSES_PATH, DEFAULT_CLASSES
from preprocess import prepare_image

MODEL = None

def load_classes():
    try:
        data = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list) and len(data) == 4:
            return data
    except Exception:
        pass
    return DEFAULT_CLASSES

CLASS_NAMES = load_classes()

try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    pass

def get_model():
    global MODEL
    if MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return MODEL

def predict(image_path):
    probabilities = get_model().predict(prepare_image(image_path), verbose=0)[0]
    if np.any(probabilities < 0) or not np.isclose(float(np.sum(probabilities)), 1.0, atol=0.02):
        exp = np.exp(probabilities - np.max(probabilities))
        probabilities = exp / np.sum(exp)
    order = np.argsort(probabilities)[::-1]
    return [
        {"class_name": CLASS_NAMES[int(i)], "confidence": float(probabilities[int(i)] * 100)}
        for i in order
    ]
