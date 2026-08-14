
import os
import json
import uuid
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "wildlife_classifier_final.keras"
CLASSES_PATH = BASE_DIR / "models" / "class_names.json"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"

IMAGE_SIZE = 384
DEFAULT_CLASSES = ["buffalo", "elephant", "rhino", "zebra"]
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "wildlife-classifier-secret")


def get_classes():
    if CLASSES_PATH.exists():
        try:
            data = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list) and len(data) == 4:
                return data
        except Exception:
            pass
    return DEFAULT_CLASSES


CLASS_NAMES = get_classes()
MODEL = None


def get_model():
    global MODEL

    if MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model not found. Put wildlife_classifier_final.keras inside models/."
            )

        MODEL = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

    return MODEL


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    predictions = get_model().predict(array, verbose=0)[0]

    # Softmax safety
    if not np.isclose(np.sum(predictions), 1.0, atol=0.02):
        exp = np.exp(predictions - np.max(predictions))
        predictions = exp / np.sum(exp)

    order = np.argsort(predictions)[::-1]

    results = []
    for index in order:
        results.append({
            "class_name": CLASS_NAMES[int(index)],
            "confidence": float(predictions[index] * 100)
        })

    return results


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    results = None
    image_url = None

    if request.method == "POST":
        uploaded = request.files.get("image")

        if not uploaded or not uploaded.filename:
            flash("Please select a wildlife image.", "error")
            return redirect(url_for("index"))

        if not allowed_file(uploaded.filename):
            flash("Supported formats: JPG, JPEG, PNG, WEBP and BMP.", "error")
            return redirect(url_for("index"))

        extension = uploaded.filename.rsplit(".", 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{extension}"
        destination = UPLOAD_DIR / filename

        try:
            uploaded.save(destination)

            results = predict_image(destination)

            prediction = results[0]["class_name"]
            confidence = results[0]["confidence"]

            image_url = url_for(
                "static",
                filename=f"uploads/{filename}"
            )

        except Exception as exc:
            destination.unlink(missing_ok=True)
            flash(f"Prediction failed: {exc}", "error")

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        results=results,
        image_url=image_url,
        classes=CLASS_NAMES
    )


@app.route("/health")
def health():
    return {
        "status": "healthy",
        "model_available": MODEL_PATH.exists(),
        "classes": CLASS_NAMES
    }


@app.route("/about")
def about():
    return render_template("about.html", classes=CLASS_NAMES)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
