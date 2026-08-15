import os

# ============================================================
# RENDER CPU CONFIGURATION
# IMPORTANT: Must be BEFORE importing TensorFlow
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import uuid
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash


# ============================================================
# PATHS & CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "wildlife_classifier_final.keras"
)

CLASSES_PATH = (
    BASE_DIR
    / "models"
    / "class_names.json"
)

UPLOAD_DIR = (
    BASE_DIR
    / "static"
    / "uploads"
)

IMAGE_SIZE = 384

DEFAULT_CLASSES = [
    "buffalo",
    "elephant",
    "rhino",
    "zebra"
]

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp",
    "bmp"
}

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TENSORFLOW CPU CONFIGURATION
# ============================================================

try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    pass


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "wildlife-classifier-secret"
)

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def get_classes():

    if CLASSES_PATH.exists():

        try:

            data = json.loads(
                CLASSES_PATH.read_text(
                    encoding="utf-8"
                )
            )

            if (
                isinstance(data, list)
                and len(data) == 4
            ):
                return data

        except Exception:
            pass

    return DEFAULT_CLASSES


CLASS_NAMES = get_classes()


# ============================================================
# MODEL
# ============================================================

MODEL = None


def get_model():
    """
    Load the model only when a prediction is requested.

    This is important for Render because the health
    endpoint does not need to load the TensorFlow model.
    """

    global MODEL

    if MODEL is None:

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                "Model not found. "
                "Put wildlife_classifier_final.keras "
                "inside the models folder."
            )

        MODEL = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

    return MODEL


# ============================================================
# FILE VALIDATION
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def prepare_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        Image.Resampling.BILINEAR
    )

    array = np.asarray(
        image,
        dtype=np.float32
    )

    array = np.expand_dims(
        array,
        axis=0
    )

    return array


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image_path):

    image_array = prepare_image(
        image_path
    )

    model = get_model()

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    # --------------------------------------------------------
    # Softmax safety
    # --------------------------------------------------------

    if (
        np.any(predictions < 0)
        or
        not np.isclose(
            np.sum(predictions),
            1.0,
            atol=0.02
        )
    ):

        exp = np.exp(
            predictions
            -
            np.max(predictions)
        )

        predictions = (
            exp
            /
            np.sum(exp)
        )

    # --------------------------------------------------------
    # Sort predictions
    # --------------------------------------------------------

    order = np.argsort(
        predictions
    )[::-1]

    results = []

    for index in order:

        index = int(index)

        results.append(
            {
                "class_name": CLASS_NAMES[index],

                "confidence": float(
                    predictions[index]
                    * 100
                )
            }
        )

    return results


# ============================================================
# HOME / PREDICTION ROUTE
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():

    prediction = None
    confidence = None
    results = None
    image_url = None

    if request.method == "POST":

        uploaded = request.files.get(
            "image"
        )

        # ----------------------------------------------------
        # Validate upload
        # ----------------------------------------------------

        if (
            uploaded is None
            or
            not uploaded.filename
        ):

            flash(
                "Please select a wildlife image.",
                "error"
            )

            return redirect(
                url_for("index")
            )

        # ----------------------------------------------------
        # Validate extension
        # ----------------------------------------------------

        if not allowed_file(
            uploaded.filename
        ):

            flash(
                "Supported formats: "
                "JPG, JPEG, PNG, WEBP and BMP.",
                "error"
            )

            return redirect(
                url_for("index")
            )

        # ----------------------------------------------------
        # Generate unique filename
        # ----------------------------------------------------

        extension = (
            uploaded.filename
            .rsplit(".", 1)[1]
            .lower()
        )

        filename = (
            f"{uuid.uuid4().hex}"
            f".{extension}"
        )

        destination = (
            UPLOAD_DIR
            /
            filename
        )

        # ----------------------------------------------------
        # Save and predict
        # ----------------------------------------------------

        try:

            uploaded.save(
                destination
            )

            results = predict_image(
                destination
            )

            if not results:

                raise RuntimeError(
                    "The model returned "
                    "no predictions."
                )

            prediction = (
                results[0]["class_name"]
            )

            confidence = (
                results[0]["confidence"]
            )

            image_url = url_for(
                "static",
                filename=(
                    f"uploads/{filename}"
                )
            )

        except Exception as exc:

            # Remove failed upload
            destination.unlink(
                missing_ok=True
            )

            # Write complete error to Render logs
            app.logger.exception(
                "Wildlife prediction failed"
            )

            flash(
                f"Prediction failed: {exc}",
                "error"
            )

    return render_template(
        "index.html",

        prediction=prediction,

        confidence=confidence,

        results=results,

        image_url=image_url,

        classes=CLASS_NAMES
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    """
    Lightweight Render health endpoint.

    IMPORTANT:
    This does NOT load TensorFlow model.
    """

    return {
        "status": "healthy",

        "model_available": (
            MODEL_PATH.exists()
        ),

        "classes": CLASS_NAMES,

        "model_loading": "lazy"
    }, 200


# ============================================================
# ABOUT
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html",
        classes=CLASS_NAMES
    )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )