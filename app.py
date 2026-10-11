import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Emotion_Detection.keras"
LABELS_PATH = BASE_DIR / "emotion_labels.json"
CASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"

st.set_page_config(page_title="Facial Emotion Detection", layout="centered")
st.title("Facial Emotion Detection")
st.write(
    "Upload a photo or use your browser camera to estimate the facial "
    "expression. The model predicts one face per image."
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_resource
def load_face_detector():
    detector = cv2.CascadeClassifier(str(CASCADE_PATH))
    if detector.empty():
        raise RuntimeError(f"Could not load {CASCADE_PATH.name}")
    return detector


@st.cache_data
def load_labels():
    with LABELS_PATH.open(encoding="utf-8") as file:
        raw = json.load(file)
    return {int(index): name for name, index in raw.items()}


def contains_rescaling(layer):
    if isinstance(layer, tf.keras.layers.Rescaling):
        return True
    return any(
        contains_rescaling(child)
        for child in getattr(layer, "layers", [])
    )


def analyze_image(pil_image):
    model = load_model()
    detector = load_face_detector()
    labels = load_labels()

    rgb = np.asarray(pil_image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return rgb, None, None, None

    x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
    face = gray[y:y + h, x:x + w]

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        if len(input_shape) != 1:
            raise ValueError("This app supports models with one input tensor.")
        input_shape = input_shape[0]

    if len(input_shape) != 4 or input_shape[1] is None or input_shape[2] is None:
        raise ValueError(f"Unsupported model input shape: {input_shape}")

    height, width = int(input_shape[1]), int(input_shape[2])
    channels = int(input_shape[3])
    face = cv2.resize(face, (width, height))

    if channels == 1:
        face = face[..., np.newaxis]
    elif channels == 3:
        face = np.repeat(face[..., np.newaxis], 3, axis=-1)
    else:
        raise ValueError(f"Unsupported model channel count: {channels}")

    batch = face[np.newaxis, ...].astype(np.float32)
    if not contains_rescaling(model):
        batch /= 255.0

    scores = np.asarray(model.predict(batch, verbose=0))[0]
    if scores.ndim != 1 or len(scores) != len(labels):
        raise ValueError(
            f"Model output shape {scores.shape} does not match "
            f"{len(labels)} emotion labels."
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("The model returned invalid numeric scores.")

    annotated = rgb.copy()
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
    return annotated, (x, y, w, h), scores, labels


source = st.radio(
    "Image source",
    ["Upload an image", "Use webcam"],
    horizontal=True,
)
if source == "Upload an image":
    image_file = st.file_uploader(
        "Choose a photo", type=["jpg", "jpeg", "png"]
    )
else:
    image_file = st.camera_input("Take a picture")

if image_file is not None:
    try:
        image = Image.open(image_file).convert("RGB")
        with st.spinner("Analyzing image..."):
            annotated, face_box, scores, labels = analyze_image(image)

        st.image(annotated, caption="Input image", use_container_width=True)

        if face_box is None:
            st.warning(
                "No face was detected. Try a clear, front-facing photo "
                "with good lighting."
            )
        else:
            best_index = int(np.argmax(scores))
            st.subheader(f"Predicted expression: {labels[best_index].title()}")
            st.caption(
                "Scores are model outputs, not calibrated probabilities. "
                "Facial-expression predictions are estimates and may be wrong."
            )
            st.bar_chart(
                {
                    labels[i].title(): float(scores[i])
                    for i in range(len(scores))
                }
            )
    except Exception as error:
        st.error(f"Could not analyze this image: {error}")
