import json
import threading
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import WebRtcMode, webrtc_streamer

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Emotion_Detection.keras"
LABELS_PATH = BASE_DIR / "emotion_labels.json"
CASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"

st.set_page_config(page_title="Live Facial Emotion Detection", layout="centered")
st.title("Live Facial Emotion Detection")
st.write(
    "Live webcam mode processes video frames continuously. Allow camera "
    "access in your browser when prompted."
)
st.info(
    "This app uses facial-expression classification to estimate expressions. "
    "Predictions may be incorrect and are not a reliable measure of a person's "
    "internal emotional state."
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH.name}")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_resource
def load_detector():
    detector = cv2.CascadeClassifier(str(CASCADE_PATH))
    if detector.empty():
        raise RuntimeError(f"Could not load face detector: {CASCADE_PATH.name}")
    return detector


@st.cache_data
def load_labels():
    with LABELS_PATH.open(encoding="utf-8") as file:
        mapping = json.load(file)
    return {int(index): label for label, index in mapping.items()}


def has_rescaling(layer):
    if isinstance(layer, tf.keras.layers.Rescaling):
        return True
    return any(has_rescaling(child) for child in getattr(layer, "layers", []))


def classify(gray_face, model, labels):
    shape = model.input_shape
    if isinstance(shape, list):
        if len(shape) != 1:
            raise ValueError("Only a single-input model is supported.")
        shape = shape[0]
    if len(shape) != 4 or shape[1] is None or shape[2] is None:
        raise ValueError(f"Unsupported model input shape: {shape}")

    height, width = int(shape[1]), int(shape[2])
    channels = int(shape[3])
    face = cv2.resize(gray_face, (width, height))
    if channels == 1:
        face = face[..., None]
    elif channels == 3:
        face = np.repeat(face[..., None], 3, axis=-1)
    else:
        raise ValueError(f"Unsupported channel count: {channels}")

    batch = face[None, ...].astype(np.float32)
    if not has_rescaling(model):
        batch /= 255.0

    with MODEL_LOCK:
        scores = np.asarray(model.predict(batch, verbose=0))[0]

    if scores.ndim != 1 or len(scores) != len(labels):
        raise ValueError("Model output does not match emotion labels.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Model returned invalid scores.")
    index = int(np.argmax(scores))
    return labels[index], float(scores[index])


MODEL_LOCK = threading.Lock()


def process_video_frame(frame):
    image = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = load_detector()
    model = load_model()
    labels = load_labels()

    faces = detector.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)
    )
    for x, y, w, h in faces:
        try:
            emotion, score = classify(gray[y:y + h, x:x + w], model, labels)
            caption = f"{emotion.title()}  {score:.0%}"
        except Exception:
            caption = "Prediction unavailable"

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 210, 0), 2)
        cv2.putText(
            image, caption, (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 210, 0), 2, cv2.LINE_AA
        )

    if len(faces) == 0:
        cv2.putText(
            image, "No face detected", (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 210, 0), 2, cv2.LINE_AA
        )

    return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_streamer(
    key="live-emotion-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=process_video_frame,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.caption(
    "If the camera does not start, allow camera access in your browser and "
    "check whether your network or firewall blocks WebRTC."
)
