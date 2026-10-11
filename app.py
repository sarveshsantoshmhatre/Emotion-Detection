import json
import threading
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Emotion_Detection.keras"
LABELS_PATH = BASE_DIR / "emotion_labels.json"
CASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"

st.set_page_config(
    page_title="EmotionLens | Facial Expression AI",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .stApp { background: radial-gradient(circle at 10% 0%, #182b42 0, #0b1020 40%, #080b14 100%); }
      .block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 3rem; }
      .hero { padding: 1.7rem 1.8rem; border: 1px solid rgba(148,163,184,.22); border-radius: 22px;
              background: linear-gradient(120deg, rgba(23,45,70,.92), rgba(15,23,42,.85)); margin-bottom: 1.25rem; }
      .eyebrow { text-transform: uppercase; letter-spacing: .16em; font-size: .75rem; color: #7dd3fc; font-weight: 700; }
      .hero h1 { margin: .45rem 0; font-size: clamp(2rem, 4vw, 3rem); color: #f8fafc; }
      .hero p { color: #cbd5e1; margin-bottom: 0; font-size: 1.02rem; }
      .panel { border: 1px solid rgba(148,163,184,.2); border-radius: 18px; padding: 1rem 1.15rem;
               background: rgba(15,23,42,.72); }
      .small-note { color: #94a3b8; font-size: .86rem; }
      div[data-testid="stMetric"] { border: 1px solid rgba(148,163,184,.2); border-radius: 14px; padding: .8rem; background: rgba(15,23,42,.65); }
      div[data-testid="stRadio"] > label { color: #e2e8f0; font-weight: 650; }
      .stButton > button { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Computer vision · Live inference</div>
      <h1>EmotionLens</h1>
      <p>Facial-expression recognition from your webcam or an image you choose.</p>
    </div>
    """,
    unsafe_allow_html=True,
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
    return labels[index], float(scores[index]), scores


MODEL_LOCK = threading.Lock()


def draw_prediction(image, gray, faces, model, labels, draw_all=True):
    results = []
    if len(faces) > 0:
        for x, y, w, h in faces:
            try:
                emotion, score, scores = classify(gray[y:y + h, x:x + w], model, labels)
                results.append((emotion, score, scores))
                cv2.rectangle(image, (x, y), (x + w, y + h), (45, 212, 191), 2)
                cv2.rectangle(image, (x, max(0, y - 30)), (x + w, y), (15, 23, 42), -1)
                cv2.putText(
                    image, f"{emotion.title()} {score:.0%}", (x + 5, max(20, y - 9)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA
                )
                if not draw_all:
                    break
            except Exception:
                continue
    return image, results


def process_video_frame(frame):
    image = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = load_detector().detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)
    )
    annotated, _ = draw_prediction(image, gray, faces, load_model(), load_labels())
    if len(faces) == 0:
        cv2.putText(
            annotated, "No face detected", (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (45, 212, 191), 2, cv2.LINE_AA
        )
    return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def get_rtc_configuration():
    """Use optional TURN credentials from Streamlit secrets; STUN alone may fail behind restrictive NATs."""
    default_servers = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
    try:
        rtc_secrets = st.secrets.get("webrtc", {})
        configured_servers = rtc_secrets.get("ice_servers", [])
        if configured_servers:
            return {"iceServers": list(configured_servers)}, True
    except Exception:
        pass
    return {"iceServers": default_servers}, False


RTC_CONFIGURATION, HAS_TURN_CONFIGURATION = get_rtc_configuration()

st.markdown("### Choose your experience")
mode = st.radio(
    "Input mode",
    ["Live webcam", "Upload an image"],
    horizontal=True,
    label_visibility="collapsed",
)

if mode == "Live webcam":
    st.markdown(
        '<div class="panel"><strong>Live analysis</strong><p class="small-note">'
        'Allow camera access in your browser. Predictions are drawn on the video as frames are processed.</p></div>',
        unsafe_allow_html=True,
    )
    webrtc_streamer(
        key="live-emotion-lens",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=process_video_frame,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=RTC_CONFIGURATION,
        async_processing=True,
    )
    if not HAS_TURN_CONFIGURATION:
        st.info(
            "If the webcam stays on “Connecting”, this hosting/network combination may require a TURN relay. "
            "Add TURN server details in Streamlit app settings → Secrets under [webrtc] ice_servers. "
            "STUN-only connections do not work on every network."
        )
    st.caption("Allow camera access. If it still cannot connect, try a different network or configure TURN in app Secrets.")
else:
    uploaded = st.file_uploader(
        "Choose a face photo", type=["jpg", "jpeg", "png"],
        help="For best results, use a clear, front-facing image with good lighting.",
    )
    if uploaded is not None:
        try:
            pil_image = Image.open(uploaded).convert("RGB")
            rgb = np.asarray(pil_image)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            faces = load_detector().detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)
            )
            annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            annotated, results = draw_prediction(
                annotated, gray, faces, load_model(), load_labels()
            )
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            left, right = st.columns([1.25, 1])
            with left:
                st.image(annotated, caption="Analyzed image", use_container_width=True)
            with right:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown("#### Analysis")
                if not results:
                    st.warning("No face detected. Try a clearer, front-facing photo.")
                else:
                    emotion, score, scores = results[0]
                    st.metric("Predicted expression", emotion.title())
                    st.metric("Model score", f"{score:.1%}")
                    st.caption("Scores are not calibrated probabilities.")
                    st.markdown("**Class scores**")
                    st.bar_chart(
                        {load_labels()[i].title(): float(scores[i]) for i in range(len(scores))}
                    )
                st.markdown("</div>", unsafe_allow_html=True)
        except Exception as error:
            st.error(f"Could not analyze this image: {error}")

st.markdown("---")
st.markdown(
    '<p class="small-note">EmotionLens · Model predictions are estimates of visible facial expressions, '
    'not a reliable measurement of internal emotional state. Images are processed for inference by this app.</p>',
    unsafe_allow_html=True,
)
