"""Run real-time facial-expression classification from a webcam.

Usage:
    python test.py
    python test.py --camera 1
    python test.py --model /path/to/model.h5
"""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

try:
    from tensorflow.keras.models import load_model
except ImportError:
    try:
        from keras.models import load_model
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow/Keras is not installed. Install the project's ML dependencies "
            "before running this script."
        ) from exc


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LABELS = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]


def parse_args():
    parser = argparse.ArgumentParser(description="Webcam facial-expression detector")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument(
        "--model",
        type=Path,
        default=BASE_DIR / "Emotion_Detection.h5",
        help="Path to a saved Keras .h5 model",
    )
    parser.add_argument(
        "--cascade",
        type=Path,
        default=BASE_DIR / "haarcascade_frontalface_default.xml",
        help="Path to the OpenCV Haar cascade XML file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.model.is_file():
        raise SystemExit(f"Model file not found: {args.model}")
    if not args.cascade.is_file():
        raise SystemExit(f"Face-detection cascade not found: {args.cascade}")

    face_detector = cv2.CascadeClassifier(str(args.cascade))
    if face_detector.empty():
        raise SystemExit(f"Could not load face-detection cascade: {args.cascade}")

    try:
        model = load_model(str(args.model), compile=False)
    except Exception as exc:
        raise SystemExit(
            f"Could not load model '{args.model}': {exc}\n"
            "Check that the model is a valid Keras model and that your TensorFlow/Keras "
            "version is compatible with the file."
        ) from exc

    # Use the model's declared input dimensions rather than assuming 48x48 grayscale.
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        raise SystemExit("This script supports single-input models only.")
    if len(input_shape) != 4:
        raise SystemExit(f"Expected a channels-last image model; got input shape {input_shape}.")

    _, model_height, model_width, channels = input_shape
    model_height = int(model_height or 48)
    model_width = int(model_width or 48)
    channels = int(channels or 1)
    if channels not in (1, 3):
        raise SystemExit(
            f"Unsupported model input channel count: {channels}. Expected 1 (grayscale) or 3 (RGB)."
        )

    output_shape = model.output_shape
    if isinstance(output_shape, list) or len(output_shape) < 2 or output_shape[-1] is None:
        raise SystemExit(f"Could not determine model output classes from shape {output_shape}.")
    class_count = int(output_shape[-1])
    if class_count != len(DEFAULT_LABELS):
        raise SystemExit(
            f"The model outputs {class_count} classes, but test.py defines "
            f"{len(DEFAULT_LABELS)} labels. Update DEFAULT_LABELS to match the training "
            "dataset's class_indices order before running predictions."
        )

    # CAP_DSHOW is Windows-specific; use the default backend for portability (including Ubuntu).
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        raise SystemExit(
            f"Could not open camera index {args.camera}. Check camera permissions, "
            "connection, and whether another application is using it. Try --camera 1."
        )

    print(f"Loaded model: {args.model}")
    print(f"Model input: {model_height}x{model_width}x{channels}")
    print("Press 'q' or Esc to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Warning: could not read a frame from the camera.", file=sys.stderr)
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
            )

            for (x, y, width, height) in faces:
                # Clip the region to frame bounds before preprocessing.
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(frame.shape[1], x + width), min(frame.shape[0], y + height)
                if x1 <= x0 or y1 <= y0:
                    continue

                face_bgr = frame[y0:y1, x0:x1]
                if channels == 1:
                    face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
                    interpolation = cv2.INTER_AREA
                else:
                    face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                    interpolation = cv2.INTER_AREA

                face = cv2.resize(face, (model_width, model_height), interpolation=interpolation)
                if channels == 1:
                    face = np.expand_dims(face, axis=-1)

                batch = np.expand_dims(face.astype(np.float32) / 255.0, axis=0)
                try:
                    scores = np.asarray(model.predict(batch, verbose=0))[0]
                except Exception as exc:
                    print(f"Prediction failed: {exc}", file=sys.stderr)
                    continue

                if scores.size != len(DEFAULT_LABELS):
                    print(
                        f"Unexpected prediction size {scores.size}; expected {len(DEFAULT_LABELS)}.",
                        file=sys.stderr,
                    )
                    continue

                class_index = int(np.argmax(scores))
                label = DEFAULT_LABELS[class_index]
                confidence = float(scores[class_index]) * 100.0

                cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                text_y = max(y0 - 10, 25)
                cv2.putText(
                    frame,
                    f"{label} ({confidence:.1f}%)",
                    (x0, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            if len(faces) == 0:
                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Emotion Detector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
