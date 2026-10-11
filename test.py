"""Run real-time facial-expression classification from a webcam.

Usage:
    python test.py
    python test.py --camera 1
    python test.py --model /path/to/model.keras

Native .keras models are preferred. Legacy HDF5 models may require the optional
 tf-keras compatibility package when running with TensorFlow/Keras 3.
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

try:
    from tensorflow.keras.models import load_model as modern_load_model
except ImportError:
    try:
        from keras.models import load_model as modern_load_model
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow/Keras is not installed. Install the project's ML dependencies "
            "before running this script."
        ) from exc

try:
    from tf_keras.models import load_model as legacy_load_model
except ImportError:
    legacy_load_model = None


BASE_DIR = Path(__file__).resolve().parent
def contains_rescaling(layer):
    """Find a Rescaling layer, including inside nested Keras models."""
    if layer.__class__.__name__ == "Rescaling":
        return True
    return any(contains_rescaling(child) for child in getattr(layer, "layers", []))


def load_emotion_labels(labels_path):
    """Load the output-index mapping written by train_kaggle.py."""
    if not labels_path.is_file():
        raise SystemExit(f"Emotion label mapping not found: {labels_path}")
    try:
        mapping = json.loads(labels_path.read_text(encoding="utf-8"))
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("expected a non-empty JSON object")
        indices = list(mapping.values())
        if not all(isinstance(index, int) for index in indices):
            raise ValueError("all label indices must be integers")
        if sorted(indices) != list(range(len(mapping))):
            raise ValueError("label indices must be unique and contiguous from zero")
        return [name.title() for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Invalid emotion label mapping in {labels_path}: {exc}") from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Webcam facial-expression detector")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument(
        "--debug-probabilities",
        action="store_true",
        help="Print raw model outputs for each emotion every 15 frames",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=BASE_DIR / "Emotion_Detection.keras",
        help="Path to a saved Keras .keras model (legacy .h5 is also supported)",
    )
    parser.add_argument(
        "--cascade",
        type=Path,
        default=BASE_DIR / "haarcascade_frontalface_default.xml",
        help="Path to the OpenCV Haar cascade XML file",
    )
    return parser.parse_args()


def load_emotion_model(model_path):
    """Load a model, retrying with Keras 2 compatibility for legacy H5 files."""
    try:
        return modern_load_model(str(model_path), compile=False)
    except Exception as modern_error:
        if legacy_load_model is None:
            raise SystemExit(
                f"Could not load model '{model_path}' with the installed TensorFlow/Keras: "
                f"{modern_error}\n\nThis may be a legacy Keras H5 model. With TensorFlow "
                "2.21, install the matching Keras 2 compatibility loader and retry:\n"
                "  python -m pip install tf-keras==2.21.0"
            ) from modern_error

        try:
            print(
                "Standard Keras loading failed; retrying with the legacy tf-keras loader.",
                file=sys.stderr,
            )
            model = legacy_load_model(str(model_path), compile=False)
            print("Model loaded with tf-keras compatibility mode.", file=sys.stderr)
            return model
        except Exception as legacy_error:
            raise SystemExit(
                f"Could not load model '{model_path}'.\n"
                f"TensorFlow/Keras loader error: {modern_error}\n"
                f"Legacy tf-keras loader error: {legacy_error}\n"
                "Confirm that the H5 file is intact and that its architecture is supported."
            ) from legacy_error


def main():
    args = parse_args()

    if not args.model.is_file():
        raise SystemExit(f"Model file not found: {args.model}")
    if not args.cascade.is_file():
        raise SystemExit(f"Face-detection cascade not found: {args.cascade}")

    face_detector = cv2.CascadeClassifier(str(args.cascade))
    if face_detector.empty():
        raise SystemExit(f"Could not load face-detection cascade: {args.cascade}")

    labels = load_emotion_labels(BASE_DIR / "emotion_labels.json")
    model = load_emotion_model(args.model)

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
    # The new training script embeds Rescaling(1/255) in the model; the legacy model does not.
    # Avoid dividing twice, which would make the new model receive near-zero pixel values.
    has_internal_rescaling = contains_rescaling(model)
    if has_internal_rescaling:
        print("Input normalization: inside model")
    else:
        print("Input normalization: in webcam script")

    class_count = int(output_shape[-1])
    if class_count != len(labels):
        raise SystemExit(
            f"The model outputs {class_count} classes, but emotion_labels.json defines "
            f"{len(labels)} labels. Check emotion_labels.json against the training "
            "dataset's class_indices order before running predictions."
        )

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

    frame_index = 0
    try:
        while True:
            frame_index += 1
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Warning: could not read a frame from the camera.", file=sys.stderr)
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
            )

            for (x, y, width, height) in faces:
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(frame.shape[1], x + width), min(frame.shape[0], y + height)
                if x1 <= x0 or y1 <= y0:
                    continue

                face_bgr = frame[y0:y1, x0:x1]
                if channels == 1:
                    face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
                else:
                    face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

                face = cv2.resize(
                    face, (model_width, model_height), interpolation=cv2.INTER_AREA
                )
                if channels == 1:
                    face = np.expand_dims(face, axis=-1)

                face_array = face.astype(np.float32)
                if not has_internal_rescaling:
                    face_array /= 255.0
                batch = np.expand_dims(face_array, axis=0)
                try:
                    scores = np.asarray(model.predict(batch, verbose=0))[0]
                    if args.debug_probabilities and frame_index % 15 == 0:
                        score_text = ", ".join(
                            f"{name}={float(score):.3f}"
                            for name, score in zip(labels, scores)
                        )
                        print(f"Model outputs (raw): {score_text}")
                except Exception as exc:
                    print(f"Prediction failed: {exc}", file=sys.stderr)
                    continue

                if scores.ndim != 1 or scores.size != len(labels):
                    print(
                        f"Unexpected prediction shape {scores.shape}; expected ({len(labels)},).",
                        file=sys.stderr,
                    )
                    continue
                if not np.all(np.isfinite(scores)):
                    print("Prediction returned non-finite scores; skipping this frame.", file=sys.stderr)
                    continue

                class_index = int(np.argmax(scores))
                label = labels[class_index]
                confidence = float(scores[class_index]) * 100.0

                cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} ({confidence:.1f}%)",
                    (x0, max(y0 - 10, 25)),
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
