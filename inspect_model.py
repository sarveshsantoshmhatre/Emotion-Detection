"""Inspect the architecture and stored metadata of Emotion_Detection.h5.

Run from the repository root:
    python inspect_model.py
This is read-only; it does not modify the model or webcam application.
"""

import json
from pathlib import Path

import h5py

try:
    import tf_keras
except ImportError as exc:
    raise SystemExit(
        "The legacy Keras loader is required for this model. Install it with:\n"
        "  python -m pip install tf-keras==2.21.0"
    ) from exc


MODEL_PATH = Path(__file__).resolve().parent / "Emotion_Detection.h5"
POSSIBLE_LABEL_KEYS = ("label", "class", "emotion", "index", "mapping")


def describe_shape(value):
    try:
        return str(tuple(value))
    except TypeError:
        return str(value)


def main():
    if not MODEL_PATH.is_file():
        raise SystemExit(f"Model file not found: {MODEL_PATH}")

    print(f"Model file: {MODEL_PATH}")
    print(f"File size: {MODEL_PATH.stat().st_size:,} bytes")

    with h5py.File(MODEL_PATH, "r") as h5file:
        keys = list(h5file.attrs.keys())
        print("\nHDF5 root attributes:", keys or "(none)")
        found_label_metadata = False

        for key in keys:
            if any(token in key.lower() for token in POSSIBLE_LABEL_KEYS):
                value = h5file.attrs[key]
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                print(f"Possible label metadata [{key}]: {value}")
                found_label_metadata = True

        config = h5file.attrs.get("model_config")
        if config is not None:
            if isinstance(config, bytes):
                config = config.decode("utf-8", errors="replace")
            try:
                parsed = json.loads(config)
                print("\nSaved model class:", parsed.get("class_name", "(not specified)"))
            except (TypeError, json.JSONDecodeError):
                print("\nA model_config attribute exists but could not be parsed.")
        else:
            print("\nNo model_config attribute was found in the HDF5 file.")

        if not found_label_metadata:
            print(
                "No obvious class/label mapping was found in root HDF5 attributes. "
                "That does not prove no custom metadata exists elsewhere in the file."
            )

    print("\nLoading model with tf-keras...")
    try:
        model = tf_keras.models.load_model(str(MODEL_PATH), compile=False)
    except Exception as exc:
        raise SystemExit(f"Model loading failed: {type(exc).__name__}: {exc}") from exc

    print("Input shape:", describe_shape(model.input_shape))
    print("Output shape:", describe_shape(model.output_shape))
    print("Total parameters:", f"{model.count_params():,}")
    print("\nLayers:")
    for index, layer in enumerate(model.layers):
        try:
            output_shape = describe_shape(layer.output.shape)
        except Exception:
            output_shape = "(unavailable)"
        params = layer.count_params()
        print(
            f"{index:>3}: {layer.name:<32} "
            f"{layer.__class__.__name__:<24} "
            f"output={output_shape:<28} params={params:,}"
        )

    print(
        "\nImportant: the network architecture and weights do not, by themselves, "
        "reliably identify which emotion name belongs to each output index. "
        "For that, the original training class_indices/class_names mapping is needed."
    )


if __name__ == "__main__":
    main()
