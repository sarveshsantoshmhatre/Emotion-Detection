"""Train a compact 5-class FER-2013 emotion model in a hosted notebook.

Designed for Kaggle Notebooks with a GPU accelerator enabled. The script can also
run elsewhere if --data-dir points to a FER-2013 train/test directory.

Expected data layout:
  DATA_DIR/
    train/{angry,happy,neutral,sad,surprise}/
    test/{angry,happy,neutral,sad,surprise}/

The class order is explicit and matches test.py:
  0 Angry, 1 Happy, 2 Neutral, 3 Sad, 4 Surprise

This intentionally excludes FER-2013's disgust and fear classes because the
current webcam app has five output classes. It does not modify the existing model.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


CLASS_NAMES = ["angry", "happy", "neutral", "sad", "surprise"]
IMAGE_SIZE = (48, 48)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a compact FER-2013 emotion CNN")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing train/ and test/; auto-detected in Kaggle if omitted",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/emotion-output"),
        help="Directory where the model and evaluation reports are saved",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def valid_split_dir(path):
    if not path.is_dir():
        return False
    return all(
        (path / name).is_dir()
        and any(
            file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
            for file in (path / name).iterdir()
        )
        for name in CLASS_NAMES
    )


def valid_dataset_root(root):
    return root.is_dir() and valid_split_dir(root / "train") and valid_split_dir(root / "test")


def locate_dataset(explicit_dir):
    if explicit_dir is not None:
        explicit_dir = explicit_dir.expanduser().resolve()
        if valid_dataset_root(explicit_dir):
            return explicit_dir
        raise SystemExit(
            f"Could not find the expected train/test folder layout in {explicit_dir}.\n"
            f"Required class folders: {', '.join(CLASS_NAMES)}"
        )

    search_roots = [Path("/kaggle/input"), Path.cwd() / "datasets" / "fer2013"]
    checked = set()
    for search_root in search_roots:
        if not search_root.exists():
            continue
        search_root = search_root.resolve()
        if search_root in checked:
            continue
        checked.add(search_root)

        if valid_dataset_root(search_root):
            return search_root
        try:
            for train_folder in search_root.rglob("train"):
                candidate = train_folder.parent
                if valid_dataset_root(candidate):
                    return candidate
        except OSError:
            continue

    raise SystemExit(
        "FER-2013 was not found. In Kaggle, use Add Input to attach the FER-2013 "
        "dataset, then rerun. Alternatively pass --data-dir /path/to/fer2013."
    )


def count_images(folder):
    return sum(
        1
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_model():
    # Mild augmentation helps with webcam lighting and pose variation. Keep
    # transformations conservative because exaggerated faces can change the label.
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.04),
            layers.RandomContrast(0.12),
        ],
        name="training_augmentation",
    )

    inputs = keras.Input(shape=(48, 48, 1), name="grayscale_face")
    x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)

    for filters, dropout_rate in [(32, 0.15), (64, 0.20), (128, 0.25)]:
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Dropout(dropout_rate)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax", name="emotion_scores")(x)

    model = keras.Model(inputs, outputs, name="fer2013_emotion_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive integers.")

    keras.utils.set_random_seed(args.seed)
    dataset_root = locate_dataset(args.data_dir)
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Dataset:", dataset_root)
    print("Class order:", list(enumerate(CLASS_NAMES)))
    print("TensorFlow:", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs:", [gpu.name for gpu in gpus] if gpus else "None (training will use CPU)")
    if not gpus:
        print("Warning: no GPU detected. A hosted GPU accelerator is recommended.")

    for split_dir in [train_dir, test_dir]:
        print(f"\n{split_dir.name} image counts:")
        for class_name in CLASS_NAMES:
            print(f"  {class_name}: {count_images(split_dir / class_name)}")

    common = dict(
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        image_size=IMAGE_SIZE,
        batch_size=args.batch_size,
    )
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.1,
        subset="training",
        seed=args.seed,
        shuffle=True,
        **common,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.1,
        subset="validation",
        seed=args.seed,
        shuffle=False,
        **common,
    )
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        shuffle=False,
        **common,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    # Use a slightly stronger inverse-frequency weighting than the baseline.
    # The exponent 0.75 avoids extreme weights while emphasizing underrepresented
    # classes. Compare the new evaluation.json against the previous run.
    counts = {name: count_images(train_dir / name) for name in CLASS_NAMES}
    total = sum(counts.values())
    mean_count = total / len(CLASS_NAMES)
    class_weights = {
        index: (mean_count / max(counts[name], 1)) ** 0.75
        for index, name in enumerate(CLASS_NAMES)
    }
    print("\nTraining counts:", counts)
    print("Class weights:", class_weights)

    model = build_model()
    model.summary()

    best_model = output_dir / "best_emotion_model.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(best_model),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(output_dir / "training_history.csv")),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # Evaluate the best saved checkpoint, rather than assuming the last epoch is best.
    if best_model.exists():
        model = keras.models.load_model(best_model)

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    true_labels = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)
    probabilities = model.predict(test_ds, verbose=1)
    predicted_labels = probabilities.argmax(axis=1)
    confusion = tf.math.confusion_matrix(
        true_labels,
        predicted_labels,
        num_classes=len(CLASS_NAMES),
    ).numpy()

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "classes": CLASS_NAMES,
        "class_indices": {name: index for index, name in enumerate(CLASS_NAMES)},
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
        "per_class": {},
    }
    for index, name in enumerate(CLASS_NAMES):
        tp = int(confusion[index, index])
        support = int(confusion[index, :].sum())
        predicted_count = int(confusion[:, index].sum())
        metrics["per_class"][name] = {
            "recall": tp / support if support else 0.0,
            "precision": tp / predicted_count if predicted_count else 0.0,
            "support": support,
        }

    (output_dir / "evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output_dir / "emotion_labels.json").write_text(
        json.dumps(metrics["class_indices"], indent=2) + "\n"
    )
    with (output_dir / "confusion_matrix.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *CLASS_NAMES])
        for name, row in zip(CLASS_NAMES, confusion):
            writer.writerow([name, *row.tolist()])

    # Save in the native Keras format. Legacy HDF5 serialization can fail
    # with Keras 3 when models include preprocessing/augmentation layers.
    model.save(output_dir / "Emotion_Detection.keras")

    print("\nEvaluation results:")
    print(json.dumps(metrics, indent=2))
    print("\nSaved artifacts:")
    for path in sorted(output_dir.iterdir()):
        print(f"  {path} ({path.stat().st_size:,} bytes)")
    print(
        "\nDownload Emotion_Detection.keras and replace the local model only after "
        "reviewing evaluation.json. Keep a backup of the current model."
    )


if __name__ == "__main__":
    main()
