# Emotion Detection

Real-time facial-expression classification using a webcam and a Keras model.

## Run the webcam demo

Use Python 3.10 or newer in a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python test.py
```

The app checks that the model and face detector exist, verifies that the camera opens, and exits cleanly if a camera frame cannot be read. If the default camera cannot be opened, try `python test.py --camera 1`.

Required files:

- `Emotion_Detection.keras` — trained model.
- `emotion_labels.json` — output-index to emotion-name mapping.
- `haarcascade_frontalface_default.xml` — face detector.
- `test.py` — webcam inference.

## Preprocessing compatibility

The current `train_kaggle.py` pipeline trains on **48 × 48 grayscale** faces and includes a `Rescaling(1/255)` layer inside the model. The inference app reads the loaded model's declared input dimensions and channel count, converts the webcam crop accordingly, and avoids applying the same rescaling twice. If you replace the model, replace `emotion_labels.json` from the same training run as well.

The removed legacy `train.py` used a different MobileNet pipeline (224 × 224 RGB). Do not mix a model and label map from different training runs. The current inference code adapts image size and channel count from model metadata, but correct normalization must still match the model's training pipeline.

## Training

`train_kaggle.py` trains a five-class model on a FER-2013-style directory with `train/` and `test/` folders. Required class directories are lowercase: `angry`, `happy`, `neutral`, `sad`, and `surprise`. Fear and disgust are excluded to match the five-output webcam model.

```bash
python train_kaggle.py --data-dir /path/to/fer2013 --output-dir ./emotion-output
```

Training can be resource-intensive; a hosted notebook with a GPU is recommended. The script generates `evaluation.json`, `confusion_matrix.csv`, and `training_history.csv`. Review those results before replacing the model. Copy both `Emotion_Detection.keras` and `emotion_labels.json` from the same output directory.

## Results and limitations

No verified benchmark report for the exact model currently committed to this repository is included, so no accuracy claim is made here. Run the training pipeline to produce evaluation metrics for its held-out test split.

Predictions depend on lighting, face angle, camera quality, and training data. A softmax score is not a calibrated probability and facial-expression classification cannot reliably determine a person's actual emotional state.

## Troubleshooting

- **Missing modules:** activate `.venv` and run `python -m pip install -r requirements.txt`.
- **Camera unavailable:** close other camera applications and try `--camera 1`.
- **Model or label mismatch:** keep `Emotion_Detection.keras` and `emotion_labels.json` from the same training run.
- **No face detected:** improve lighting and face the camera.
