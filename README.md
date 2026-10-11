# Emotion Detection

Real-time facial emotion detection using a webcam and a Keras model.

## Run the webcam demo

1. Install Python dependencies in your virtual environment (including OpenCV, TensorFlow/Keras, and NumPy).
2. Ensure `Emotion_Detection.keras`, `emotion_labels.json`, and `haarcascade_frontalface_default.xml` are in the project directory.
3. Run:

```bash
python test.py
```

## Training

`train_kaggle.py` contains the training pipeline for the FER-2013 dataset. Run it in an environment where the dataset is available and configured.

## Files

- `test.py` — webcam inference.
- `Emotion_Detection.keras` — trained model used by the demo.
- `emotion_labels.json` — model output label mapping.
- `haarcascade_frontalface_default.xml` — face detector cascade.
- `train_kaggle.py` — training and evaluation pipeline.
- `inspect_model.py` — optional model inspection/debugging utility.
