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

Ensure these files are in the project directory:

- `Emotion_Detection.keras` — trained model.
- `emotion_labels.json` — maps model output indices to emotion names.
- `haarcascade_frontalface_default.xml` — face detector.

Press `q` or Esc in the webcam window to exit. If the default camera cannot be opened, try `python test.py --camera 1`.

## Training

`train_kaggle.py` trains a five-class model on a FER-2013-style directory with `train/` and `test/` folders. The supported classes are `angry`, `happy`, `neutral`, `sad`, and `surprise`; fear and disgust are intentionally excluded.

Example:

```bash
python train_kaggle.py --data-dir /path/to/fer2013 --output-dir ./emotion-output
```

Training can be resource-intensive; a hosted notebook with a GPU is recommended. Review `evaluation.json` before replacing the model used by the webcam app. Copy both `Emotion_Detection.keras` and `emotion_labels.json` from the training output together.

## Limitations

Predictions depend on lighting, face angle, camera quality, and the training data. The displayed softmax score is not a calibrated probability, and facial-expression classification cannot reliably determine a person's actual emotional state.
