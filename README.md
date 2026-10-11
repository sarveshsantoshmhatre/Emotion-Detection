# Emotion Detection

Facial-expression classification using a Keras model, with a desktop webcam demo and a Streamlit web app.

## Deploy the Streamlit web app

The hosted app supports continuous webcam video through WebRTC and photo upload. Live webcam mode requires browser camera permission and a network that permits WebRTC connections.

1. Confirm these files are committed to the repository root: `app.py`, `Emotion_Detection.keras`, `emotion_labels.json`, `haarcascade_frontalface_default.xml`, and `requirements.txt`.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. Choose **Create app**, select this repository and the `main` branch, and set **Main file path** to `app.py`.
4. Select **Deploy** and review the app logs if the build fails.

Streamlit installs the dependencies in `requirements.txt`. The model runs on the hosting instance, typically on CPU. Live webcam mode uses `streamlit-webrtc`; if the video fails to connect, check browser camera permission and network/firewall restrictions. Photo upload remains available as a fallback.

## Run the Streamlit app locally

Use Python 3.10 or newer in a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit.

## Run the desktop webcam demo

The hosted requirements use `opencv-python-headless`, which does not include OpenCV's desktop GUI. To use `test.py` locally, replace it with the regular OpenCV package inside your virtual environment:

```bash
python -m pip uninstall -y opencv-python-headless
python -m pip install opencv-python
python test.py
```

The desktop app checks that the model and face detector exist, verifies that the camera opens, and exits if a camera frame cannot be read. If the default camera cannot be opened, try `python test.py --camera 1`.

## Required files

- `Emotion_Detection.keras` — trained model.
- `emotion_labels.json` — output-index to emotion-name mapping.
- `haarcascade_frontalface_default.xml` — face detector.
- `app.py` — Streamlit web app.
- `test.py` — desktop webcam inference.
- `train_kaggle.py` — training pipeline.

## Preprocessing compatibility

The current `train_kaggle.py` pipeline trains on **48 × 48 grayscale** faces and includes a `Rescaling(1/255)` layer inside the model. Both inference paths read the model's declared input dimensions and channel count and avoid applying the same rescaling twice. If you replace the model, replace `emotion_labels.json` from the same training run as well.

The removed legacy `train.py` used a different MobileNet pipeline (224 × 224 RGB). Do not mix a model and label map from different training runs. Correct normalization must match the model's training pipeline.

## Training

`train_kaggle.py` trains a five-class model on a FER-2013-style directory with `train/` and `test/` folders. Required class directories are lowercase: `angry`, `happy`, `neutral`, `sad`, and `surprise`. Fear and disgust are excluded to match the five-output model.

```bash
python train_kaggle.py --data-dir /path/to/fer2013 --output-dir ./emotion-output
```

Training can be resource-intensive; a hosted notebook with a GPU is recommended. The script generates `evaluation.json`, `confusion_matrix.csv`, and `training_history.csv`. Review those results before replacing the model. Copy both `Emotion_Detection.keras` and `emotion_labels.json` from the same output directory.

## Results and limitations

No verified benchmark report for the exact model currently committed to this repository is included, so no accuracy claim is made here. Run the training pipeline to produce evaluation metrics for its held-out test split.

Predictions depend on lighting, face angle, camera quality, and training data. Displayed model scores are not calibrated probabilities, and facial-expression classification cannot reliably determine a person's actual emotional state.

## Troubleshooting

- **Missing modules:** activate `.venv` and run `python -m pip install -r requirements.txt`.
- **Camera unavailable:** close other camera applications and try `--camera 1` in the desktop demo.
- **Model or label mismatch:** keep `Emotion_Detection.keras` and `emotion_labels.json` from the same training run.
- **No face detected:** improve lighting and face the camera.
