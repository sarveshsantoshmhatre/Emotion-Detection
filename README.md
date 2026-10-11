# EmotionLens — Facial Expression Recognition

EmotionLens estimates visible facial expressions from a webcam or a still image. It runs locally on **Windows and Ubuntu** using a native desktop window—no Streamlit or web browser required. The Streamlit-hosted demo remains available separately.

> Model predictions are estimates of visible facial expressions, not a reliable measurement of a person's internal emotional state. Displayed scores are not calibrated probabilities.

## Native desktop app (recommended for client handover)

The desktop application supports:
- Live webcam inference with face boxes and expression labels.
- Image-file selection and analysis.
- Score breakdown for the five model classes.
- Local inference: images are processed on the user's computer by this desktop app.

### Windows 10/11

1. Install **Python 3.11 (64-bit)** from [python.org](https://www.python.org/downloads/release/python-3119/). During installation, enable **Add Python to PATH**.
2. Download the project ZIP from GitHub (**Code → Download ZIP**) and extract it, or clone the repository:
   ```powershell
   git clone https://github.com/sarveshsantoshmhatre/Emotion-Detection.git
   cd Emotion-Detection
   ```
3. Double-click `run_windows.bat`.

The launcher creates a local `.venv`, installs requirements, and starts `desktop_app.py`. The first launch needs an internet connection to download dependencies. Later launches can use the installed environment. If you prefer PowerShell commands:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-desktop.txt
python desktop_app.py
```

### Ubuntu Linux

Install Python, venv support, and Tkinter first:

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-tk git
```

Clone the project, then run:

```bash
git clone https://github.com/sarveshsantoshmhatre/Emotion-Detection.git
cd Emotion-Detection
chmod +x run_ubuntu.sh
./run_ubuntu.sh
```

The script creates `.venv`, installs the Python dependencies, and launches the desktop app. The first run requires an internet connection.

### Required files

Keep these files together in the project folder:
- `desktop_app.py` — native desktop UI.
- `Emotion_Detection.keras` — trained model.
- `emotion_labels.json` — model output-index mapping.
- `haarcascade_frontalface_default.xml` — face detector.
- `requirements-desktop.txt` — dependencies for the native app (GUI-enabled OpenCV).\n- `requirements.txt` — dependencies for the hosted Streamlit app (headless OpenCV).
- `run_windows.bat` / `run_ubuntu.sh` — setup and launcher scripts.

### Troubleshooting

- **Camera unavailable:** close other camera apps and check OS camera permissions. A USB webcam may be required.
- **Ubuntu GUI errors:** run from a normal desktop session (not a headless terminal) and make sure `python3-tk` is installed.
- **Python not found on Windows:** install Python 3.11 64-bit and reopen the terminal.
- **Model/label mismatch:** keep `Emotion_Detection.keras` and `emotion_labels.json` from the same training run.
- **Slow inference:** the model runs on CPU unless TensorFlow has a supported accelerator configured.

## Streamlit web app (optional)

The hosted browser version is separate from the native desktop app:
[Open the deployed web app](https://emotion-detection-rntbhgnzygzgq33vlyrwnn.streamlit.app/).

To run the web version locally, install the requirements and run `streamlit run app.py`. For the native client version above, run `desktop_app.py` instead.

## Model training

`train_kaggle.py` trains a five-class model using FER-2013-style `train/` and `test/` directories with lowercase class folders: `angry`, `happy`, `neutral`, `sad`, and `surprise`.

```bash
python train_kaggle.py --data-dir /path/to/fer2013 --output-dir ./emotion-output
```

No verified benchmark report for the exact currently committed model is included here, so this README makes no accuracy claim. Evaluate on a held-out dataset before using the model for consequential decisions.
