"""Native desktop EmotionLens app for Windows and Ubuntu (no Streamlit required)."""

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Emotion_Detection.keras"
LABELS_PATH = BASE_DIR / "emotion_labels.json"
CASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"


def contains_rescaling(layer):
    if layer.__class__.__name__ == "Rescaling":
        return True
    return any(contains_rescaling(child) for child in getattr(layer, "layers", []))


def load_labels():
    mapping = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return [name.title() for name, idx in sorted(mapping.items(), key=lambda item: item[1])]


class EmotionLensApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EmotionLens — Facial Expression Recognition")
        self.root.geometry("1050x760")
        self.root.minsize(760, 600)
        self.root.configure(bg="#0b1020")

        self.model = None
        self.face_detector = None
        self.labels = load_labels()
        self.rescaling = False
        self.camera = None
        self.camera_running = False
        self.after_id = None
        self.last_frame = None
        self.last_scores = None
        self.tk_frame = None
        self.photo_path = None

        self._configure_styles()
        self._build_ui()
        self.status_var.set("Loading model…")
        threading.Thread(target=self._load_model_worker, daemon=True).start()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#0b1020")
        style.configure("Panel.TFrame", background="#111a2d")
        style.configure("TLabel", background="#0b1020", foreground="#e2e8f0", font=("Segoe UI", 10))
        style.configure("Hero.TLabel", background="#111a2d", foreground="#f8fafc", font=("Segoe UI", 24, "bold"))
        style.configure("Sub.TLabel", background="#111a2d", foreground="#93c5fd", font=("Segoe UI", 10))
        style.configure("Status.TLabel", background="#0b1020", foreground="#94a3b8", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=(13, 9))
        style.map("TButton", background=[("active", "#2dd4bf")])

    def _build_ui(self):
        hero = ttk.Frame(self.root, style="Panel.TFrame", padding=(24, 18))
        hero.pack(fill="x", padx=20, pady=(20, 12))
        ttk.Label(hero, text="EmotionLens", style="Hero.TLabel").pack(anchor="w")
        ttk.Label(
            hero,
            text="FACIAL EXPRESSION RECOGNITION  ·  PRIVATE LOCAL INFERENCE",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(5, 0))

        controls = ttk.Frame(self.root, padding=(20, 0))
        controls.pack(fill="x")
        self.live_btn = ttk.Button(controls, text="Start live webcam", command=self.toggle_camera)
        self.live_btn.pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Open image…", command=self.open_image).pack(side="left", padx=8)
        ttk.Button(controls, text="Stop / clear", command=self.stop_and_clear).pack(side="left", padx=8)

        body = ttk.Frame(self.root, padding=(20, 14))
        body.pack(fill="both", expand=True)
        self.preview = tk.Label(
            body,
            text="Loading the model…",
            bg="#111a2d",
            fg="#94a3b8",
            font=("Segoe UI", 16),
            bd=0,
            compound="center",
        )
        self.preview.pack(side="left", fill="both", expand=True, padx=(0, 14))

        side = ttk.Frame(body, style="Panel.TFrame", padding=20, width=260)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        ttk.Label(side, text="ANALYSIS", style="Sub.TLabel").pack(anchor="w")
        self.prediction_var = tk.StringVar(value="—")
        ttk.Label(
            side, textvariable=self.prediction_var,
            background="#111a2d", foreground="#f8fafc",
            font=("Segoe UI", 20, "bold"), wraplength=220,
        ).pack(anchor="w", pady=(12, 6))
        self.score_var = tk.StringVar(value="Model score: —")
        ttk.Label(side, textvariable=self.score_var, background="#111a2d", foreground="#5eead4").pack(anchor="w")
        ttk.Separator(side).pack(fill="x", pady=16)
        ttk.Label(side, text="CLASS SCORES", style="Sub.TLabel").pack(anchor="w", pady=(0, 8))
        self.score_rows = {}
        self.score_bars = {}
        for label in self.labels:
            row = ttk.Frame(side, style="Panel.TFrame")
            row.pack(fill="x", pady=4)
            ttk.Label(row, text=label, background="#111a2d", foreground="#cbd5e1").pack(anchor="w")
            bar = ttk.Progressbar(row, maximum=100, length=205, mode="determinate")
            bar.pack(fill="x", pady=(3, 0))
            value = ttk.Label(row, text="—", background="#111a2d", foreground="#94a3b8")
            value.pack(anchor="e")
            self.score_bars[label] = bar
            self.score_rows[label] = value

        footer = ttk.Frame(self.root, padding=(20, 0, 20, 16))
        footer.pack(fill="x")
        self.status_var = tk.StringVar(value="Starting…")
        ttk.Label(footer, textvariable=self.status_var, style="Status.TLabel").pack(anchor="w")
        ttk.Label(
            footer,
            text="Predictions estimate visible facial expressions and may be wrong; scores are not calibrated probabilities.",
            style="Status.TLabel",
            wraplength=1000,
        ).pack(anchor="w", pady=(4, 0))

    def _load_model_worker(self):
        try:
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
            if not CASCADE_PATH.is_file():
                raise FileNotFoundError(f"Face detector not found: {CASCADE_PATH}")
            import tensorflow as tf
            model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
            detector = cv2.CascadeClassifier(str(CASCADE_PATH))
            if detector.empty():
                raise RuntimeError("OpenCV could not load the Haar cascade.")
            labels = load_labels()
            if int(model.output_shape[-1]) != len(labels):
                raise ValueError("Model output count does not match emotion_labels.json.")
            self.model = model
            self.face_detector = detector
            self.rescaling = contains_rescaling(model)
            self.root.after(0, self._model_loaded)
        except Exception as exc:
            self.root.after(0, lambda message=str(exc): self._model_failed(message))

    def _model_loaded(self):
        self.status_var.set("Model ready. Start the webcam or open an image.")
        self.preview.configure(text="Choose “Start live webcam” or “Open image…”")

    def _model_failed(self, message):
        self.status_var.set("Could not load model.")
        messagebox.showerror(
            "Model loading failed",
            f"{message}\n\nCheck that the model, labels, and cascade files are beside desktop_app.py.",
        )
        self.preview.configure(text="Model failed to load. See the error dialog.")

    def _classify(self, gray_face):
        shape = self.model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        height, width, channels = int(shape[1]), int(shape[2]), int(shape[3])
        face = cv2.resize(gray_face, (width, height), interpolation=cv2.INTER_AREA)
        if channels == 1:
            face = face[..., None]
        elif channels == 3:
            face = np.repeat(face[..., None], 3, axis=-1)
        else:
            raise ValueError(f"Unsupported model channels: {channels}")
        batch = face[None, ...].astype(np.float32)
        if not self.rescaling:
            batch /= 255.0
        scores = np.asarray(self.model.predict(batch, verbose=0))[0]
        if scores.ndim != 1 or len(scores) != len(self.labels) or not np.all(np.isfinite(scores)):
            raise ValueError("Model produced an invalid score vector.")
        return scores

    def _annotate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
        all_scores = None
        for x, y, w, h in faces:
            try:
                scores = self._classify(gray[y:y+h, x:x+w])
                index = int(np.argmax(scores))
                label = self.labels[index]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (45, 212, 191), 2)
                cv2.rectangle(frame, (x, max(0, y-30)), (x+w, y), (15, 23, 42), -1)
                cv2.putText(frame, f"{label} {scores[index]:.0%}", (x+5, max(20, y-9)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
                if all_scores is None:
                    all_scores = scores
            except Exception:
                continue
        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (45, 212, 191), 2, cv2.LINE_AA)
        return frame, all_scores

    def _show_frame(self, frame, scores=None):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        width = max(320, self.preview.winfo_width())
        height = max(280, self.preview.winfo_height())
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.tk_frame = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.tk_frame, text="")
        if scores is None:
            self.prediction_var.set("No face detected")
            self.score_var.set("Model score: —")
            for label in self.labels:
                self.score_bars[label]["value"] = 0
                self.score_rows[label].configure(text="—")
        else:
            idx = int(np.argmax(scores))
            self.prediction_var.set(self.labels[idx])
            self.score_var.set(f"Model score: {float(scores[idx]):.1%}")
            for label, score in zip(self.labels, scores):
                self.score_bars[label]["value"] = max(0.0, min(100.0, float(score) * 100))
                self.score_rows[label].configure(text=f"{float(score):.1%}")

    def toggle_camera(self):
        if self.model is None:
            messagebox.showinfo("Please wait", "The model is still loading.")
            return
        if self.camera_running:
            self._stop_camera()
            self.status_var.set("Webcam stopped.")
            return
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None
            messagebox.showerror(
                "Camera unavailable",
                "Could not open the default camera. Close other camera apps and check permissions.",
            )
            return
        self.camera_running = True
        self.live_btn.configure(text="Stop live webcam")
        self.status_var.set("Live webcam active. Press Stop live webcam to end.")
        self._camera_tick()

    def _camera_tick(self):
        if not self.camera_running or self.camera is None:
            return
        ok, frame = self.camera.read()
        if ok and frame is not None:
            try:
                annotated, scores = self._annotate(frame)
                self._show_frame(annotated, scores)
            except Exception as exc:
                self.status_var.set(f"Inference error: {exc}")
        else:
            self.status_var.set("Camera frame unavailable.")
        self.after_id = self.root.after(15, self._camera_tick)

    def _stop_camera(self):
        self.camera_running = False
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.live_btn.configure(text="Start live webcam")

    def open_image(self):
        if self.model is None:
            messagebox.showinfo("Please wait", "The model is still loading.")
            return
        path = filedialog.askopenfilename(
            title="Choose a photo",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            image = cv2.imread(path)
            if image is None:
                raise ValueError("OpenCV could not read this image.")
            annotated, scores = self._annotate(image)
            self._show_frame(annotated, scores)
            self.photo_path = path
            self.status_var.set(f"Analyzed image: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Image analysis failed", str(exc))

    def stop_and_clear(self):
        self._stop_camera()
        self.preview.configure(image="", text="Choose “Start live webcam” or “Open image…”")
        self.prediction_var.set("—")
        self.score_var.set("Model score: —")
        self.status_var.set("Ready.")
        for label in self.labels:
            self.score_bars[label]["value"] = 0
            self.score_rows[label].configure(text="—")

    def close(self):
        self._stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        EmotionLensApp(root)
    except Exception as exc:
        messagebox.showerror("Startup failed", str(exc))
        root.destroy()
        raise


if __name__ == "__main__":
    main()
