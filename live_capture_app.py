"""Élő polip-elemző GUI: kamera, videófájl vagy képmappa, klasszifikáció és magyarázó térkép.

Alap modell: DINOv2 (ONNX, onnxruntime, torch nélkül).
A régi PyTorch / Keras modellek (DeepLabV3+, U-Net, ViT-B/16, ResNet50V2) csak akkor érhetők el,
ha a requirements_legacy.txt csomagjai is telepítve vannak.
"""
from __future__ import annotations

import csv
import ctypes
import json
import os
import random
import sqlite3
import threading
import time
import traceback
from ctypes import wintypes
from dataclasses import dataclass

import cv2
import customtkinter as ctk
import numpy as np
import tkinter as tk
from PIL import Image

from polyp_infer import PolypClassifier, letterbox_to_square

try:
    import torch
    import segmentation_models_pytorch as smp
    import utils.ext_transforms as et
    from network.modeling import deeplabv3plus_mobilenet
    from utils.explainability import (
        blend_heatmap,
        find_last_keras_conv_layer,
        grad_cam_keras,
        grad_cam_torch,
        map_cam_to_full_frame,
        build_keras_grad_model,
        vit_attention_map,
    )
    from utils.vit_classifier import (
        load_vit_checkpoint,
        preprocess_rgb_uint8,
        predict_vit,
        THRESHOLD as VIT_THRESHOLD,
    )
    LEGACY_IMPORT_ERROR = None
except ImportError as _legacy_exc:
    torch = None
    LEGACY_IMPORT_ERROR = _legacy_exc


def _setup_tensorflow():
    """(tf, ResNetPreprocessLayer, hiba). A Keras ResNet nélkül is elindul az app."""
    try:
        import tensorflow as tf
    except Exception as exc:
        return None, None, exc

    # A régi Keras modell RandomContrast rétege value_range argumentummal mentődött.
    try:
        original_init = tf.keras.layers.RandomContrast.__init__

        def patched_init(self, factor, seed=None, **kwargs):
            kwargs.pop("value_range", None)
            original_init(self, factor=factor, seed=seed, **kwargs)

        tf.keras.layers.RandomContrast.__init__ = patched_init
    except Exception as exc:
        print(f"RandomContrast javítás sikertelen: {exc}")

    @tf.keras.utils.register_keras_serializable(package="PolpyCLI", name="ResNetPreprocessLayer")
    class ResNetPreprocessLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            return tf.keras.applications.resnet_v2.preprocess_input(inputs)

    return tf, ResNetPreprocessLayer, None


tf = None
ResNetPreprocessLayer = None
_TF_IMPORT_ERROR = None
if torch is not None:
    tf, ResNetPreprocessLayer, _TF_IMPORT_ERROR = _setup_tensorflow()


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERFORMANCE_LOG_FILE = os.path.join(_BASE_DIR, "performance_log.csv")
OUTPUT_SAVE_DIR = os.path.join(_BASE_DIR, "saved_results")
PREDICTIONS_DB_FILE = os.path.join(OUTPUT_SAVE_DIR, "predictions.db")

MODEL_DIR = os.path.join(_BASE_DIR, "models")
DINOV2_ONNX_NAME = "dinov2_polyp.onnx"

DEEPLAB_CHECKPOINT = os.path.join(_BASE_DIR, "checkpoints_0409", "best_model_0409_epoch_31.pth")
UNET_CHECKPOINT = os.path.join(_BASE_DIR, "checkpoints_unet", "best_model_epoch_22.pth")

# classificator_models/: *.keras a ROI klasszifikátor, a *_metadata.json (input_shape) opcionális.
CLASSIFIER_MODEL_DIR = os.path.join(_BASE_DIR, "classificator_models")
CLASSIFIER_KERAS_PREFERRED = "cnn_s2_lr2e4_tb_os.keras"
CLASSIFIER_KERAS_LEGACY = "resnet50v2_polyp_20260217_131050.keras"
CLASSIFIER_INPUT_SIZE = 512
CLASSIFIER_CLASS_NAMES = ["Non-neoplastic (JNET 1)", "Neoplastic (JNET 2a/2b/3)"]
VIT_MODEL_DIR = os.path.join(CLASSIFIER_MODEL_DIR, "vit_gui_share")
VIT_CHECKPOINT_NAME = "vit_final_224.pth"

NUM_CLASSES = 2
OUTPUT_STRIDE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch is not None else None
INPUT_SIZE = 513


@dataclass(frozen=True)
class ClassifierSpec:
    label: str
    full_frame: bool
    status: str


# A menü sorrendje is ez. full_frame: nincs szegmentálás, a modell a teljes képet kapja.
CLASSIFIERS = {
    "dinov2": ClassifierSpec("DINOv2", True, "Space: teljes képes klasszifikáció, szegmentálás nélkül"),
    "vit": ClassifierSpec("ViT-B/16", True, "Space: teljes képes klasszifikáció, szegmentálás nélkül"),
    "keras": ClassifierSpec("ResNet50V2", False, "Space: szegmentálás + ROI klasszifikáció"),
}


def _resolve_dinov2_path():
    """Env DINOV2_MODEL_PATH, majd models/, majd az app mappája. Ha egyik sincs, a várt útvonal."""
    env = (os.environ.get("DINOV2_MODEL_PATH") or "").strip()
    preferred = os.path.join(MODEL_DIR, DINOV2_ONNX_NAME)
    for candidate in (env, preferred, os.path.join(_BASE_DIR, DINOV2_ONNX_NAME)):
        if candidate and os.path.isfile(candidate):
            return candidate
    return preferred


def _resolve_vit_path():
    env = (os.environ.get("VIT_MODEL_PATH") or "").strip()
    if env and os.path.isfile(env):
        return env
    return os.path.join(VIT_MODEL_DIR, VIT_CHECKPOINT_NAME)


def _resolve_classifier_keras_path():
    """Env CLASSIFIER_MODEL_PATH, preferált fájl, régi név, végül a legfrissebb *.keras."""
    env = (os.environ.get("CLASSIFIER_MODEL_PATH") or "").strip()
    if env and os.path.isfile(env):
        return env
    preferred = os.path.join(CLASSIFIER_MODEL_DIR, CLASSIFIER_KERAS_PREFERRED)
    if os.path.isfile(preferred):
        return preferred
    legacy = os.path.join(CLASSIFIER_MODEL_DIR, CLASSIFIER_KERAS_LEGACY)
    if os.path.isfile(legacy):
        return legacy
    if os.path.isdir(CLASSIFIER_MODEL_DIR):
        candidates = [
            os.path.join(CLASSIFIER_MODEL_DIR, f)
            for f in os.listdir(CLASSIFIER_MODEL_DIR)
            if f.lower().endswith(".keras")
        ]
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]
    return preferred


def _read_classifier_input_size(keras_path, default=CLASSIFIER_INPUT_SIZE):
    """A <stem>_metadata.json input_shape értékéből a négyzetes oldalhossz, ha van."""
    stem, _ = os.path.splitext(os.path.basename(keras_path))
    meta_path = os.path.join(os.path.dirname(keras_path), f"{stem}_metadata.json")
    if not os.path.isfile(meta_path):
        return default
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        shape = data.get("input_shape")
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            h, w = int(shape[0]), int(shape[1])
            if h > 0 and w > 0 and h == w:
                return h
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return default



def _init_predictions_db():
    """SQLite-ADTB trigger-eseményekhez: maszk BLOB + metaadatok; WAL a többszálas hozzáféréshez."""
    os.makedirs(OUTPUT_SAVE_DIR, exist_ok=True)
    conn = sqlite3.connect(PREDICTIONS_DB_FILE, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp      TEXT    NOT NULL,
            class_name     TEXT    NOT NULL,
            confidence     REAL    NOT NULL,
            seg_confidence REAL,
            seg_model      TEXT,
            cls_model      TEXT,
            input_path     TEXT,
            mask_path      TEXT,
            mask_blob      BLOB
        );
        CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(timestamp);
        """
    )
    conn.commit()
    return conn


def _imread_unicode(path):
    """Kép betöltése útvonalról. Windows-on a cv2.imread() gyakran elbukik ékezetes / Unicode útvonalon (OneDrive stb.)."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except OSError:
        return None


def list_available_cameras():
    """Lists available cameras and video files."""
    camera_list = []
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        devices = graph.get_input_devices()
        for i, name in enumerate(devices):
             camera_list.append(f"Camera {i}: {name}")
    except ImportError:
        print("pygrabber not found, falling back to basic indexing")
        camera_list = [f"Camera {i}" for i in range(5)]
    except Exception as e:
        print(f"Error listing cameras: {e}")
        camera_list = [f"Camera {i}" for i in range(5)]
        
    # Videófájlok az app mappájából, nem a cwd-ből.
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    try:
        base_files = os.listdir(_BASE_DIR)
    except OSError:
        base_files = []
    for f in base_files:
        if f.lower().endswith(video_extensions):
            camera_list.append(f"Video: {f}")
        
    return camera_list


class VideoCaptureThread:
    def __init__(self, src=0, simulation_mode=False, image_folder=None):
        self.simulation_mode = simulation_mode
        self.image_folder = image_folder
        self.image_files = []
        self.video_file_mode = False
        
        if self.simulation_mode and self.image_folder:
             allowed_ids = [
                 'PI009', 'PI015', 'PI016', 'PI037', 'PI048', 'PI050', 'PI055', 'PI057', 'PI068',
                 'PI072', 'PI073', 'PI078', 'PI080', 'PI085', 'PI086', 'PI091', 'PI112', 'PI113',
                 'PI114', 'PI121', 'PI123', 'PI130', 'PI135', 'PI136', 'PI137', 'PI142', 'PI149',
                 'PI152', 'PI166', 'PI171', 'PI175', 'PI178', 'PI181', 'PI184', 'PI187', 'PI189',
                 'PI191', 'PI197', 'PI198', 'PI204', 'pi205', 'PI207', 'PI210', 'PI211', 'PI219',
                 'PI224', 'PI230', 'PI238', 'PI243', 'PI247', 'PI249', 'PI252', 'PI261', 'PI266',
                 'PI270', 'PI272', 'PI277', 'PI280', 'PI290', 'PI298', 'PI310', 'PI335', 'PI338',
                 'PI348', 'PI349', 'PI367', 'PI368', 'PI370', 'PI372', 'PI375', 'PI376', 'PI389',
                 'PI390', 'PI399', 'PI400', 'PI405', 'PI408', 'PI415', 'PI418', 'PI419', 'PI432',
                 'PI433', 'PI445', 'PI454', 'PI456', 'PI462', 'PI468', 'PI472', 'PI473', 'PI478',
                 'PI480'
             ]
             allowed_ids_lower = [pid.lower() for pid in allowed_ids]

             all_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
             self.image_files = []
             
             for f in all_files:
                 f_lower = f.lower()
                 if any(pid in f_lower for pid in allowed_ids_lower):
                     self.image_files.append(os.path.join(image_folder, f))
                     
             print(f"Simulation mode: Found {len(self.image_files)} images matching filter (Total allowed IDs: {len(allowed_ids)}).")

        if not self.simulation_mode:
            if isinstance(src, str):
                self.video_file_mode = True
                print(f"Opening video file: {src}")
                self.cap = cv2.VideoCapture(src)
            else:
                self.cap = cv2.VideoCapture(src)
                
            self.ret, self.frame = self.cap.read()
        else:
            self.cap = None
            self.ret = False
            self.frame = None
            if self.image_files:
                self.load_random_image()

        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
    def load_random_image(self):
        if not self.image_files:
            return

        img_path = random.choice(self.image_files)
        try:
            img = _imread_unicode(img_path)
            if img is not None:
                 self.frame = img
                 self.ret = True
            else:
                 print(f"Failed to load image: {img_path}")
        except Exception as e:
            print(f"Error loading random image: {e}")

    def update(self):
        fps = 30.0
        if self.cap:
             fps = self.cap.get(cv2.CAP_PROP_FPS)
             if fps <= 0: fps = 30.0
        frame_time = 1.0 / fps
        
        last_sim_update = time.time()

        while self.running:
            start_loop = time.time()

            if not self.simulation_mode:
                if self.cap and self.cap.isOpened():
                    if self.video_file_mode:
                        # A lock a set_position (tekerés) ütközése miatt kell.
                        with self.lock:
                            ret, frame = self.cap.read()

                            # Videó végén újraindul.
                            if not ret:
                                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                ret, frame = self.cap.read()
                            
                            if ret:
                                self.ret = ret
                                self.frame = frame
                        
                        elapsed = time.time() - start_loop
                        wait = frame_time - elapsed
                        if wait > 0:
                            time.sleep(wait)
                    else:
                        with self.lock:
                            ret, frame = self.cap.read()
                            if ret:
                                self.ret = ret
                                self.frame = frame

                        time.sleep(0.005)
                else:
                    time.sleep(0.01)
            else:
                if time.time() - last_sim_update > 2.0:
                    self.load_random_image()
                    last_sim_update = time.time()
                time.sleep(0.1)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            try:
                return self.ret, self.frame.copy()
            except MemoryError:
                return False, None

    def get_fps(self):
        with self.lock:
            if self.cap:
                 return self.cap.get(cv2.CAP_PROP_FPS)
        return 30.0

    def set_position(self, percent):
        with self.lock:
            if self.cap and self.video_file_mode:
                 total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                 frame_idx = int(total_frames * percent / 100.0)
                 self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def get_position_percent(self):
        """Lejátszási pozíció 0-100% (csak videófájl módban)."""
        with self.lock:
            if not self.cap or not self.video_file_mode or not self.cap.isOpened():
                return None
            total = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            if total is None or total <= 0:
                return None
            return float(max(0.0, min(100.0, 100.0 * pos / total)))

    def stop(self):
        self.running = False
        self.thread.join(timeout=5.0)
        if self.cap:
            self.cap.release()

@dataclass
class ExplainContext:
    segmented_polyp: np.ndarray
    input_tensor: torch.Tensor | None
    roi_batch: np.ndarray | None
    bbox: tuple[int, int, int, int] | None
    cls_label: str
    malignant_prob: float
    model_type: str
    pred_mask: np.ndarray
    vit_input_tensor: torch.Tensor | None = None
    dinov2_overlay: np.ndarray | None = None


class MedicalAnalyzer:
    def __init__(self, device):
        self.device = device
        self.current_model_type = "deeplab"
        self.models_loaded = False
        self.classifier = None
        self.classifier_keras_path = None
        self.classifier_input_size = CLASSIFIER_INPUT_SIZE
        self.deeplab_model = None
        self.unet_model = None
        self.model = None
        self.explain_mode = "classification"
        self._keras_cam_layer = None
        self._keras_grad_model = None
        self._last_explain_ctx = None
        self.vit_classifier = None
        self.vit_path = None
        self.dinov2 = None
        self.dinov2_path = None
        self.load_notes = []
        self.classifier_backend = "dinov2"

        if not os.path.exists(PERFORMANCE_LOG_FILE):
            with open(PERFORMANCE_LOG_FILE, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp",
                    "Seg_Model", "Cls_Model",
                    "Total_Time_ms",
                    "Preprocess_ms",
                    "Segmentation_Inference_ms",
                    "Postprocess_ms",
                    "Classification_ms"
                ])

        self.transform = None
        if torch is not None:
            self.transform = et.ExtCompose([
                et.ExtResize(size=INPUT_SIZE),
                et.ExtCenterCrop(size=INPUT_SIZE),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def load_models(self, status_callback=None):
        """Modellek betöltése és warm-up (külön szálon futtatható)."""
        def say(text):
            if status_callback:
                status_callback(text)

        self.models_loaded = False
        self.load_notes = []
        try:
            self._load_dinov2(say)
            if torch is not None:
                self._load_legacy(say)
            else:
                print(f"Régi modellek kihagyva (torch / tensorflow nincs telepítve): {LEGACY_IMPORT_ERROR}")

            ready = self.available_backends()
            self.classifier_backend = ready[0] if ready else "dinov2"
            print(f"Classifier backend: {self.classifier_backend}")
            self.models_loaded = True

            if ready:
                say(f"Rendszer kész. (Klasszifikátor: {CLASSIFIERS[self.classifier_backend].label})")
            else:
                reason = self.load_notes[0] if self.load_notes else f"nincs modell: {self.dinov2_path}"
                say(f"Rendszer kész, de nincs klasszifikátor. {reason}")
        except Exception as e:
            traceback.print_exc()
            self.models_loaded = False
            msg = str(e).replace("\n", " ")[:150]
            say(f"HIBA: modell betöltés sikertelen, {msg}")

    def _load_dinov2(self, say):
        self.dinov2 = None
        self.dinov2_path = _resolve_dinov2_path()
        say("DINOv2 betöltése...")
        if not os.path.isfile(self.dinov2_path):
            self.load_notes.append(f"DINOv2 modell nem található: {self.dinov2_path}")
            print(self.load_notes[-1])
            return
        try:
            clf = PolypClassifier(self.dinov2_path)
            clf.predict(Image.new("RGB", (64, 64)))
            self.dinov2 = clf
            print(f"DINOv2 classifier: {self.dinov2_path} (input {clf.size}x{clf.size}, küszöb {clf.threshold})")
        except Exception as exc:
            self.load_notes.append(f"DINOv2 betöltés sikertelen: {exc}")
            print(self.load_notes[-1])

    def _load_legacy(self, say):
        """DeepLabV3+ / U-Net szegmentálás, ViT-B/16 (PyTorch) és ResNet50V2 (Keras)."""
        say("DeepLabV3+ betöltése...")
        self.deeplab_model = self._load_deeplab()
        say("U-Net betöltése...")
        self.unet_model = self._load_unet()

        seg_ready = self.seg_models_available()
        if seg_ready:
            self.current_model_type = "deeplab" if "deeplab" in seg_ready else "unet"
            self.model = self.deeplab_model if self.current_model_type == "deeplab" else self.unet_model
            say("Modellek bemelegítése (warm-up)...")
            dummy_input_pt = torch.zeros((1, 3, INPUT_SIZE, INPUT_SIZE)).to(self.device)
            with torch.no_grad():
                for net in (self.deeplab_model, self.unet_model):
                    if net is not None:
                        net(dummy_input_pt)

        self.classifier = None
        self.vit_classifier = None
        self.vit_path = _resolve_vit_path()
        self.classifier_keras_path = _resolve_classifier_keras_path()
        self.classifier_input_size = _read_classifier_input_size(
            self.classifier_keras_path, CLASSIFIER_INPUT_SIZE
        )

        say("ViT klasszifikátor betöltése...")
        if os.path.isfile(self.vit_path):
            try:
                self.vit_classifier = load_vit_checkpoint(self.vit_path, self.device)
                with torch.inference_mode():
                    _ = self.vit_classifier(torch.zeros((1, 3, 224, 224), device=self.device))
                print(f"ViT classifier: {self.vit_path} (input 224x224, full-frame)")
            except Exception as vit_err:
                print(f"ViT betöltés sikertelen: {vit_err}")
                self.vit_classifier = None
        else:
            print(f"ViT checkpoint nem található: {self.vit_path}")

        say("Keras klasszifikátor betöltése...")
        if tf is None:
            print(f"TensorFlow import hiba (Keras ROI ki): {_TF_IMPORT_ERROR}")
            return
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        try:
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(2)
        except Exception:
            pass
        if not os.path.isfile(self.classifier_keras_path):
            print(f"Keras klasszifikátor nem található: {self.classifier_keras_path}")
            return
        try:
            print(
                f"Classifier (Keras ROI): {self.classifier_keras_path} "
                f"(input {self.classifier_input_size}x{self.classifier_input_size})"
            )
            self.classifier = tf.keras.models.load_model(
                self.classifier_keras_path,
                custom_objects={
                    "preprocess_input": tf.keras.applications.resnet_v2.preprocess_input,
                    "ResNetPreprocessLayer": ResNetPreprocessLayer
                },
                compile=False
            )
            dummy_input_tf = np.zeros(
                (1, self.classifier_input_size, self.classifier_input_size, 3),
                dtype=np.float32,
            )
            _ = self.classifier.predict(dummy_input_tf, verbose=0)
            self._keras_cam_layer = find_last_keras_conv_layer(self.classifier)
            self._keras_grad_model = None
            if self._keras_cam_layer is not None:
                print(f"Grad-CAM Keras layer: {self._keras_cam_layer.name}")
                try:
                    self._keras_grad_model = build_keras_grad_model(
                        self.classifier, self._keras_cam_layer
                    )
                    print("Grad-CAM Keras model ready.")
                except Exception as cam_err:
                    print(f"Grad-CAM modell építés sikertelen (heatmap ki): {cam_err}")
                    self._keras_cam_layer = None
                    self._keras_grad_model = None
        except Exception as keras_err:
            print(f"Keras klasszifikátor betöltés sikertelen: {keras_err}")
            self.classifier = None

    def _load_checkpoint(self, model, checkpoint_path):
        """Checkpoint betöltése kulcsformátumtól függetlenül. Hiba esetén None, nem tanítatlan modell."""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint nem található: {checkpoint_path}")
            return None
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            elif 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
            elif 'state_dict' in ckpt:
                model.load_state_dict(ckpt['state_dict'])
            else:
                model.load_state_dict(ckpt)
            print(f"Loaded: {checkpoint_path}")
        except Exception as e:
            print(f"Error loading {checkpoint_path}: {e}")
            return None
        return model

    def _load_deeplab(self):
        model = deeplabv3plus_mobilenet(num_classes=NUM_CLASSES, output_stride=OUTPUT_STRIDE)
        model = self._load_checkpoint(model, DEEPLAB_CHECKPOINT)
        if model is None:
            return None
        model.to(self.device)
        model.eval()
        return model

    def _load_unet(self):
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        model = self._load_checkpoint(model, UNET_CHECKPOINT)
        if model is None:
            return None
        model.to(self.device)
        model.eval()
        return model

    def seg_models_available(self):
        return [
            key for key, net in (("deeplab", self.deeplab_model), ("unet", self.unet_model))
            if net is not None
        ]

    def backend_ready(self, key):
        if key == "dinov2":
            return self.dinov2 is not None
        if key == "vit":
            return self.vit_classifier is not None
        if key == "keras":
            return self.classifier is not None and bool(self.seg_models_available())
        return False

    def available_backends(self):
        return [key for key in CLASSIFIERS if self.backend_ready(key)]

    @property
    def full_frame(self):
        spec = CLASSIFIERS.get(self.classifier_backend)
        return bool(spec and spec.full_frame and self.backend_ready(self.classifier_backend))

    def switch_model(self, model_type):
        """Váltás a betöltött szegmentáló modellek között ('deeplab' / 'unet')."""
        net = {"deeplab": self.deeplab_model, "unet": self.unet_model}.get(model_type)
        if net is None:
            print(f"Szegmentáló nem elérhető: {model_type}")
            return self.current_model_type
        self.model = net
        self.current_model_type = model_type
        print(f"Switched to {model_type}")
        return self.current_model_type

    def switch_classifier_backend(self, backend: str) -> str:
        if backend not in CLASSIFIERS or not self.backend_ready(backend):
            print(f"Klasszifikátor nem elérhető: {backend}")
            return self.classifier_backend
        self.classifier_backend = backend
        print(f"Classifier backend: {CLASSIFIERS[backend].label}")
        return self.classifier_backend

    def active_cls_model_name(self) -> str:
        if self.backend_ready(self.classifier_backend):
            return CLASSIFIERS[self.classifier_backend].label
        return "none"

    def active_seg_model_name(self) -> str:
        return "none" if self.full_frame else self.current_model_type

    def preprocess(self, image):
        """PIL kép -> (batch tensor, tensor). Az ext_transforms (kép, maszk) párt vár, ezért üres maszk kell."""
        w, h = image.size
        dummy_mask = Image.new('L', (w, h), 0)
        img_trans, _ = self.transform(image, dummy_mask)
        return img_trans.unsqueeze(0).to(self.device), img_trans

    def _extract_roi(self, img_rgb, pred_mask):
        """Bounding box a polip maszkból, kivágás, átméretezés a klasszifikátor bemenetére."""
        rows = np.any(pred_mask, axis=1)
        cols = np.any(pred_mask, axis=0)
        if not (np.any(rows) and np.any(cols)):
            return None
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        roi = img_rgb[rmin:rmax + 1, cmin:cmax + 1]
        roi_resized = cv2.resize(
            roi,
            (self.classifier_input_size, self.classifier_input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        return roi_resized.astype(np.float32)

    def analyze(self, frame_bgr):
        results, _ctx = self._analyze_core(frame_bgr)
        return results

    def analyze_with_context(self, frame_bgr):
        return self._analyze_core(frame_bgr)

    def _log_performance(self, total_ms, pre_ms, seg_ms, post_ms, cls_ms):
        seg_name = self.active_seg_model_name()
        cls_name = self.active_cls_model_name()
        try:
            with open(PERFORMANCE_LOG_FILE, mode='a', newline='') as f:
                csv.writer(f).writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    seg_name, cls_name,
                    f"{total_ms:.2f}",
                    f"{pre_ms:.2f}",
                    f"{seg_ms:.2f}",
                    f"{post_ms:.2f}",
                    f"{cls_ms:.2f}"
                ])
            print(
                f"Logged Performance: Total={total_ms:.2f}ms (Pre={pre_ms:.2f}, Seg={seg_ms:.2f}, "
                f"Post={post_ms:.2f}, Cls={cls_ms:.2f} [{cls_name}])"
            )
        except Exception as e:
            print(f"Error logging performance: {e}")

    def _analyze_core(self, frame_bgr):
        if not self.models_loaded or not self.available_backends():
            empty_img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
            empty = (empty_img, "Modellek betöltése...", 0.0, empty_img, 0.0)
            return empty, None

        if self.classifier_backend == "dinov2":
            return self._analyze_dinov2(frame_bgr)

        t_start = time.perf_counter()

        t_pre_start = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        input_tensor, trans_img_tensor = self.preprocess(pil_img)
        t_pre_end = time.perf_counter()

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = trans_img_tensor.cpu().numpy().transpose(1, 2, 0)
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        orig_img_np = (img_np * 255).astype(np.uint8)
        trans_img_vis = orig_img_np

        use_vit = self.classifier_backend == "vit" and self.vit_classifier is not None

        if use_vit:
            t_post_start = time.perf_counter()

            segmented_polyp = orig_img_np.copy()
            pred_mask = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
            seg_conf = 0.0
            bbox = (0, INPUT_SIZE - 1, 0, INPUT_SIZE - 1)

            t_cls_start = time.perf_counter()
            vit_input_tensor = preprocess_rgb_uint8(orig_img_np)
            cls_result, conf_score, malignant_prob = predict_vit(
                self.vit_classifier, vit_input_tensor, threshold=VIT_THRESHOLD
            )
            t_cls_end = time.perf_counter()
            t_post_end = time.perf_counter()
            t_end = time.perf_counter()

            self._log_performance(
                total_ms=(t_end - t_start) * 1000,
                pre_ms=(t_pre_end - t_pre_start) * 1000,
                seg_ms=0.0,
                post_ms=(t_post_end - t_post_start) * 1000,
                cls_ms=(t_cls_end - t_cls_start) * 1000,
            )

            results = (segmented_polyp, cls_result, conf_score, trans_img_vis, seg_conf)
            ctx = ExplainContext(
                segmented_polyp=segmented_polyp,
                input_tensor=None,
                roi_batch=None,
                bbox=bbox,
                cls_label=cls_result,
                malignant_prob=malignant_prob,
                model_type=self.current_model_type,
                pred_mask=pred_mask,
                vit_input_tensor=vit_input_tensor.detach().clone(),
            )
            self._last_explain_ctx = ctx
            return results, ctx

        t_inf_start = time.perf_counter()
        with torch.no_grad():
            output = self.model(input_tensor)
            if self.current_model_type == "unet":
                probs = torch.sigmoid(output).squeeze().cpu().numpy()
                pred_mask = (probs > 0.5).astype(np.uint8)
            else:
                probs = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
                pred_mask = output.max(1)[1].cpu().numpy()[0]
        t_inf_end = time.perf_counter()

        t_post_start = time.perf_counter()

        final_img = np.zeros_like(orig_img_np)
        polyp_mask_bool = pred_mask.astype(bool)
        polyp_mask_rgb = np.stack([polyp_mask_bool] * 3, axis=-1)
        final_img[polyp_mask_rgb] = orig_img_np[polyp_mask_rgb]

        segmented_polyp = final_img
        seg_conf = float(np.mean(probs[polyp_mask_bool])) if np.any(polyp_mask_bool) else 0.0

        rows = np.any(polyp_mask_bool, axis=1)
        cols = np.any(polyp_mask_bool, axis=0)

        cls_result = "No Polyp Detected"
        conf_score = 0.0
        malignant_prob = 0.0
        classify_ms = 0.0
        bbox = None
        roi_batch = None

        if np.any(rows) and np.any(cols):
            rmin, rmax = int(np.where(rows)[0][[0, -1]][0]), int(np.where(rows)[0][[0, -1]][1])
            cmin, cmax = int(np.where(cols)[0][[0, -1]][0]), int(np.where(cols)[0][[0, -1]][1])
            bbox = (rmin, rmax, cmin, cmax)

            white_bg_img = np.full_like(orig_img_np, 255)
            white_bg_img[polyp_mask_rgb] = orig_img_np[polyp_mask_rgb]
            roi = self._extract_roi(white_bg_img, polyp_mask_bool)
            t_cls_start = time.perf_counter()
            cls_result, conf_score, malignant_prob, roi_batch = self._classify_roi(roi)
            t_cls_end = time.perf_counter()
            classify_ms = (t_cls_end - t_cls_start) * 1000

        t_post_end = time.perf_counter()
        t_end = time.perf_counter()

        self._log_performance(
            total_ms=(t_end - t_start) * 1000,
            pre_ms=(t_pre_end - t_pre_start) * 1000,
            seg_ms=(t_inf_end - t_inf_start) * 1000,
            post_ms=(t_post_end - t_post_start) * 1000,
            cls_ms=classify_ms,
        )

        results = (segmented_polyp, cls_result, conf_score, trans_img_vis, seg_conf)
        keep_cam = bbox is not None
        ctx = ExplainContext(
            segmented_polyp=segmented_polyp,
            input_tensor=input_tensor.detach().clone() if keep_cam else None,
            roi_batch=roi_batch if keep_cam else None,
            bbox=bbox,
            cls_label=cls_result,
            malignant_prob=malignant_prob,
            model_type=self.current_model_type,
            pred_mask=pred_mask,
            vit_input_tensor=None,
        )
        self._last_explain_ctx = ctx
        return results, ctx

    def _analyze_dinov2(self, frame_bgr):
        """Teljes képes klasszifikáció. A CAM a gráfban készül, külön magyarázó lépés nincs."""
        t_start = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        net_input = np.asarray(letterbox_to_square(pil_img)[0])
        t_pre_end = time.perf_counter()

        res = self.dinov2.predict(pil_img)
        t_cls_end = time.perf_counter()

        p1 = res.prob_neoplastic
        cls_result = CLASSIFIER_CLASS_NAMES[res.class_index]
        conf_score = p1 if res.class_index == 1 else 1.0 - p1

        h, w = frame_rgb.shape[:2]
        ctx = ExplainContext(
            segmented_polyp=frame_rgb,
            input_tensor=None,
            roi_batch=None,
            bbox=(0, h - 1, 0, w - 1),
            cls_label=cls_result,
            malignant_prob=p1,
            model_type=self.current_model_type,
            pred_mask=np.zeros((h, w), dtype=np.uint8),
            dinov2_overlay=np.asarray(res.overlay),
        )
        self._last_explain_ctx = ctx

        t_end = time.perf_counter()
        self._log_performance(
            total_ms=(t_end - t_start) * 1000,
            pre_ms=(t_pre_end - t_start) * 1000,
            seg_ms=0.0,
            post_ms=0.0,
            cls_ms=(t_cls_end - t_pre_end) * 1000,
        )
        return (frame_rgb, cls_result, conf_score, net_input, 0.0), ctx

    def _classify_roi(self, roi):
        """Címke, konfidencia, nyers neoplasztikus valószínűség és a Grad-CAM-hoz a ROI batch."""
        if self.classifier is None:
            return "Classifier unavailable", 0.0, 0.0, None
        if roi is None:
            return CLASSIFIER_CLASS_NAMES[0], 0.0, 0.0, None
        roi_batch = np.expand_dims(roi, 0)
        pred = self.classifier.predict(roi_batch, verbose=0)
        malignant_prob = float(pred[0][0])
        if malignant_prob >= 0.5:
            return CLASSIFIER_CLASS_NAMES[1], malignant_prob, malignant_prob, roi_batch
        return CLASSIFIER_CLASS_NAMES[0], 1.0 - malignant_prob, malignant_prob, roi_batch

    def compute_heatmap(self, ctx: ExplainContext | None, mode: str):
        """Grad-CAM / ViT attention overlay; (display_rgb, explain_ms) vagy (None, 0)."""
        if mode == "off" or ctx is None or ctx.bbox is None:
            return None, 0.0

        if ctx.dinov2_overlay is not None:
            return (ctx.dinov2_overlay, 0.0) if mode == "classification" else (None, 0.0)

        t0 = time.perf_counter()
        try:
            if mode == "classification":
                if (
                    self.classifier_backend == "vit"
                    and self.vit_classifier is not None
                    and ctx.vit_input_tensor is not None
                ):
                    cam_224 = vit_attention_map(self.vit_classifier, ctx.vit_input_tensor)
                    cam_full = cv2.resize(
                        cam_224, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR
                    )
                elif (
                    self.classifier is not None
                    and ctx.roi_batch is not None
                    and self._keras_cam_layer is not None
                ):
                    class_index = 1 if ctx.malignant_prob >= 0.5 else 0
                    cam_small = grad_cam_keras(
                        self.classifier,
                        ctx.roi_batch,
                        class_index,
                        self._keras_cam_layer,
                        grad_model=self._keras_grad_model,
                    )
                    cam_full = map_cam_to_full_frame(
                        cam_small, ctx.bbox, out_hw=(INPUT_SIZE, INPUT_SIZE)
                    )
                else:
                    return None, 0.0
            elif mode == "segmentation":
                if ctx.input_tensor is None:
                    return None, 0.0
                cam_full = grad_cam_torch(
                    self.model,
                    ctx.input_tensor,
                    ctx.model_type,
                    target_class=1,
                )
            else:
                return None, 0.0

            display_rgb = blend_heatmap(ctx.segmented_polyp, cam_full)
            explain_ms = (time.perf_counter() - t0) * 1000
            tag = "attention" if (
                mode == "classification" and self.classifier_backend == "vit"
            ) else "gradcam"
            print(f"Explain heatmap ({tag}): {explain_ms:.1f}ms (mode={mode})")
            return display_rgb, explain_ms
        except Exception as e:
            print(f"Heatmap error ({mode}): {e}")
            traceback.print_exc()
            return None, 0.0


# (világos, sötét) párok. Minden szöveg/háttér pár legalább 4.5:1, a keret hátterén (#DBDBDB / #2B2B2B) is.
_CLASS_COLORS = {
    "non": (("#176B41", "#7FD6A8"), ("#E5F4EC", "#17382A")),
    "neo": (("#9A4A0B", "#F0A868"), ("#F8EDE3", "#3F2A17")),
    "neutral": (("#4A525A", "#B8BEC5"), ("#EEF1F4", "#2F3439")),
}

# Egyetlen akcentus (Elemzés), a többi gomb semleges. Szöveg: fehér az akcentuson, sötét/világos a semlegesen.
_ACCENT = ("#1F5FA8", "#2E6DB8")
_ACCENT_HOVER = ("#184E8C", "#245FA0")
_BTN_NEUTRAL = ("#DDE1E6", "#3D434A")
_BTN_NEUTRAL_HOVER = ("#CBD1D8", "#4D545C")
_BTN_NEUTRAL_TEXT = ("#16191D", "#F2F4F6")
_BTN_BORDER = ("#6B7480", "#8A929B")

_NO_VALUE = "nincs"

# Fókuszgyűrű: legalább 3:1 a kerethez (#DBDBDB / #2B2B2B).
_FOCUS_RING = ("#0B3D91", "#FFD54A")

TOOLBAR_H = 92
STATUS_H = 28


def make_keyboard_accessible(widget, activate):
    """Tab-bal elérhetővé, Enterrel aktiválhatóvá és látható fókuszgyűrűvel látja el a CTk vezérlőt.

    A CustomTkinter vászon-alapú gombjai és menüi alapból kimaradnak a Tab-sorrendből.
    """
    canvas = widget._canvas
    canvas.configure(takefocus=1)

    def show_ring(_event=None):
        canvas.delete("focus_ring")
        color = _FOCUS_RING[1 if ctk.get_appearance_mode() == "Dark" else 0]
        canvas.create_rectangle(
            1, 1, max(2, canvas.winfo_width() - 1), max(2, canvas.winfo_height() - 1),
            outline=color, width=2, tags="focus_ring",
        )

    def hide_ring(_event=None):
        canvas.delete("focus_ring")

    def on_enter(_event):
        activate()
        return "break"

    canvas.bind("<FocusIn>", show_ring, add="+")
    canvas.bind("<FocusOut>", hide_ring, add="+")
    canvas.bind("<Return>", on_enter, add="+")
    canvas.bind("<KP_Enter>", on_enter, add="+")


def display_class_name(cls_res):
    """GUI felirat: Nem neoplasztikus / Neoplasztikus / Nincs polip."""
    raw = (cls_res or "").strip()
    low = raw.lower()
    if "no polyp" in low or "nincs polip" in low:
        return "Nincs polip", "neutral"
    if "non-neoplastic" in low or low.startswith("non-neo") or low.startswith("nem neo"):
        return "Nem neoplasztikus", "non"
    if "neoplastic" in low or low.startswith("neo"):
        return "Neoplasztikus", "neo"
    if not raw or raw == _NO_VALUE:
        return _NO_VALUE, "neutral"
    return raw, "neutral"


def aspect_name_for_ratio(ratio):
    """16:9, 16:10 vagy 4:3. A 16:10 ugyanazt az elrendezést kapja, mint a 16:9."""
    if ratio >= 1.7:
        return "16:9", 16 / 9
    if ratio >= 1.5:
        return "16:10", 16 / 10
    return "4:3", 4 / 3


def _work_area(root):
    """Windows munkaterület (tálca nélkül). Sikertelen híváskor a teljes képernyő."""
    try:
        rect = wintypes.RECT()
        ok = ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(rect), 0)
        if ok:
            width = int(rect.right - rect.left)
            height = int(rect.bottom - rect.top)
            if width > 400 and height > 300:
                return width, height, int(rect.left), int(rect.top)
    except (AttributeError, OSError, ValueError):
        pass
    return root.winfo_screenwidth(), root.winfo_screenheight(), 0, 0


def choose_fixed_window(root):
    """Egyszeri ablakméret a monitor arányából. Utána nem méretezhető.

    A Win32 munkaterület csak akkor számít, ha ugyanabban a pixelterben van mint a Tk
    képernyő, különben a DPI-skála kétszer érvényesülne.
    """
    screen_w = max(1, int(root.winfo_screenwidth()))
    screen_h = max(1, int(root.winfo_screenheight()))
    name, aspect = aspect_name_for_ratio(screen_w / screen_h)
    area_w, area_h, area_x, area_y = _work_area(root)
    if area_w <= int(screen_w * 1.05) and area_h <= int(screen_h * 1.05):
        box_w, box_h, origin_x, origin_y = area_w, area_h, area_x, area_y
    else:
        box_w, box_h, origin_x, origin_y = screen_w, screen_h, 0, 0
    max_w = max(1, box_w)
    max_h = max(1, box_h)
    if max_w / max_h > aspect:
        win_h = max_h
        win_w = int(round(win_h * aspect))
    else:
        win_w = max_w
        win_h = int(round(win_w / aspect))
    x = origin_x + max(0, (box_w - win_w) // 2)
    y = origin_y + max(0, (box_h - win_h) // 2)
    root.geometry(f"{win_w}x{win_h}+{x}+{y}")
    root.resizable(False, False)
    return win_w, win_h, name


def compute_preview_slots(win_w, win_h):
    """Bal oldal: két kisebb kép egymás alatt. Jobb oldal: a nagy eredmény.

    A 16:10 ugyanazt a számítást használja, mint a 16:9. A slotok a teljes
    ablakmagasságot kitöltik.
    """
    toolbar_h = TOOLBAR_H
    pad = 10
    title_h = 26
    text_h = 92
    footer_h = 22
    gap = 8
    inner_w = max(320, win_w - 2 * pad)
    inner_h = max(240, win_h - toolbar_h - STATUS_H - 2 * pad)
    small_img = max(100, (inner_h - 2 * title_h - footer_h - gap) // 2)
    result_img = max(small_img + 32, inner_h - title_h - text_h)
    overflow = small_img + gap + result_img - (inner_w - 8)
    if overflow > 0:
        cut_result = min(overflow, max(0, result_img - (small_img + 32)))
        result_img -= cut_result
        overflow -= cut_result
        if overflow > 0:
            small_img = max(80, small_img - overflow)
            result_img = max(small_img, inner_w - 8 - gap - small_img)
    return small_img, result_img, footer_h


class App:
    def __init__(self, root, window_title="HDMI Live Polyp Segmentation"):
        self.root = root
        self.root.title(window_title)
        self._win_w, self._win_h, self._aspect_name = choose_fixed_window(root)
        self._live_img_side, self._result_img_side, self._footer_row_h = compute_preview_slots(
            self._win_w, self._win_h
        )
        self._snap_img_side = self._live_img_side
        self._preview_sz = self._live_img_side

        self._live_image_ref = None
        self._snap_image_ref = None
        self._seg_image_ref = None
        # A programozott slider.set() ne fusson össze a felhasználói húzással.
        self._slider_updating = False
        self._slider_user_until = 0.0

        self.analyzer = MedicalAnalyzer(DEVICE)
        self.video_thread = None
        self._last_result = None
        self._is_analyzing_live = False
        self._analysis_busy = False
        self._clf_change_guard = False
        self._explain_generation = 0
        # A heatmap külön szálon, sorosan fut, hogy ne blokkolja az élőképet.
        self._explain_lock = threading.Lock()
        # Cache: alap kép és mindkét magyarázó réteg, így a menüváltás nem számol újra.
        self._overlay_cache = {
            "base": None,
            "classification": None,
            "segmentation": None,
        }
        self._overlay_cache_gen = -1
        self._last_snap_rgb = None
        self._last_seg_rgb = None

        # SQLite: trigger-eseményenként audit-napló (maszk BLOB + metaadatok).
        try:
            self._db_conn = _init_predictions_db()
        except sqlite3.Error as e:
            print(f"SQLite init error: {e}")
            self._db_conn = None
        self._db_lock = threading.Lock()

        self.main_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.top_panel = ctk.CTkFrame(self.main_frame, height=TOOLBAR_H, corner_radius=0)
        self.top_panel.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.top_panel.pack_propagate(False)
        row_source = ctk.CTkFrame(self.top_panel, fg_color="transparent", height=TOOLBAR_H // 2)
        row_source.pack(fill=tk.X)
        row_model = ctk.CTkFrame(self.top_panel, fg_color="transparent", height=TOOLBAR_H // 2)
        row_model.pack(fill=tk.X)

        self.camera_list = list_available_cameras() or ["No Camera"]
        default_cam = "No Camera"
        if self.camera_list and self.camera_list[0] != "No Camera":
            candidates = [c for c in self.camera_list if "USB" in c or "Capture" in c]
            if candidates:
                default_cam = candidates[0]
            elif len(self.camera_list) > 1:
                default_cam = self.camera_list[1]
            else:
                default_cam = self.camera_list[0]

        def neutral_button(parent, text, command, width):
            return ctk.CTkButton(
                parent, text=text, command=command, width=width, height=28,
                fg_color=_BTN_NEUTRAL, hover_color=_BTN_NEUTRAL_HOVER, text_color=_BTN_NEUTRAL_TEXT,
                border_width=1, border_color=_BTN_BORDER,
            )

        menu_style = dict(
            fg_color=_BTN_NEUTRAL, button_color=_BTN_NEUTRAL_HOVER, button_hover_color=_BTN_NEUTRAL_HOVER,
            text_color=_BTN_NEUTRAL_TEXT,
        )

        ctk.CTkLabel(row_source, text="Forrás").pack(side=tk.LEFT, padx=(10, 2))
        self.combo_cam = ctk.CTkComboBox(
            row_source, values=self.camera_list, width=260, state="readonly",
            button_color=_BTN_BORDER, button_hover_color=_BTN_BORDER,
            border_color=_BTN_BORDER,
        )
        self.combo_cam.set(default_cam)
        self.combo_cam.pack(side=tk.LEFT, padx=2)

        self.btn_hdmi = neutral_button(row_source, "Élő indítás", lambda: self.start_source(sim=False), 96)
        self.btn_hdmi.pack(side=tk.LEFT, padx=3, pady=8)
        self.btn_sim = neutral_button(row_source, "Szimuláció", lambda: self.start_source(sim=True), 96)
        self.btn_sim.pack(side=tk.LEFT, padx=3, pady=8)

        self.check_live_analysis_var = tk.BooleanVar(value=False)
        self.check_live_analysis = ctk.CTkCheckBox(
            row_source, text="Élő elemzés", variable=self.check_live_analysis_var,
            fg_color=_ACCENT, hover_color=_ACCENT_HOVER,
        )
        self.check_live_analysis.pack(side=tk.LEFT, padx=(10, 4))

        ctk.CTkLabel(row_model, text="Szegmentáló").pack(side=tk.LEFT, padx=(10, 2))
        self._seg_labels = {
            "DeepLabV3+": "deeplab",
            "U-Net": "unet",
        }
        self.seg_var = tk.StringVar(value="DeepLabV3+")
        self.seg_menu = ctk.CTkOptionMenu(
            row_model,
            values=list(self._seg_labels.keys()),
            variable=self.seg_var,
            command=self._on_seg_model_change,
            width=112,
            height=28,
            **menu_style,
        )
        self.seg_menu.pack(side=tk.LEFT, padx=(0, 6), pady=8)

        ctk.CTkLabel(row_model, text="Klasszifikátor").pack(side=tk.LEFT, padx=(4, 2))
        self._clf_labels = {spec.label: key for key, spec in CLASSIFIERS.items()}
        self.clf_var = tk.StringVar(value=CLASSIFIERS["dinov2"].label)
        self.clf_menu = ctk.CTkOptionMenu(
            row_model,
            values=list(self._clf_labels.keys()),
            variable=self.clf_var,
            command=self._on_classifier_change,
            width=124,
            height=28,
            **menu_style,
        )
        self.clf_menu.pack(side=tk.LEFT, padx=(0, 6), pady=8)

        ctk.CTkLabel(row_model, text="Magyarázat").pack(side=tk.LEFT, padx=(4, 2))
        self._explain_labels = {
            "Klasszifikáció": "classification",
            "Szegmentálás": "segmentation",
            "Ki": "off",
        }
        self.explain_var = tk.StringVar(value="Klasszifikáció")
        self._explain_last_mode = "classification"
        self.explain_menu = ctk.CTkOptionMenu(
            row_model,
            values=list(self._explain_labels.keys()),
            variable=self.explain_var,
            command=self._on_explain_mode_change,
            width=124,
            height=28,
            **menu_style,
        )
        self.explain_menu.pack(side=tk.LEFT, padx=(0, 10), pady=8)
        self.analyzer.explain_mode = "classification"

        self.btn_analyze = ctk.CTkButton(
            row_model, text="Elemzés (Space)", command=self.capture_and_analyze,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=_ACCENT, hover_color=_ACCENT_HOVER, text_color=("#FFFFFF", "#FFFFFF"),
            width=124, height=28,
        )
        self.btn_analyze.pack(side=tk.LEFT, padx=3, pady=8)

        self.save_mode_var = tk.StringVar(value="manual")
        self.save_mode_var.trace_add("write", lambda *_: self._update_save_button_state())
        self._save_mode_labels = {"Kézi mentés": "manual", "Auto mentés": "auto"}
        self.save_mode_menu = ctk.CTkOptionMenu(
            row_model,
            values=list(self._save_mode_labels.keys()),
            command=lambda choice: self.save_mode_var.set(self._save_mode_labels.get(choice, "manual")),
            width=112,
            height=28,
            **menu_style,
        )
        self.save_mode_menu.set("Kézi mentés")
        self.save_mode_menu.pack(side=tk.LEFT, padx=(14, 2), pady=8)
        self.btn_save = neutral_button(row_model, "Mentés", self._manual_save, 72)
        self.btn_save.pack(side=tk.LEFT, padx=3, pady=8)

        for control in (
            self.btn_hdmi, self.btn_sim, self.btn_analyze, self.btn_save,
        ):
            make_keyboard_accessible(control, control.invoke)
        for control in (
            self.seg_menu, self.clf_menu, self.explain_menu, self.save_mode_menu,
        ):
            make_keyboard_accessible(control, control._clicked)
        make_keyboard_accessible(self.check_live_analysis, self.check_live_analysis.toggle)

        def _panel_frame(parent, title):
            f = ctk.CTkFrame(parent, corner_radius=8)
            lbl = ctk.CTkLabel(f, text=title, font=ctk.CTkFont(size=12, weight="bold"))
            lbl.pack(anchor="w", padx=6, pady=(4, 2))
            f._title_label = lbl
            return f

        small = self._live_img_side
        footer = self._footer_row_h
        result = self._result_img_side
        small_h = small + footer
        hint_wrap = max(120, small - 24)

        self.left_column = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.left_column.grid(row=1, column=0, padx=(8, 4), pady=8, sticky="n")

        self.live_panel_frame = _panel_frame(self.left_column, "Élőkép")
        self.live_panel_frame.pack(anchor="n", pady=(0, 6))
        self._live_stack = ctk.CTkFrame(
            self.live_panel_frame, fg_color="transparent", width=small, height=small_h
        )
        self._live_stack.grid_propagate(False)
        self._live_stack.pack(anchor="n", padx=6, pady=(0, 6))
        self._live_stack.grid_columnconfigure(0, weight=1)
        self._live_stack.grid_rowconfigure(0, weight=0, minsize=small)
        self._live_stack.grid_rowconfigure(1, weight=0, minsize=footer)
        self.live_panel = ctk.CTkLabel(
            self._live_stack,
            text="Nincs forrás. Válassz kamerát vagy videót, majd kattints az Élő indításra.",
            wraplength=hint_wrap,
        )
        self.live_panel.grid(row=0, column=0, sticky="nsew")
        self.video_slider = ctk.CTkSlider(
            self._live_stack,
            from_=0,
            to=100,
            command=self.on_slider_move,
            height=16,
            button_length=14,
        )
        self._video_slider_packed = False

        self.snapshot_panel_frame = _panel_frame(self.left_column, "Hálózat bemenet")
        self.snapshot_panel_frame.pack(anchor="n")
        self._snapshot_inner = ctk.CTkFrame(
            self.snapshot_panel_frame, fg_color="transparent", width=small, height=small
        )
        self._snapshot_inner.grid_propagate(False)
        self._snapshot_inner.pack(anchor="n", padx=6, pady=(0, 6))
        self._snapshot_inner.grid_columnconfigure(0, weight=1)
        self._snapshot_inner.grid_rowconfigure(0, weight=0, minsize=small)
        self.snapshot_panel = ctk.CTkLabel(
            self._snapshot_inner,
            text="Az első elemzés után itt látszik, mit kap a hálózat.",
            wraplength=hint_wrap,
        )
        self.snapshot_panel.grid(row=0, column=0, sticky="nsew")

        self.result_panel_frame = _panel_frame(self.main_frame, "Eredmény")
        self.result_panel_frame.grid(row=1, column=1, padx=(4, 8), pady=8, sticky="n")
        self._result_inner = ctk.CTkFrame(
            self.result_panel_frame, fg_color="transparent", width=result, height=result
        )
        self._result_inner.grid_propagate(False)
        self._result_inner.pack(anchor="n", padx=6, pady=(0, 2))
        self._result_inner.grid_columnconfigure(0, weight=1)
        self._result_inner.grid_rowconfigure(0, weight=1)
        self.result_panel = ctk.CTkLabel(
            self._result_inner,
            text="Indíts forrást, majd nyomd meg a Space-t vagy az Elemzés gombot.",
            wraplength=max(160, result - 40),
        )
        self.result_panel.grid(row=0, column=0, sticky="nsew")

        self._readout = ctk.CTkFrame(self.result_panel_frame, fg_color="transparent")
        self._readout.pack(anchor="center", padx=8, pady=(2, 8))
        cap_font = ctk.CTkFont(size=13)
        neutral_fg, neutral_bg = _CLASS_COLORS["neutral"]
        self.lbl_class_caption = ctk.CTkLabel(self._readout, text="Osztály", font=cap_font)
        self.lbl_class_caption.grid(row=0, column=0, sticky="w", padx=(4, 16))
        self.lbl_class_name = ctk.CTkLabel(
            self._readout,
            text=_NO_VALUE,
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=neutral_fg,
            fg_color=neutral_bg,
            corner_radius=6,
            padx=10,
            pady=2,
        )
        self.lbl_class_name.grid(row=1, column=0, sticky="w", padx=(4, 16), pady=(0, 4))
        self.lbl_conf_caption = ctk.CTkLabel(self._readout, text="Konfidencia", font=cap_font)
        self.lbl_conf_caption.grid(row=0, column=1, sticky="w", padx=(8, 16))
        self.lbl_conf_value = ctk.CTkLabel(
            self._readout,
            text=_NO_VALUE,
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=neutral_fg,
        )
        self.lbl_conf_value.grid(row=1, column=1, sticky="w", padx=(8, 16))
        self._seg_readout = ctk.CTkFrame(self._readout, fg_color="transparent")
        self._seg_readout.grid(row=0, column=2, rowspan=2, sticky="nsw", padx=(8, 4))
        ctk.CTkLabel(self._seg_readout, text="Szegmentáció", font=cap_font).pack(anchor="w")
        self.lbl_seg_value = ctk.CTkLabel(
            self._seg_readout,
            text=_NO_VALUE,
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        self.lbl_seg_value.pack(anchor="w")
        self._seg_readout.grid_remove()

        self.status_bar = ctk.CTkFrame(self.main_frame, height=STATUS_H, corner_radius=0)
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.status_bar.pack_propagate(False)
        self.lbl_status = ctk.CTkLabel(
            self.status_bar, text="Rendszer inicializálása...", font=ctk.CTkFont(size=12), anchor="w"
        )
        self.lbl_status.pack(fill=tk.X, padx=10)

        self.main_frame.columnconfigure(0, weight=0)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=0)
        self.main_frame.rowconfigure(1, weight=1)
        self.main_frame.rowconfigure(2, weight=0)

        self.root.bind_all('<Key-space>', self._on_space_key)
        self.root.bind('<m>', self._toggle_explain_overlay)
        self.root.bind('<M>', self._toggle_explain_overlay)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._update_save_button_state()
        self.root.after(100, self._start_async_loading)
        self.update_live_feed()

    def _start_async_loading(self):
        """Modellek betöltése külön szálon, amíg a vezérlők le vannak tiltva."""
        self._set_ui_state("disabled")
        threading.Thread(target=self.analyzer.load_models, args=(self._update_loading_status,), daemon=True).start()

    def _update_loading_status(self, text):
        """Betöltő szálból hívva: a státusz és a menük frissítése a GUI szálon."""
        self.root.after(0, lambda t=text: self.lbl_status.configure(text=t))
        if isinstance(text, str) and (text.startswith("Rendszer kész") or text.startswith("HIBA:")):
            self.root.after(500, lambda: self._set_ui_state("normal"))
            self.root.after(500, self._refresh_classifier_menu)

    def _set_ui_state(self, state):
        """Gombok tiltása/engedélyezése (CustomTkinter: 'normal' / 'disabled')."""
        self.btn_hdmi.configure(state=state)
        self.btn_sim.configure(state=state)
        self.seg_menu.configure(state=state)
        self.clf_menu.configure(state=state)
        self.btn_analyze.configure(state=state)
        self.combo_cam.configure(state="disabled" if state == "disabled" else "readonly")
        if state == "normal":
            self._update_save_button_state()
        else:
            self.btn_save.configure(state=state)

    def _save_image(self, path, img_bgr):
        """PNG mentés imencode-dal: a cv2.imwrite Windows-on elhasal ékezetes útvonalon."""
        ok, buf = cv2.imencode(".png", img_bgr)
        if ok:
            with open(path, "wb") as f:
                f.write(buf.tobytes())

    def _save_results(self, frame, trans_img_vis, segmented_polyp, cls_res, conf, seg_conf=0.0,
                      seg_model="none", cls_model="none"):
        """Trigger-esemény rögzítése: PNG képek + SQLite sor (maszk BLOB + metaadatok)."""
        os.makedirs(OUTPUT_SAVE_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = os.path.join(OUTPUT_SAVE_DIR, timestamp)
        input_path = f"{prefix}_input.png"
        mask_path = f"{prefix}_masked.png"
        input_bgr = cv2.cvtColor(trans_img_vis, cv2.COLOR_RGB2BGR)
        mask_bgr = cv2.cvtColor(segmented_polyp, cv2.COLOR_RGB2BGR)
        self._save_image(input_path, input_bgr)
        self._save_image(mask_path, mask_bgr)

        ok, buf = cv2.imencode(".png", mask_bgr)
        mask_blob = buf.tobytes() if ok else None

        if self._db_conn is not None:
            try:
                with self._db_lock:
                    self._db_conn.execute(
                        """INSERT INTO predictions
                           (timestamp, class_name, confidence, seg_confidence,
                            seg_model, cls_model, input_path, mask_path, mask_blob)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            timestamp, cls_res, float(conf), float(seg_conf),
                            seg_model, cls_model, input_path, mask_path, mask_blob,
                        ),
                    )
                    self._db_conn.commit()
                self.lbl_status.configure(
                    text=f"Mentve DB-be: {os.path.basename(PREDICTIONS_DB_FILE)}"
                )
            except sqlite3.Error as e:
                print(f"SQLite insert error: {e}")
                self.lbl_status.configure(text=f"Mentés hiba (DB): {e}")
        else:
            self.lbl_status.configure(text=f"Mentve (csak PNG): {OUTPUT_SAVE_DIR}")

    def _manual_save(self):
        if self._last_result is None:
            self.lbl_status.configure(text="Nincs menthető eredmény. Előbb készíts egy elemzést.")
            return
        self._save_results(*self._last_result)

    def _update_save_button_state(self):
        """Auto módban a Mentés gomb tiltva."""
        self.btn_save.configure(state="normal" if self.save_mode_var.get() == "manual" else "disabled")

    def _on_seg_model_change(self, choice):
        model_type = self._seg_labels.get(choice, "deeplab")
        self.analyzer.switch_model(model_type)
        if model_type == "unet":
            self.lbl_status.configure(text="Szegmentáló: U-Net (ResNet34)")
        else:
            self.lbl_status.configure(text="Szegmentáló: DeepLabV3+ (MobileNet)")

    def _on_explain_mode_change(self, choice):
        mode = self._explain_labels.get(choice, "classification")
        if mode != "off":
            self._explain_last_mode = mode
        self.analyzer.explain_mode = mode
        self._show_cached_overlay(mode)

    def _toggle_explain_overlay(self, _event=None):
        """M: a legutóbbi magyarázat és a Ki állapot között vált, új elemzés nélkül."""
        if self.analyzer.explain_mode == "off":
            mode = self._explain_last_mode or "classification"
        else:
            self._explain_last_mode = self.analyzer.explain_mode
            mode = "off"
        inv = {v: k for k, v in self._explain_labels.items()}
        label = inv.get(mode, "Ki")
        self.explain_var.set(label)
        self._on_explain_mode_change(label)
        return "break"

    def _on_space_key(self, event):
        """Space bárhonnan, a legördülő fókusza után is. Beviteli mezőben nem nyeljük le."""
        if getattr(event, "repeat", False) or self._analysis_busy:
            return "break"
        if isinstance(event.widget, (tk.Entry, tk.Text)):
            return None
        self.capture_and_analyze()
        return "break"

    def _on_classifier_change(self, choice):
        """Klasszifikátorváltás. A menü parancsa közben nem rajzolunk (a CustomTkinter _draw
        update_idletasks-ot hív, és a nyitott menü fogása beragad), a felület 50 ms múlva frissül."""
        if self._clf_change_guard:
            return
        self._clf_change_guard = True
        try:
            backend = self._clf_labels.get(choice, self.analyzer.classifier_backend)
            applied = self.analyzer.switch_classifier_backend(backend)
            if not self.analyzer.full_frame:
                self.analyzer.switch_model(self._seg_labels.get(self.seg_var.get(), "deeplab"))
        finally:
            self._clf_change_guard = False
        self.root.after(50, lambda applied=applied: self._finish_classifier_switch_ui(applied))

    def _finish_classifier_switch_ui(self, applied):
        """Menü bezárása után: menük, Elemzés gomb, fókusz, placeholder, státusz."""
        try:
            self.root.grab_release()
        except tk.TclError:
            pass
        self._sync_menus_for_classifier()
        if not self._analysis_busy and self.analyzer.available_backends():
            self.btn_analyze.configure(state="normal")
        self._reset_result_panel_for_classifier()
        spec = CLASSIFIERS.get(applied)
        if spec is not None:
            self.lbl_status.configure(text=f"Klasszifikátor: {spec.label}. {spec.status}")
        try:
            self.root.focus_set()
        except tk.TclError:
            pass

    def _reset_result_panel_for_classifier(self):
        """Modellváltáskor a régi heatmap és cache érvénytelen, a panel címe és szövege frissül."""
        self._next_explain_generation()
        self._overlay_cache = {
            "base": None,
            "classification": None,
            "segmentation": None,
        }
        self._overlay_cache_gen = -1
        self._last_seg_rgb = None
        title = "Eredmény"
        if not self.analyzer.available_backends():
            placeholder = f"Nincs betöltött modell. DINOv2 ONNX helye: {self.analyzer.dinov2_path}"
        elif self.analyzer.full_frame:
            title = "Eredmény (teljes kép)"
            placeholder = "Space: teljes képes klasszifikáció, szegmentálás nélkül"
        else:
            placeholder = "Space: szegmentálás + ROI klasszifikáció"
        if hasattr(self.result_panel_frame, "_title_label"):
            self.result_panel_frame._title_label.configure(text=title)
        blank = Image.new("RGB", (4, 4), (0, 0, 0))
        self._seg_image_ref = ctk.CTkImage(
            light_image=blank, dark_image=blank, size=(4, 4)
        )
        self._set_ctk_label_image(self.result_panel, self._seg_image_ref, placeholder)
        self._apply_readout(None)

    def _sync_menus_for_classifier(self):
        """Teljes képes klasszifikátornál nincs szegmentálás, így a Szegmentáló menü és a
        Szegmentálás magyarázat nem választható."""
        seg_ready = self.analyzer.seg_models_available()
        seg_names = [label for label, key in self._seg_labels.items() if key in seg_ready]
        if seg_names:
            self.seg_menu.configure(
                values=seg_names, state="disabled" if self.analyzer.full_frame else "normal"
            )
            if self.seg_var.get() not in seg_names:
                self.seg_var.set(seg_names[0])
                self.analyzer.switch_model(self._seg_labels[seg_names[0]])
        else:
            self.seg_menu.configure(values=[_NO_VALUE], state="disabled")
            self.seg_var.set(_NO_VALUE)

        explain_names = [
            label for label, mode in self._explain_labels.items()
            if not (self.analyzer.full_frame and mode == "segmentation")
        ]
        self.explain_menu.configure(values=explain_names)
        if self.explain_var.get() not in explain_names:
            self.explain_var.set("Klasszifikáció")
            self.analyzer.explain_mode = "classification"
            self._explain_last_mode = "classification"

    def _refresh_classifier_menu(self):
        """Betöltés után: csak a ténylegesen betöltött klasszifikátorok szerepelnek a menüben."""
        ready = self.analyzer.available_backends()
        if not ready:
            self.clf_menu.configure(values=[_NO_VALUE], state="disabled")
            self.clf_var.set(_NO_VALUE)
            self.btn_analyze.configure(state="disabled")
        else:
            self.clf_menu.configure(values=[CLASSIFIERS[k].label for k in ready], state="normal")
            self.clf_var.set(CLASSIFIERS[self.analyzer.classifier_backend].label)
        self._sync_menus_for_classifier()
        self._reset_result_panel_for_classifier()

    def _next_explain_generation(self):
        self._explain_generation += 1
        return self._explain_generation

    def _apply_readout(self, cls_res, conf=None, seg_conf=None):
        """Osztály, konfidencia és (szegmentáló úton) szegmentáció. JNET nincs a feliraton."""
        if cls_res is None:
            name, kind = _NO_VALUE, "neutral"
            conf_txt = _NO_VALUE
        else:
            name, kind = display_class_name(cls_res)
            conf_txt = _NO_VALUE if conf is None else f"{conf * 100:.1f}%"
        fg, bg = _CLASS_COLORS[kind]
        self.lbl_class_name.configure(text=name, text_color=fg, fg_color=bg)
        self.lbl_conf_value.configure(text=conf_txt, text_color=fg)
        if self.analyzer.full_frame:
            self._seg_readout.grid_remove()
            return
        self._seg_readout.grid()
        seg_txt = _NO_VALUE if seg_conf is None else f"{seg_conf * 100:.1f}%"
        self.lbl_seg_value.configure(text=seg_txt)

    def _fit_rgb_square(self, rgb, side):
        """Négyzetes kijelzés, az arány megmarad. A modell bemenetét nem ez adja."""
        h, w = rgb.shape[:2]
        if h < 1 or w < 1:
            return np.zeros((side, side, 3), dtype=np.uint8)
        scale = min(side / h, side / w)
        nw = max(1, min(side, int(w * scale)))
        nh = max(1, min(side, int(h * scale)))
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_NEAREST)
        if nw == side and nh == side:
            return resized
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        y0 = (side - nh) // 2
        x0 = (side - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas

    def _set_ctk_label_image(self, panel, ctk_image, text):
        """A tk címke régi PhotoImage-ét előbb levesszük, különben a configure elhasal."""
        label = getattr(panel, "_label", None)
        if label is not None:
            try:
                label.configure(image="")
            except tk.TclError:
                pass
        panel.configure(image=ctk_image, text=text)

    def _update_seg_panel_image(self, segmented_rgb):
        self._last_seg_rgb = segmented_rgb
        sz = self._result_img_side
        fitted = self._fit_rgb_square(segmented_rgb, sz)
        seg_img = Image.fromarray(fitted)
        self._seg_image_ref = ctk.CTkImage(
            light_image=seg_img, dark_image=seg_img, size=(sz, sz)
        )
        self._set_ctk_label_image(self.result_panel, self._seg_image_ref, "")

    def _set_snap_panel_image(self, snap_rgb):
        self._last_snap_rgb = snap_rgb
        sz = self._snap_img_side
        fitted = self._fit_rgb_square(snap_rgb, sz)
        snap_img = Image.fromarray(fitted)
        self._snap_image_ref = ctk.CTkImage(
            light_image=snap_img, dark_image=snap_img, size=(sz, sz)
        )
        self._set_ctk_label_image(self.snapshot_panel, self._snap_image_ref, "")

    def _show_cached_overlay(self, mode=None):
        """Megjeleníti a választott réteget a cache-ből (off = base maszk)."""
        if mode is None:
            mode = self.analyzer.explain_mode
        if self._overlay_cache_gen != self._explain_generation:
            return
        if mode == "off":
            img = self._overlay_cache.get("base")
        else:
            img = self._overlay_cache.get(mode)
            if img is None:
                img = self._overlay_cache.get("base")
        if img is not None:
            self._update_seg_panel_image(img)

    def _store_overlay(self, mode, display_rgb, gen, explain_ms=0.0):
        """Háttérszálról: cache frissítés + ha ez az aktív mód, panel frissítés."""
        if gen != self._explain_generation:
            return
        self._overlay_cache[mode] = display_rgb
        if self.analyzer.explain_mode == mode:
            self._update_seg_panel_image(display_rgb)
            if explain_ms > 0:
                status = self.lbl_status.cget("text")
                base_status = status.split(" | Magyarázat")[0]
                kind = "szegmentálás" if mode == "segmentation" else "klasszifikáció"
                self.lbl_status.configure(
                    text=f"{base_status} | Magyarázat ({kind}) +{explain_ms:.0f} ms"
                )

    def _schedule_heatmap(self, ctx, gen):
        """Mindkét magyarázó réteg elkészül; a kiválasztott előbb, a másik utána. Váltás cache-ből."""
        if ctx is None or ctx.bbox is None:
            return

        self._overlay_cache = {
            "base": ctx.segmented_polyp.copy(),
            "classification": None,
            "segmentation": None,
        }
        self._overlay_cache_gen = gen

        preferred = self.analyzer.explain_mode
        if self.analyzer.full_frame:
            order = ["classification"]
        elif preferred == "off":
            order = ["classification", "segmentation"]
        elif preferred == "segmentation":
            order = ["segmentation", "classification"]
        else:
            order = ["classification", "segmentation"]

        def worker():
            with self._explain_lock:
                if gen != self._explain_generation:
                    return
                for mode in order:
                    if gen != self._explain_generation:
                        return
                    display_rgb, explain_ms = self.analyzer.compute_heatmap(ctx, mode)
                    if display_rgb is None:
                        continue
                    self.root.after(
                        0,
                        lambda d=display_rgb, m=mode, g=gen, e=explain_ms: self._store_overlay(
                            m, d, g, e
                        ),
                    )
                ctx.input_tensor = None
                ctx.roi_batch = None
                ctx.vit_input_tensor = None
                ctx.dinov2_overlay = None

        threading.Thread(target=worker, daemon=True, name="heatmap-worker").start()

    def start_source(self, sim=False):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None

        if sim:
            folder = os.path.join(_BASE_DIR, "images")
            if not os.path.isdir(folder):
                self.lbl_status.configure(text=f"Nincs képmappa a szimulációhoz: {folder}")
                self._reset_video_slider_for_non_file()
                return
            thread = VideoCaptureThread(simulation_mode=True, image_folder=folder)
            if not thread.image_files:
                thread.stop()
                self.lbl_status.configure(text=f"A szimulációs mappában nincs használható kép: {folder}")
                self._reset_video_slider_for_non_file()
                return
            self.video_thread = thread
            self.lbl_status.configure(text=f"Szimuláció indítva (képmappa: {folder})")
            self._reset_video_slider_for_non_file()
            return

        selection = self.combo_cam.get()
        try:
            if selection.startswith("Video: "):
                filename = selection.replace("Video: ", "")
                vpath = filename if os.path.isabs(filename) else os.path.join(_BASE_DIR, filename)
                if not os.path.isfile(vpath):
                    self.lbl_status.configure(text=f"A videófájl nem található: {vpath}")
                    self._reset_video_slider_for_non_file()
                    return
                self.video_thread = VideoCaptureThread(src=vpath, simulation_mode=False)
                self.lbl_status.configure(text=f"Videó indítva: {os.path.basename(vpath)}")
                self._show_video_slider()
                self._slider_updating = True
                self.video_slider.set(0)
                self._slider_updating = False
                return

            # "Camera N: név" vagy "Camera N"; ha nem értelmezhető, az első kamera.
            parts = selection.split()
            idx = int(parts[1].replace(':', '')) if len(parts) >= 2 and parts[0] == "Camera" else 0
            thread = VideoCaptureThread(src=idx, simulation_mode=False)
            if thread.cap is None or not thread.cap.isOpened():
                thread.stop()
                self.lbl_status.configure(
                    text=f"A kamera nem nyitható meg (index {idx}). Ellenőrizd a csatlakozást, vagy válassz videót."
                )
                self._hide_video_slider()
                return
            self.video_thread = thread
            self.lbl_status.configure(text=f"Kamera indítva (index {idx})")
            self._reset_video_slider_for_non_file()
        except Exception as e:
            print(f"Forrás indítási hiba: {e}")
            self.lbl_status.configure(text=f"Nem sikerült elindítani a forrást: {e}")
            self._hide_video_slider()

    def _show_video_slider(self):
        """Csúszka a videó alatt (grid 1. sor), csak videófájl módban."""
        if self._video_slider_packed:
            return
        self._video_slider_packed = True
        self.video_slider.configure(state="normal")
        self.video_slider.grid(row=1, column=0, sticky="ew", padx=2, pady=(2, 0))

    def _hide_video_slider(self):
        """Csúszka elrejtése (kamera, szimuláció, nincs videó)."""
        self._slider_updating = True
        try:
            self.video_slider.set(0)
            self.video_slider.configure(state="disabled")
            if self._video_slider_packed:
                self.video_slider.grid_remove()
                self._video_slider_packed = False
        finally:
            self._slider_updating = False

    def _reset_video_slider_for_non_file(self):
        self._hide_video_slider()

    def on_slider_move(self, val):
        if self._slider_updating:
            return
        self._slider_user_until = time.time() + 0.5
        if self.video_thread and self.video_thread.video_file_mode:
            self.video_thread.set_position(float(val))

    def update_live_feed(self):
        delay = 33
        try:
            if self.video_thread:
                ret, frame = self.video_thread.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    tgt = self._live_img_side
                    scale = min(tgt / h, tgt / w)
                    display_w = max(1, min(tgt, int(w * scale)))
                    display_h = max(1, min(tgt, int(h * scale)))

                    frame_resized = cv2.resize(frame, (display_w, display_h))

                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    self._live_image_ref = ctk.CTkImage(
                        light_image=img, dark_image=img, size=(max(1, display_w), max(1, display_h))
                    )
                    self._set_ctk_label_image(self.live_panel, self._live_image_ref, "")

                    # Élő elemzésnél nincs heatmap: nem terheli tovább a CPU-t és a memóriát.
                    if (
                        self.check_live_analysis_var.get()
                        and self.analyzer.models_loaded
                        and self.analyzer.available_backends()
                        and not self._is_analyzing_live
                    ):
                        self._is_analyzing_live = True
                        threading.Thread(
                            target=self._async_live_analyze,
                            args=(frame, display_w, display_h),
                            daemon=True,
                        ).start()

                # Videófájlnál a csúszka követi a lejátszást, húzás közben fél másodpercig nem.
                if (
                    self.video_thread
                    and self.video_thread.video_file_mode
                    and time.time() >= self._slider_user_until
                ):
                    pct = self.video_thread.get_position_percent()
                    if pct is not None:
                        self._slider_updating = True
                        try:
                            self.video_slider.set(pct)
                        finally:
                            self._slider_updating = False
        except Exception as e:
            # Kivétel nem törheti el az after() láncot, különben az élőkép megáll.
            print(f"Live feed update error: {e}")
        finally:
            try:
                self.root.after(delay, self.update_live_feed)
            except Exception:
                pass

    def _async_live_analyze(self, frame, display_w, display_h):
        try:
            results, _ctx = self.analyzer.analyze_with_context(frame)
            self.root.after(
                0,
                lambda: self._apply_live_analysis_result(frame, display_w, display_h, results),
            )
        except Exception as e:
            print(f"Élő elemzés hiba: {e}")
            self.root.after(0, lambda: setattr(self, '_is_analyzing_live', False))

    def _apply_live_analysis_result(self, frame, display_w, display_h, results):
        self.run_analysis_on_frame(frame, display_w, display_h, results=results)
        self._is_analyzing_live = False

    def run_analysis_on_frame(self, frame, display_w, display_h, results=None):
        """Eredmények megjelenítése. Ha results megadva, azt használja; különben analyze(frame)."""
        if results is not None:
            segmented_polyp, cls_res, conf, trans_img_vis, seg_conf = results
        else:
            segmented_polyp, cls_res, conf, trans_img_vis, seg_conf = self.analyzer.analyze(frame)

        self._set_snap_panel_image(trans_img_vis)
        self._update_seg_panel_image(segmented_polyp)

        title = "Eredmény (teljes kép)" if self.analyzer.full_frame else "Eredmény"
        if hasattr(self.result_panel_frame, "_title_label"):
            self.result_panel_frame._title_label.configure(text=title)
        self._apply_readout(cls_res, conf, None if self.analyzer.full_frame else seg_conf)

    def capture_and_analyze(self):
        if not self.analyzer.available_backends():
            self.lbl_status.configure(
                text=f"Nincs betöltött klasszifikátor. DINOv2 ONNX helye: {self.analyzer.dinov2_path}"
            )
            return
        if not self.video_thread:
            self.lbl_status.configure(text="Nincs forrás. Válassz kamerát vagy videót, majd kattints az Élő indításra.")
            return

        self.lbl_status.configure(text="Kép rögzítése...")
        self._analysis_busy = True
        self.btn_analyze.configure(state="disabled")
        self.root.update_idletasks()

        ret, frame = self.video_thread.read()
        if not ret or frame is None:
            self._analysis_busy = False
            self.lbl_status.configure(text="A forrás még nem adott képkockát. Próbáld újra egy pillanat múlva.")
            self.btn_analyze.configure(state="normal")
            return

        preview_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            self._set_snap_panel_image(preview_rgb)
            self._update_seg_panel_image(preview_rgb)
            self._apply_readout(None)
            self.lbl_class_name.configure(text="Elemzés...")
        except Exception as e:
            self._analysis_busy = False
            self.btn_analyze.configure(state="normal")
            self.lbl_status.configure(text=f"Hiba: {e}")
            return
        self._next_explain_generation()
        self.lbl_status.configure(text="Kép elkészült, elemzés...")
        self.root.update_idletasks()

        h, w = frame.shape[:2]
        display_h = self._snap_img_side
        scale = display_h / h
        display_w = int(w * scale)

        # A hálózat külön szálon fut, hogy a GUI ne fagyjon.
        threading.Thread(target=self._async_analyze, args=(frame, display_w, display_h), daemon=True).start()

    def _async_analyze(self, frame, display_w, display_h):
        gen = self._next_explain_generation()
        start_time = time.time()
        try:
            results, ctx = self.analyzer.analyze_with_context(frame)
            elapsed = time.time() - start_time
            # Az eredmény azonnal megjelenik, a magyarázó réteg külön szálon követi.
            self.root.after(
                0,
                lambda: self._apply_analysis_result(frame, display_w, display_h, results, elapsed),
            )
            self._schedule_heatmap(ctx, gen)
        except Exception as e:
            print(f"Elemzés hiba: {e}")
            traceback.print_exc()
            self.root.after(0, lambda err=e: self._finish_analysis_error(err))

    def _finish_analysis_error(self, err):
        self._analysis_busy = False
        self.lbl_status.configure(text=f"Hiba: {err}")
        self.btn_analyze.configure(state="normal")

    def _apply_analysis_result(self, frame, display_w, display_h, results, elapsed):
        try:
            segmented_polyp, cls_res, conf, trans_img_vis, seg_conf = results
            self._last_result = (
                frame, trans_img_vis, segmented_polyp, cls_res, conf, seg_conf,
                self.analyzer.active_seg_model_name(), self.analyzer.active_cls_model_name(),
            )
            self.run_analysis_on_frame(frame, display_w, display_h, results=results)
            if self.save_mode_var.get() == "auto":
                self._save_results(*self._last_result)
            shown, _kind = display_class_name(cls_res)
            self.lbl_status.configure(text=f"{shown} ({conf*100:.1f}%) | {elapsed:.3f} s")
        except Exception as e:
            print(f"Eredmény megjelenítés hiba: {e}")
            self.lbl_status.configure(text=f"Hiba: {e}")
        finally:
            self._analysis_busy = False
            self.btn_analyze.configure(state="normal")

    def on_close(self):
        try:
            self.root.unbind_all("<Key-space>")
        except tk.TclError:
            pass
        if self.video_thread:
            self.video_thread.stop()
        if getattr(self, "_db_conn", None) is not None:
            try:
                with self._db_lock:
                    self._db_conn.close()
            except sqlite3.Error as e:
                print(f"SQLite close error: {e}")
        self.root.destroy()


def run_app():
    crash_log = os.path.join(_BASE_DIR, "LiveCapture_crash.log")
    try:
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")
        root = ctk.CTk()
        App(root)
        root.mainloop()
    except Exception:
        tb = traceback.format_exc()
        try:
            with open(crash_log, "w", encoding="utf-8") as f:
                f.write(tb)
        except OSError:
            pass
        try:
            import tkinter.messagebox as _mb

            _mb.showerror("LiveCapture hiba", f"Indítási hiba. Részletek:\n{crash_log}")
        except Exception:
            pass
        raise SystemExit(1) from None


if __name__ == "__main__":
    run_app()
