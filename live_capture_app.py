import ctypes
import cv2
import threading
import time
import csv
import json
import sqlite3
import sys
import traceback
from ctypes import wintypes
from dataclasses import dataclass
import numpy as np
import torch
import os
try:
    import tensorflow as tf
    _TF_IMPORT_ERROR = None

    # Fix for Keras version mismatch with RandomContrast (value_range argument)
    try:
        original_init = tf.keras.layers.RandomContrast.__init__
        def patched_init(self, factor, seed=None, **kwargs):
            kwargs.pop("value_range", None)
            original_init(self, factor=factor, seed=seed, **kwargs)
        tf.keras.layers.RandomContrast.__init__ = patched_init
    except Exception as e:
        print(f"Failed to patch RandomContrast: {e}")

    # Register custom layer used in the new Keras model
    @tf.keras.utils.register_keras_serializable(package="PolpyCLI", name="ResNetPreprocessLayer")
    class ResNetPreprocessLayer(tf.keras.layers.Layer):
        def call(self, inputs):
            return tf.keras.applications.resnet_v2.preprocess_input(inputs)

except Exception as _tf_exc:
    tf = None
    _TF_IMPORT_ERROR = _tf_exc

import customtkinter as ctk
from PIL import Image
import network
from network.modeling import deeplabv3plus_mobilenet
import segmentation_models_pytorch as smp
import tkinter as tk
import random
import utils.ext_transforms as et
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

# --- Configuration ---
def _app_base_dir():
    """Fejlesztésben: script mappa; PyInstaller exe-nél: az .exe mappája (modellek ide másolandók)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


_BASE_DIR = _app_base_dir()
PERFORMANCE_LOG_FILE = os.path.join(_BASE_DIR, "performance_log.csv")

DEEPLAB_CHECKPOINT = os.path.join(_BASE_DIR, "checkpoints_0409", "best_model_0409_epoch_31.pth")
UNET_CHECKPOINT = os.path.join(_BASE_DIR, "checkpoints_unet", "best_model_epoch_22.pth")

# --- classificator_models/ (TensorFlow–Keras, .gitignore alatt is lehet) ---
#   *.keras            : tanított bináris klasszifikátor (ResNet50V2 + fej), betöltés: tf.keras.models.load_model
#   *_metadata.json    : input_shape, backbone — dokumentáció; ha létezik, a bemeneti méret onnan olvasható
#   *_history.json     : epochonkénti loss/acc (tanítás utáni elemzéshez), a futó app NEM használja
CLASSIFIER_MODEL_DIR = os.path.join(_BASE_DIR, "classificator_models")
CLASSIFIER_KERAS_PREFERRED = "cnn_s2_lr2e4_tb_os.keras"
CLASSIFIER_KERAS_LEGACY = "resnet50v2_polyp_20260217_131050.keras"
CLASSIFIER_INPUT_SIZE = 512
CLASSIFIER_CLASS_NAMES = ["Non-neoplastic (JNET 1)", "Neoplastic (JNET 2a/2b/3)"]
VIT_MODEL_DIR = os.path.join(CLASSIFIER_MODEL_DIR, "vit_gui_share")
VIT_CHECKPOINT_NAME = "vit_final_224.pth"


def _resolve_vit_path():
    """ViT .pth: env VIT_MODEL_PATH → classificator_models/vit_gui_share/vit_final_224.pth."""
    env = (os.environ.get("VIT_MODEL_PATH") or "").strip()
    if env and os.path.isfile(env):
        return env
    preferred = os.path.join(VIT_MODEL_DIR, VIT_CHECKPOINT_NAME)
    if os.path.isfile(preferred):
        return preferred
    return preferred


def _resolve_classifier_keras_path():
    """Klasszifikátor .keras útvonal: env → preferált fájl → régi név → legfrissebb *.keras a mappában."""
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
    """Ha van <stem>_metadata.json és input_shape, abból a négyzetes oldalhossz."""
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

NUM_CLASSES = 2
OUTPUT_STRIDE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_SIZE = 513
OUTPUT_SAVE_DIR = os.path.join(_BASE_DIR, "saved_results")
PREDICTIONS_DB_FILE = os.path.join(OUTPUT_SAVE_DIR, "predictions.db")


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
    
    # 1. Physical Cameras
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
        
    # 2. Video fájlok az alkalmazás mappájában (nem a cwd-től függően)
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    try:
        base_files = os.listdir(_BASE_DIR)
    except OSError:
        base_files = []
    for f in base_files:
        if f.lower().endswith(video_extensions):
            camera_list.append(f"Video: {f}")
        
    return camera_list

# --- 1. Threaded Video Capture ---
class VideoCaptureThread:
    def __init__(self, src=0, simulation_mode=False, image_folder=None):
        self.simulation_mode = simulation_mode
        self.image_folder = image_folder
        self.image_files = []
        self.video_file_mode = False
        
        if self.simulation_mode and self.image_folder:
             # Allowed IDs for filtering (Simulation)
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
             # Normalize to uppercase for comparison (except pi205 handled by lower())
             allowed_ids_lower = [pid.lower() for pid in allowed_ids]

             all_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
             self.image_files = []
             
             for f in all_files:
                 # Check if any ID is in filename
                 f_lower = f.lower()
                 if any(pid in f_lower for pid in allowed_ids_lower):
                     self.image_files.append(os.path.join(image_folder, f))
                     
             print(f"Simulation mode: Found {len(self.image_files)} images matching filter (Total allowed IDs: {len(allowed_ids)}).")

        if not self.simulation_mode:
            # Check if src is a string (video file path) or int (camera index)
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
            
        # Pick a random image
        img_path = random.choice(self.image_files)
        try:
            # Load image using OpenCV
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
        
        last_sim_update = time.time() # For simulation mode
        
        while self.running:
            start_loop = time.time()
            
            if not self.simulation_mode:
                if self.cap and self.cap.isOpened():
                    # For video files, we need to respect FPS inside the thread
                    if self.video_file_mode:
                        # Use lock to prevent conflict with set_position (seeking)
                        with self.lock:
                            ret, frame = self.cap.read()
                            
                            # Loop video
                            if not ret:
                                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                ret, frame = self.cap.read()
                            
                            if ret:
                                self.ret = ret
                                self.frame = frame
                        
                        # Wait to match FPS
                        elapsed = time.time() - start_loop
                        wait = frame_time - elapsed
                        if wait > 0:
                            time.sleep(wait)
                    else:
                        # Camera mode: read as fast as possible (hardware limits FPS)
                        # Lock reading to be safe too
                        with self.lock:
                            ret, frame = self.cap.read()
                            if ret:
                                self.ret = ret
                                self.frame = frame
                        
                        time.sleep(0.005) # Tiny sleep to prevent 100% CPU usage
                else:
                    time.sleep(0.01)
            else:
                # Simulation Mode
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
                # Heatmap / OOM közben ne törje el a live feed láncot
                return False, None

    def get_fps(self):
        with self.lock:
            if self.cap:
                 return self.cap.get(cv2.CAP_PROP_FPS)
        return 30.0 # Default

    def set_position(self, percent):
        with self.lock:
            if self.cap and self.video_file_mode:
                 total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                 frame_idx = int(total_frames * percent / 100.0)
                 self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def get_position_percent(self):
        """Aktuális lejátszási pozíció 0–100% (videófájl módban)."""
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


# --- 2. Analyzer (Segmentation & Classification) ---
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
        self.classifier_backend = "keras"  # "vit" | "keras" — load_models sets preferred

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

        self.transform = et.ExtCompose([
            et.ExtResize(size=INPUT_SIZE),
            et.ExtCenterCrop(size=INPUT_SIZE),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_models(self, status_callback=None):
        """Modellek betöltése és warm-up (külön szálon futtatható)."""
        self.models_loaded = False
        try:
            if status_callback:
                status_callback("DeepLabV3+ betöltése...")
            self.deeplab_model = self._load_deeplab()

            if status_callback:
                status_callback("U-Net betöltése...")
            self.unet_model = self._load_unet()

            self.model = self.deeplab_model

            if status_callback:
                status_callback("Modellek bemelegítése (warm-up)...")
            dummy_input_pt = torch.zeros((1, 3, INPUT_SIZE, INPUT_SIZE)).to(self.device)
            with torch.no_grad():
                _ = self.deeplab_model(dummy_input_pt)
                _ = self.unet_model(dummy_input_pt)
            
            # --- Klasszifikátorok: ViT (PyTorch, full-frame) + Keras ResNet (ROI) ---
            self.classifier = None
            self.vit_classifier = None
            self.vit_path = _resolve_vit_path()
            self.classifier_keras_path = _resolve_classifier_keras_path()
            self.classifier_input_size = _read_classifier_input_size(
                self.classifier_keras_path, CLASSIFIER_INPUT_SIZE
            )

            if status_callback:
                status_callback("ViT klasszifikátor betöltése...")
            if os.path.isfile(self.vit_path):
                try:
                    self.vit_classifier = load_vit_checkpoint(self.vit_path, self.device)
                    with torch.inference_mode():
                        _ = self.vit_classifier(
                            torch.zeros((1, 3, 224, 224), device=self.device)
                        )
                    print(f"ViT classifier: {self.vit_path} (input 224×224, full-frame)")
                except Exception as vit_err:
                    print(f"ViT betöltés sikertelen: {vit_err}")
                    self.vit_classifier = None
            else:
                print(f"ViT checkpoint nem található: {self.vit_path}")

            if status_callback:
                status_callback("Keras klasszifikátor betöltése (opcionális)...")
            if tf is not None:
                try:
                    tf.config.set_visible_devices([], "GPU")
                except Exception:
                    pass
                try:
                    tf.config.threading.set_inter_op_parallelism_threads(1)
                    tf.config.threading.set_intra_op_parallelism_threads(2)
                except Exception:
                    pass
                if os.path.isfile(self.classifier_keras_path):
                    try:
                        print(
                            f"Classifier (Keras ROI): {self.classifier_keras_path} "
                            f"(input {self.classifier_input_size}×{self.classifier_input_size})"
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
                else:
                    print(f"Keras klasszifikátor nem található: {self.classifier_keras_path}")
            else:
                print(f"TensorFlow import hiba (Keras ROI ki): {_TF_IMPORT_ERROR}")

            # Alap backend: ViT ha van, különben Keras
            if self.vit_classifier is not None:
                self.classifier_backend = "vit"
            elif self.classifier is not None:
                self.classifier_backend = "keras"
            else:
                self.classifier_backend = "keras"
            print(f"Classifier backend: {self.classifier_backend}")

            self.models_loaded = True
            if status_callback:
                has_clf = self.vit_classifier is not None or self.classifier is not None
                if not has_clf:
                    status_callback("Rendszer kész. (Klasszifikátor: nincs betöltve)")
                else:
                    status_callback(f"Rendszer kész. (Cls: {self.classifier_backend})")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.models_loaded = False
            msg = str(e).replace("\n", " ")[:280]
            if status_callback:
                status_callback(f"HIBA: modell betöltés sikertelen – {msg}")

    def _load_checkpoint(self, model, checkpoint_path):
        """Checkpoint betöltése modellbe, különböző kulcsformátumok kezelésével."""
        if not os.path.exists(checkpoint_path):
            print(f"WARNING: Checkpoint not found: {checkpoint_path}")
            return model
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
        return model

    def _load_deeplab(self):
        model = deeplabv3plus_mobilenet(num_classes=NUM_CLASSES, output_stride=OUTPUT_STRIDE)
        model = self._load_checkpoint(model, DEEPLAB_CHECKPOINT)
        model.to(self.device)
        model.eval()
        return model

    def _load_unet(self):
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        model = self._load_checkpoint(model, UNET_CHECKPOINT)
        model.to(self.device)
        model.eval()
        return model

    def switch_model(self, model_type):
        """Váltás 'deeplab' és 'unet' között."""
        if model_type == "deeplab":
            self.model = self.deeplab_model
            self.current_model_type = "deeplab"
            print("Switched to DeepLabV3+")
        elif model_type == "unet":
            self.model = self.unet_model
            self.current_model_type = "unet"
            print("Switched to U-Net")
        return self.current_model_type

    def switch_classifier_backend(self, backend: str) -> str:
        """Váltás 'vit' (full-frame) és 'keras' (ROI) között."""
        if backend == "vit":
            if self.vit_classifier is None:
                print("ViT nem elérhető")
                return self.classifier_backend
            self.classifier_backend = "vit"
            print("Classifier backend: ViT-B/16 (full-frame)")
        elif backend == "keras":
            if self.classifier is None:
                print("Keras ResNet nem elérhető")
                return self.classifier_backend
            self.classifier_backend = "keras"
            print("Classifier backend: ResNet50V2 (ROI)")
        return self.classifier_backend

    def active_cls_model_name(self) -> str:
        if self.classifier_backend == "vit" and self.vit_classifier is not None:
            return "ViT-B/16"
        if self.classifier is not None:
            return "ResNet50V2"
        return "none"


    def preprocess(self, image):
        # Image is a PIL Image
        # ext_transforms expects (img, mask), so we create a dummy mask
        w, h = image.size
        dummy_mask = Image.new('L', (w, h), 0)
        
        # Apply transformation
        img_trans, _ = self.transform(image, dummy_mask)
        
        # Return transformed image (tensor) and also convert it back to PIL for visualization
        # To visualize the input tensor, we need to unnormalize it
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        
        return img_trans.unsqueeze(0).to(self.device), img_trans

    def _extract_roi(self, img_rgb, pred_mask):
        """Bounding box a polip maszkból, kivágás, resize 512x512-re."""
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

    def _analyze_core(self, frame_bgr):
        if not self.models_loaded:
            empty_img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
            empty = (empty_img, "Modellek betöltése...", 0.0, empty_img, 0.0)
            return empty, None

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

        use_vit = (
            self.classifier_backend == "vit" and self.vit_classifier is not None
        )

        # --- ViT: nincs szegmentáció — csak full-frame klasszifikáció ---
        if use_vit:
            t_inf_start = time.perf_counter()
            t_inf_end = t_inf_start  # seg skipped
            t_post_start = time.perf_counter()

            segmented_polyp = orig_img_np.copy()  # jobb panel: teljes frame
            pred_mask = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
            seg_conf = 0.0
            bbox = (0, INPUT_SIZE - 1, 0, INPUT_SIZE - 1)
            roi_batch = None

            t_cls_start = time.perf_counter()
            vit_input_tensor = preprocess_rgb_uint8(orig_img_np)
            cls_result, conf_score, malignant_prob = predict_vit(
                self.vit_classifier, vit_input_tensor, threshold=VIT_THRESHOLD
            )
            t_cls_end = time.perf_counter()
            classify_ms = (t_cls_end - t_cls_start) * 1000
            t_post_end = time.perf_counter()
            t_end = time.perf_counter()

            total_ms = (t_end - t_start) * 1000
            pre_ms = (t_pre_end - t_pre_start) * 1000
            inf_ms = 0.0
            post_ms = (t_post_end - t_post_start) * 1000
            cls_name = self.active_cls_model_name()

            try:
                with open(PERFORMANCE_LOG_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        "none", cls_name,
                        f"{total_ms:.2f}",
                        f"{pre_ms:.2f}",
                        f"{inf_ms:.2f}",
                        f"{post_ms:.2f}",
                        f"{classify_ms:.2f}"
                    ])
                print(
                    f"Logged Performance: Total={total_ms:.2f}ms "
                    f"(Pre={pre_ms:.2f}, Seg=skip, Cls={classify_ms:.2f} [{cls_name}])"
                )
            except Exception as e:
                print(f"Error logging performance: {e}")

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

        # --- ResNet / egyéb: szegmentáció + ROI klasszifikáció ---
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
        vit_input_tensor = None

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

        total_ms = (t_end - t_start) * 1000
        pre_ms = (t_pre_end - t_pre_start) * 1000
        inf_ms = (t_inf_end - t_inf_start) * 1000
        post_ms = (t_post_end - t_post_start) * 1000
        cls_name = self.active_cls_model_name()

        try:
            with open(PERFORMANCE_LOG_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    self.current_model_type, cls_name,
                    f"{total_ms:.2f}",
                    f"{pre_ms:.2f}",
                    f"{inf_ms:.2f}",
                    f"{post_ms:.2f}",
                    f"{classify_ms:.2f}"
                ])
            print(f"Logged Performance: Total={total_ms:.2f}ms (Pre={pre_ms:.2f}, Seg={inf_ms:.2f}, Post={post_ms:.2f}, Cls={classify_ms:.2f} [{cls_name}])")
        except Exception as e:
            print(f"Error logging performance: {e}")

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

    def _classify_roi(self, roi):
        """Return label, confidence, raw malignant prob, and ROI batch for Grad-CAM."""
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

    def classify(self, roi):
        """Klasszifikáció: egyetlen sigmoid kimenet, threshold 0.5."""
        label, conf, _prob, _batch = self._classify_roi(roi)
        return label, conf

    def compute_heatmap(self, ctx: ExplainContext | None, mode: str):
        """Grad-CAM / ViT attention overlay; returns (display_rgb, explain_ms) or (None, 0)."""
        if mode == "off" or ctx is None or ctx.bbox is None:
            return None, 0.0

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
            import traceback
            traceback.print_exc()
            return None, 0.0

# Kijelzett osztálynevek — JNET nélkül. A modell belső címkéje ettől független marad.
_CLASS_COLORS = {
    "non": ("#1F7A4D", "#E5F4EC"),
    "neo": ("#C46A1A", "#F8EDE3"),
    "neutral": ("#5C6570", "#EEF1F4"),
}


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
    if not raw or raw == "—":
        return "—", "neutral"
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

    A méret a Tk képernyő-koordinátájában készül. A Win32 munkaterület csak akkor
    számít, ha ugyanabban a pixelterben van — különben a DPI-skála kétszer számítana.
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
    toolbar_h = 52
    pad = 10
    title_h = 26
    text_h = 92
    footer_h = 22
    gap = 8
    inner_w = max(320, win_w - 2 * pad)
    inner_h = max(240, win_h - toolbar_h - 2 * pad)
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


# --- 3. GUI Application (CustomTkinter – modern kinézet, tkinter háttér) ---
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

        # Dinamikus képekhez referencia (GC ellen)
        self._live_image_ref = None
        self._snap_image_ref = None
        self._seg_image_ref = None
        # Csúszka: ne vessünk össze a programatikus frissítéssel / user húzással
        self._slider_updating = False
        self._slider_user_until = 0.0

        # Initialize Logic
        self.analyzer = MedicalAnalyzer(DEVICE)
        self.video_thread = None
        self._last_result = None
        self._is_analyzing_live = False # Flag a live analysishoz
        self._explain_generation = 0
        # Heatmap külön szálon, sorosan — soha ne blokkolja az élőképet / phase-1-et
        self._explain_lock = threading.Lock()
        # Cache: base maszk + mindkét Grad-CAM overlay (váltás újrafuttatás nélkül)
        self._overlay_cache = {
            "base": None,
            "classification": None,
            "segmentation": None,
        }
        self._overlay_cache_gen = -1
        # Utolsó megjelenített RGB (átméretezéskor újrarajzolás)
        self._last_snap_rgb = None
        self._last_seg_rgb = None

        # SQLite: trigger-eseményenként audit-log (maszk BLOB + metaadatok)
        try:
            self._db_conn = _init_predictions_db()
        except sqlite3.Error as e:
            print(f"SQLite init error: {e}")
            self._db_conn = None
        self._db_lock = threading.Lock()

        self.main_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.top_panel = ctk.CTkFrame(self.main_frame, height=52, corner_radius=0)
        self.top_panel.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.top_panel.grid_propagate(False)

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

        self.lbl_status = ctk.CTkLabel(
            self.top_panel,
            text="Rendszer inicializálása...",
            font=ctk.CTkFont(size=12),
            anchor="e",
            width=120,
        )
        self.lbl_status.pack(side=tk.RIGHT, padx=(4, 8))

        self.lbl_cam = ctk.CTkLabel(self.top_panel, text="Kamera")
        self.lbl_cam.pack(side=tk.LEFT, padx=(8, 2))

        self.combo_cam = ctk.CTkComboBox(
            self.top_panel, values=self.camera_list, width=100, state="readonly"
        )
        self.combo_cam.set(default_cam)
        self.combo_cam.pack(side=tk.LEFT, padx=2)

        self.btn_hdmi = ctk.CTkButton(
            self.top_panel, text="Élő indítás",
            command=lambda: self.start_source(sim=False),
            fg_color=("#2980b9", "#2980b9"), width=96, height=28,
        )
        self.btn_hdmi.pack(side=tk.LEFT, padx=3, pady=8)

        self.btn_sim = ctk.CTkButton(
            self.top_panel, text="Szimuláció",
            command=lambda: self.start_source(sim=True),
            fg_color=("#27ae60", "#27ae60"), width=88, height=28,
        )
        self.btn_sim.pack(side=tk.LEFT, padx=3, pady=8)

        ctk.CTkLabel(self.top_panel, text="Szegmentáló").pack(side=tk.LEFT, padx=(6, 2))
        self._seg_labels = {
            "DeepLabV3+": "deeplab",
            "U-Net": "unet",
        }
        self.seg_var = tk.StringVar(value="DeepLabV3+")
        self.seg_menu = ctk.CTkOptionMenu(
            self.top_panel,
            values=list(self._seg_labels.keys()),
            variable=self.seg_var,
            command=self._on_seg_model_change,
            width=112,
            height=28,
        )
        self.seg_menu.pack(side=tk.LEFT, padx=(0, 4), pady=8)

        ctk.CTkLabel(self.top_panel, text="Modell").pack(side=tk.LEFT, padx=(4, 2))
        self._clf_labels = {
            "ViT-B/16": "vit",
            "ResNet50V2": "keras",
        }
        self.clf_var = tk.StringVar(value="ViT-B/16")
        self.clf_menu = ctk.CTkOptionMenu(
            self.top_panel,
            values=list(self._clf_labels.keys()),
            variable=self.clf_var,
            command=self._on_classifier_change,
            width=112,
            height=28,
        )
        self.clf_menu.pack(side=tk.LEFT, padx=(0, 4), pady=8)

        ctk.CTkLabel(self.top_panel, text="Magyarázat").pack(side=tk.LEFT, padx=(4, 2))
        self._explain_labels = {
            "Klasszifikáció": "classification",
            "Szegmentálás": "segmentation",
            "Ki": "off",
        }
        self.explain_var = tk.StringVar(value="Klasszifikáció")
        self._explain_last_mode = "classification"
        self.explain_menu = ctk.CTkOptionMenu(
            self.top_panel,
            values=list(self._explain_labels.keys()),
            variable=self.explain_var,
            command=self._on_explain_mode_change,
            width=108,
            height=28,
        )
        self.explain_menu.pack(side=tk.LEFT, padx=(0, 4), pady=8)
        self.analyzer.explain_mode = "classification"

        self.check_live_analysis_var = tk.BooleanVar(value=False)
        self.check_live_analysis = ctk.CTkCheckBox(
            self.top_panel, text="Élő anal.", variable=self.check_live_analysis_var
        )
        self.check_live_analysis.pack(side=tk.LEFT, padx=4)

        self.btn_analyze = ctk.CTkButton(
            self.top_panel, text="Elemzés (Space)", command=self.capture_and_analyze,
            font=ctk.CTkFont(size=13, weight="bold"), fg_color=("#e74c3c", "#c0392b"),
            width=108, height=28,
        )
        self.btn_analyze.pack(side=tk.LEFT, padx=3, pady=8)

        self.save_mode_var = tk.StringVar(value="manual")
        self.save_mode_var.trace_add("write", lambda *_: self._update_save_button_state())
        self._save_mode_labels = {"Kézi": "manual", "Auto": "auto"}
        self.save_mode_menu = ctk.CTkOptionMenu(
            self.top_panel,
            values=list(self._save_mode_labels.keys()),
            command=lambda choice: self.save_mode_var.set(self._save_mode_labels.get(choice, "manual")),
            width=78,
            height=28,
        )
        self.save_mode_menu.set("Kézi")
        self.save_mode_menu.pack(side=tk.LEFT, padx=(4, 2), pady=8)
        self.btn_save = ctk.CTkButton(
            self.top_panel, text="Mentés", command=self._manual_save,
            fg_color=("#27ae60", "#27ae60"), width=72, height=28,
        )
        self.btn_save.pack(side=tk.LEFT, padx=3, pady=8)

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
        self.live_panel = ctk.CTkLabel(self._live_stack, text="Várakozás")
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
        self.snapshot_panel = ctk.CTkLabel(self._snapshot_inner, text="Nincs kép")
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
        self.result_panel = ctk.CTkLabel(self._result_inner, text="Eredmény")
        self.result_panel.grid(row=0, column=0, sticky="nsew")

        self._readout = ctk.CTkFrame(self.result_panel_frame, fg_color="transparent")
        self._readout.pack(anchor="center", padx=8, pady=(2, 8))
        cap_font = ctk.CTkFont(size=13)
        self.lbl_class_caption = ctk.CTkLabel(self._readout, text="Osztály", font=cap_font)
        self.lbl_class_caption.grid(row=0, column=0, sticky="w", padx=(4, 16))
        self.lbl_class_name = ctk.CTkLabel(
            self._readout,
            text="—",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=_CLASS_COLORS["neutral"][0],
            fg_color=_CLASS_COLORS["neutral"][1],
            corner_radius=6,
            padx=10,
            pady=2,
        )
        self.lbl_class_name.grid(row=1, column=0, sticky="w", padx=(4, 16), pady=(0, 4))
        self.lbl_conf_caption = ctk.CTkLabel(self._readout, text="Konfidencia", font=cap_font)
        self.lbl_conf_caption.grid(row=0, column=1, sticky="w", padx=(8, 16))
        self.lbl_conf_value = ctk.CTkLabel(
            self._readout,
            text="—",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=_CLASS_COLORS["neutral"][0],
        )
        self.lbl_conf_value.grid(row=1, column=1, sticky="w", padx=(8, 16))
        self._seg_readout = ctk.CTkFrame(self._readout, fg_color="transparent")
        self._seg_readout.grid(row=0, column=2, rowspan=2, sticky="nsw", padx=(8, 4))
        ctk.CTkLabel(self._seg_readout, text="Szegmentáció", font=cap_font).pack(anchor="w")
        self.lbl_seg_value = ctk.CTkLabel(
            self._seg_readout,
            text="—",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        self.lbl_seg_value.pack(anchor="w")
        self._seg_readout.grid_remove()

        self.main_frame.columnconfigure(0, weight=0)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=0)
        self.main_frame.rowconfigure(1, weight=1)

        self.root.bind('<space>', lambda e: self.capture_and_analyze())
        self.root.bind('<m>', self._toggle_explain_overlay)
        self.root.bind('<M>', self._toggle_explain_overlay)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._update_save_button_state()
        self.root.after(100, self._start_async_loading)
        self.update_live_feed()

    def _start_async_loading(self):
        """Modellek betöltésének indítása külön szálon."""
        self._set_ui_state("disabled")
        threading.Thread(target=self.analyzer.load_models, args=(self._update_loading_status,), daemon=True).start()

    def _update_loading_status(self, text):
        """Státusz frissítése a GUI szálon."""
        self.root.after(0, lambda t=text: self.lbl_status.configure(text=t))
        if (
            (isinstance(text, str) and text.startswith("Rendszer kész."))
            or (isinstance(text, str) and text.startswith("HIBA:"))
        ):
            self.root.after(500, lambda: self._set_ui_state("normal"))
            # Klasszifikátor menü frissítése betöltés után
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
        """Kép mentése (cv2.imwrite unicode útvonal hibájának megkerülése Windows-on)."""
        ok, buf = cv2.imencode(".png", img_bgr)
        if ok:
            with open(path, "wb") as f:
                f.write(buf.tobytes())

    def _save_results(self, frame, trans_img_vis, segmented_polyp, cls_res, conf, seg_conf=0.0):
        """Trigger-esemény rögzítése: PNG képek a mappába + SQLite sor (maszk BLOB + metaadatok)."""
        os.makedirs(OUTPUT_SAVE_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = os.path.join(OUTPUT_SAVE_DIR, timestamp)
        input_path = f"{prefix}_input.png"
        mask_path = f"{prefix}_masked.png"
        input_bgr = cv2.cvtColor(trans_img_vis, cv2.COLOR_RGB2BGR)
        mask_bgr = cv2.cvtColor(segmented_polyp, cv2.COLOR_RGB2BGR)
        self._save_image(input_path, input_bgr)
        self._save_image(mask_path, mask_bgr)

        # Maszk bináris tömörített reprezentációja (PNG bytes) BLOB-ként a DB-be
        ok, buf = cv2.imencode(".png", mask_bgr)
        mask_blob = buf.tobytes() if ok else None

        seg_model = getattr(self.analyzer, "current_model_type", "deeplab")
        cls_model = "ResNet50V2"

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
        """Manuális mentés: utolsó eredmény mentése."""
        if self._last_result is None:
            self.lbl_status.configure(text="Nincs menthető eredmény")
            return
        self._save_results(*self._last_result)

    def _update_save_button_state(self):
        """Auto módban Mentés gomb letiltása."""
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

    def _on_classifier_change(self, choice):
        """Modell legördülő: ViT → csak full-frame cls (nincs szeg); ResNet → szeg+ROI."""
        backend = self._clf_labels.get(choice, "vit")
        applied = self.analyzer.switch_classifier_backend(backend)
        inv = {v: k for k, v in self._clf_labels.items()}
        self.clf_var.set(inv.get(applied, choice))
        self._sync_seg_menu_for_classifier()
        self._reset_result_panel_for_classifier()
        if applied == "vit":
            self.lbl_status.configure(
                text="Modell: ViT-B/16 — Space: full-frame klasszifikáció (szegmentálás nélkül)"
            )
        else:
            self.lbl_status.configure(
                text="Modell: ResNet50V2 — Space: szegmentálás + ROI klasszifikáció"
            )

    def _reset_result_panel_for_classifier(self):
        """Modellváltáskor: régi heatmap/cache érvénytelen, panel cím + placeholder."""
        self._next_explain_generation()
        self._overlay_cache = {
            "base": None,
            "classification": None,
            "segmentation": None,
        }
        self._overlay_cache_gen = -1
        self._last_seg_rgb = None
        self._seg_image_ref = None
        if self.analyzer.classifier_backend == "vit":
            title = "Eredmény (teljes kép)"
            placeholder = "Space → teljes kép"
        else:
            title = "Eredmény"
            placeholder = "Space → szegmentálás + ROI"
        if hasattr(self.result_panel_frame, "_title_label"):
            self.result_panel_frame._title_label.configure(text=title)
        self.result_panel.configure(image=None, text=placeholder)
        self._apply_readout(None)

    def _sync_seg_menu_for_classifier(self):
        """ViT módban a szegmentáló menü nem releváns — tiltva."""
        if self.analyzer.classifier_backend == "vit" and self.analyzer.vit_classifier is not None:
            self.seg_menu.configure(state="disabled")
        else:
            self.seg_menu.configure(state="normal")

    def _refresh_classifier_menu(self):
        """Betöltés után: csak a ténylegesen betöltött klasszifikátorok a Modell listában."""
        values = []
        if self.analyzer.vit_classifier is not None:
            values.append("ViT-B/16")
        if self.analyzer.classifier is not None:
            values.append("ResNet50V2")
        if not values:
            self.clf_menu.configure(values=["—"], state="disabled")
            self.clf_var.set("—")
            self.seg_menu.configure(state="normal")
            return
        self.clf_menu.configure(values=values, state="normal")
        inv = {v: k for k, v in self._clf_labels.items()}
        default = inv.get(self.analyzer.classifier_backend, values[0])
        if default not in values:
            default = values[0]
            self.analyzer.switch_classifier_backend(self._clf_labels[default])
        self.clf_var.set(default)
        self._sync_seg_menu_for_classifier()
        self._reset_result_panel_for_classifier()

    def _next_explain_generation(self):
        self._explain_generation += 1
        return self._explain_generation

    def _apply_readout(self, cls_res, conf=None, seg_conf=None):
        """Osztály, konfidencia és (ResNet úton) szegmentáció. JNET nincs a feliraton."""
        if cls_res is None:
            name, kind = "—", "neutral"
            conf_txt = "—"
        else:
            name, kind = display_class_name(cls_res)
            conf_txt = "—" if conf is None else f"{conf * 100:.1f}%"
        fg, bg = _CLASS_COLORS[kind]
        self.lbl_class_name.configure(text=name, text_color=fg, fg_color=bg)
        self.lbl_conf_value.configure(text=conf_txt, text_color=fg)
        if self.analyzer.classifier_backend == "vit":
            self._seg_readout.grid_remove()
            return
        self._seg_readout.grid()
        seg_txt = "—" if seg_conf is None else f"{seg_conf * 100:.1f}%"
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

    def _update_seg_panel_image(self, segmented_rgb):
        self._last_seg_rgb = segmented_rgb
        sz = self._result_img_side
        fitted = self._fit_rgb_square(segmented_rgb, sz)
        seg_img = Image.fromarray(fitted)
        self._seg_image_ref = ctk.CTkImage(
            light_image=seg_img, dark_image=seg_img, size=(sz, sz)
        )
        self.result_panel.configure(image=self._seg_image_ref, text="")

    def _set_snap_panel_image(self, snap_rgb):
        self._last_snap_rgb = snap_rgb
        sz = self._snap_img_side
        fitted = self._fit_rgb_square(snap_rgb, sz)
        snap_img = Image.fromarray(fitted)
        self._snap_image_ref = ctk.CTkImage(
            light_image=snap_img, dark_image=snap_img, size=(sz, sz)
        )
        self.snapshot_panel.configure(image=self._snap_image_ref, text="")

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
                # Ne duplázza a heatmap sort
                base_status = status.split(" | Heatmap")[0]
                self.lbl_status.configure(
                    text=f"{base_status} | Heatmap ({mode[:3]}) +{explain_ms:.0f}ms"
                )

    def _schedule_heatmap(self, ctx, gen):
        """Mindkét Grad-CAM elkészül; a bejelölt mód előbb, a másik utána. Váltás cache-ből."""
        if ctx is None or ctx.bbox is None:
            return

        # Base maszk azonnal a cache-ben (Ki módhoz)
        self._overlay_cache = {
            "base": ctx.segmented_polyp.copy(),
            "classification": None,
            "segmentation": None,
        }
        self._overlay_cache_gen = gen

        preferred = self.analyzer.explain_mode
        # ViT: nincs szegmentáció → csak klasszifikációs attention
        if self.analyzer.classifier_backend == "vit":
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
                # Tenzorok felszabadítása mindkét map után
                try:
                    ctx.input_tensor = None
                    ctx.roi_batch = None
                    ctx.vit_input_tensor = None
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True, name="heatmap-worker").start()

    def start_source(self, sim=False):
        if self.video_thread:
            self.video_thread.stop()
        
        if sim:
            self.video_thread = VideoCaptureThread(
                simulation_mode=True, image_folder=os.path.join(_BASE_DIR, "images")
            )
            self.lbl_status.configure(text="Started Simulation Mode (Folder: images)")
            self._reset_video_slider_for_non_file()
        else:
            # Parse selected camera index or file
            selection = self.combo_cam.get()
            try:
                # Check if it is a video file
                if selection.startswith("Video: "):
                    filename = selection.replace("Video: ", "")
                    vpath = filename if os.path.isabs(filename) else os.path.join(_BASE_DIR, filename)
                    if os.path.isfile(vpath):
                        self.video_thread = VideoCaptureThread(src=vpath, simulation_mode=False)
                        self.lbl_status.configure(text=f"Started Video File: {os.path.basename(vpath)}")
                        self._show_video_slider()
                        self._slider_updating = True
                        self.video_slider.set(0)
                        self._slider_updating = False
                    else:
                         self.lbl_status.configure(text=f"Error: File not found: {filename}")
                         self._reset_video_slider_for_non_file()
                    return

                # Expecting format "Camera N: ..." or "Camera N"
                # Extract number after "Camera "
                parts = selection.split()
                if len(parts) >= 2 and parts[0] == "Camera":
                     # Remove potential colon from the index part (e.g. "2:")
                     idx_str = parts[1].replace(':', '')
                     idx = int(idx_str)
                else:
                     idx = 0 # Fallback
                
                self.video_thread = VideoCaptureThread(src=idx, simulation_mode=False)
                self.lbl_status.configure(text=f"Started Camera Input (Index {idx})")
                self._reset_video_slider_for_non_file()
            except Exception as e:
                print(f"Error parsing camera selection: {e}")
                self.lbl_status.configure(text="Error selecting camera. Check console.")
                self._hide_video_slider()

    def _show_video_slider(self):
        """Csúszka sor megjelenítése (videófájl) — grid 1. sor, ugyanolyan széles mint a videó."""
        if self._video_slider_packed:
            return
        self._video_slider_packed = True
        self.video_slider.configure(state="normal")
        self.video_slider.grid(row=1, column=0, sticky="ew", padx=2, pady=(2, 0))

    def _hide_video_slider(self):
        """Csúszka elrejtése (kamera / szimuláció / nincs videó)."""
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
        """Kamera / szimuláció: csúszka rejtve."""
        self._hide_video_slider()

    def on_slider_move(self, val):
        if self._slider_updating:
            return
        self._slider_user_until = time.time() + 0.5
        if self.video_thread and self.video_thread.video_file_mode:
            self.video_thread.set_position(float(val))

    def update_live_feed(self):
        # UI update rate can be independent of video FPS, e.g. 30 FPS (33ms)
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
                    self.live_panel.configure(image=self._live_image_ref, text="")

                    # Live Analysis: csak szeg+klassz — heatmap NINCS (CPU/memória, élőkép)
                    if self.check_live_analysis_var.get() and self.analyzer.models_loaded:
                        if not getattr(self, '_is_analyzing_live', False):
                            self._is_analyzing_live = True
                            threading.Thread(
                                target=self._async_live_analyze,
                                args=(frame, display_w, display_h),
                                daemon=True,
                            ).start()

                # Videófájl: csúszka követi a lejátszást (user húzás közben ~0.5s szünet)
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
            # Kritikus: exception NE törje el az after() láncot → bal panel ne fagyjon be
            print(f"Live feed update error: {e}")
        finally:
            try:
                self.root.after(delay, self.update_live_feed)
            except Exception:
                pass

    def _async_live_analyze(self, frame, display_w, display_h):
        # Élő mód: csak gyors predikció — Grad-CAM szándékosan kihagyva
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
        
        # 1. Update Network Input Panel — egységes méret a többi panelhoz
        self._set_snap_panel_image(trans_img_vis)
        
        # 2. Update Segmentation / ViT Panel (phase 1: mask or full frame)
        self._update_seg_panel_image(segmented_polyp)

        if self.analyzer.classifier_backend == "vit":
            if hasattr(self.result_panel_frame, "_title_label"):
                self.result_panel_frame._title_label.configure(text="Eredmény (teljes kép)")
            self._apply_readout(cls_res, conf, None)
        else:
            if hasattr(self.result_panel_frame, "_title_label"):
                self.result_panel_frame._title_label.configure(text="Eredmény")
            self._apply_readout(cls_res, conf, seg_conf)

    def capture_and_analyze(self):
        if not self.video_thread:
             self.lbl_status.configure(text="Please select a source first!")
             return

        self.lbl_status.configure(text="Kép rögzítése...")
        self.btn_analyze.configure(state="disabled")
        self.root.update_idletasks()

        ret, frame = self.video_thread.read()
        if not ret or frame is None:
            self.lbl_status.configure(text="Error: No video frame available.")
            self.btn_analyze.configure(state="normal")
            return

        preview_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._set_snap_panel_image(preview_rgb)
        self._update_seg_panel_image(preview_rgb)
        self._apply_readout(None)
        self.lbl_class_name.configure(text="Elemzés...")
        self._next_explain_generation()
        self.lbl_status.configure(text="Kép elkészült, elemzés...")
        self.root.update_idletasks()

        h, w = frame.shape[:2]
        display_h = self._snap_img_side
        scale = display_h / h
        display_w = int(w * scale)
        
        # Külön szálon futtatjuk a hálózatot, hogy ne fagyjon a GUI
        threading.Thread(target=self._async_analyze, args=(frame, display_w, display_h), daemon=True).start()

    def _async_analyze(self, frame, display_w, display_h):
        gen = self._next_explain_generation()
        start_time = time.time()
        try:
            results, ctx = self.analyzer.analyze_with_context(frame)
            elapsed = time.time() - start_time
            # Phase 1 azonnal — gomb / élőkép nem vár heatmapre
            self.root.after(
                0,
                lambda: self._apply_analysis_result(frame, display_w, display_h, results, elapsed),
            )
            # Phase 2: Grad-CAM külön szálon (nem ezen a workeren)
            self._schedule_heatmap(ctx, gen)
        except Exception as e:
            print(f"Elemzés hiba: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: (
                self.lbl_status.configure(text=f"Hiba: {e}"),
                self.btn_analyze.configure(state="normal"),
            ))

    def _apply_analysis_result(self, frame, display_w, display_h, results, elapsed):
        segmented_polyp, cls_res, conf, trans_img_vis, seg_conf = results
        # Store for manual save
        self._last_result = (frame, trans_img_vis, segmented_polyp, cls_res, conf, seg_conf)
        
        # Update UI
        self.run_analysis_on_frame(frame, display_w, display_h, results=results)
        
        # Auto save if enabled
        if self.save_mode_var.get() == "auto":
            self._save_results(*self._last_result)
        
        shown, _kind = display_class_name(cls_res)
        self.lbl_status.configure(text=f"{shown} ({conf*100:.1f}%) | {elapsed:.3f} s")
        self.btn_analyze.configure(state="normal")

    def on_close(self):
        print("Closing application...")
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
    """GUI indítás (bootstrap / közvetlen futtatás is ezt hívja)."""
    import multiprocessing

    multiprocessing.freeze_support()

    def _crash_log_path():
        base = os.path.dirname(os.path.abspath(sys.executable)) if getattr(sys, "frozen", False) else _BASE_DIR
        return os.path.join(base, "LiveCapture_crash.log")

    try:
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")
        root = ctk.CTk()
        app = App(root)
        root.mainloop()
    except Exception:
        tb = traceback.format_exc()
        try:
            with open(_crash_log_path(), "w", encoding="utf-8") as f:
                f.write(tb)
        except OSError:
            pass
        try:
            import tkinter.messagebox as _mb

            _mb.showerror(
                "LiveCapture hiba",
                f"Indítási hiba. Részletek:\n{_crash_log_path()}",
            )
        except Exception:
            pass
        raise SystemExit(1) from None


if __name__ == "__main__":
    run_app()
