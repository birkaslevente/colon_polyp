import cv2
import threading
import time
import csv
import json
import sqlite3
import sys
import traceback
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

# --- Configuration ---
def _app_base_dir():
    """Fejlesztésben: script mappa; PyInstaller exe-nél: az .exe mappája (modellek ide másolandók)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


_BASE_DIR = _app_base_dir()
PERFORMANCE_LOG_FILE = os.path.join(_BASE_DIR, "performance_log.csv")

DEEPLAB_CHECKPOINT = os.path.join(_BASE_DIR, "checkpoints", "best_model_epoch_38.pth")
UNET_CHECKPOINT = os.path.join(_BASE_DIR, "checkpoints_unet", "best_model_epoch_22.pth")

# --- classificator_models/ (TensorFlow–Keras, .gitignore alatt is lehet) ---
#   *.keras            : tanított bináris klasszifikátor (ResNet50V2 + fej), betöltés: tf.keras.models.load_model
#   *_metadata.json    : input_shape, backbone — dokumentáció; ha létezik, a bemeneti méret onnan olvasható
#   *_history.json     : epochonkénti loss/acc (tanítás utáni elemzéshez), a futó app NEM használja
CLASSIFIER_MODEL_DIR = os.path.join(_BASE_DIR, "classificator_models")
CLASSIFIER_KERAS_PREFERRED = "cnn_s2_lr2e4_tb_os.keras"
CLASSIFIER_KERAS_LEGACY = "resnet50v2_polyp_20260217_131050.keras"
CLASSIFIER_INPUT_SIZE = 512
CLASSIFIER_CLASS_NAMES = ["Benign (JNET 1)", "Malignant (JNET 2a/2b/3)"]


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
            return self.ret, self.frame.copy()

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
            
            # TensorFlow / classifier opcionális: ha DLL hiba van, az app ettől még fusson.
            self.classifier = None
            self.classifier_keras_path = _resolve_classifier_keras_path()
            self.classifier_input_size = _read_classifier_input_size(
                self.classifier_keras_path, CLASSIFIER_INPUT_SIZE
            )
            if status_callback:
                status_callback("Klasszifikátor betöltése (TensorFlow)...")
            if tf is not None:
                try:
                    tf.config.set_visible_devices([], "GPU")
                except Exception:
                    # CPU-only fallback, ha a GPU tiltás nem támogatott.
                    pass
                if not os.path.isfile(self.classifier_keras_path):
                    raise FileNotFoundError(
                        f"Keras klasszifikátor nem található: {self.classifier_keras_path} "
                        f"(mappa: {CLASSIFIER_MODEL_DIR})"
                    )
                print(
                    f"Classifier: {self.classifier_keras_path} "
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
            else:
                print(f"TensorFlow import hiba: {_TF_IMPORT_ERROR}")

            self.models_loaded = True
            if status_callback:
                if self.classifier is None:
                    status_callback("Rendszer kész. (Klasszifikátor: kikapcsolva, TensorFlow hiba)")
                else:
                    status_callback("Rendszer kész.")
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

    def classify(self, roi):
        """Klasszifikáció: egyetlen sigmoid kimenet, threshold 0.5."""
        if self.classifier is None:
            return "Classifier unavailable", 0.0
        if roi is None:
            return CLASSIFIER_CLASS_NAMES[0], 0.0
        roi_batch = np.expand_dims(roi, 0)
        pred = self.classifier.predict(roi_batch, verbose=0)
        malignant_prob = float(pred[0][0])
        if malignant_prob >= 0.5:
            return CLASSIFIER_CLASS_NAMES[1], malignant_prob
        else:
            return CLASSIFIER_CLASS_NAMES[0], 1.0 - malignant_prob

    def analyze(self, frame_bgr):
        if not self.models_loaded:
            empty_img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
            return empty_img, "Modellek betöltése...", 0.0, empty_img, 0.0

        t_start = time.perf_counter()
        
        # 1. Convert BGR to RGB and PIL
        t_pre_start = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # 2. Inference
        input_tensor, trans_img_tensor = self.preprocess(pil_img)
        t_pre_end = time.perf_counter()
        
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

        # 3. Visualization (Matches colon_short.ipynb logic exactly)
        t_post_start = time.perf_counter()
        
        # Unnormalize input tensor for visualization
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # trans_img_tensor is (C, H, W)
        img_np = trans_img_tensor.cpu().numpy().transpose(1, 2, 0) # (H, W, C)
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        # Convert to uint8 (orig_img_np)
        orig_img_np = (img_np * 255).astype(np.uint8)
        
        # Create masked image (final_img)
        # Background is black, polyp is original image content
        final_img = np.zeros_like(orig_img_np)
        polyp_mask_bool = pred_mask.astype(bool)
        
        # Stack mask to 3 channels
        polyp_mask_rgb = np.stack([polyp_mask_bool] * 3, axis=-1)
        
        final_img[polyp_mask_rgb] = orig_img_np[polyp_mask_rgb]
        
        segmented_polyp = final_img # 513x513
        trans_img_vis = orig_img_np # 513x513

        seg_conf = float(np.mean(probs[polyp_mask_bool])) if np.any(polyp_mask_bool) else 0.0

        # 4. Classification
        rows = np.any(polyp_mask_bool, axis=1)
        cols = np.any(polyp_mask_bool, axis=0)
        
        cls_result = "No Polyp Detected"
        conf_score = 0.0
        classify_ms = 0.0

        if np.any(rows) and np.any(cols):
            # Fehér hátterű kép generálása a klasszifikátornak
            white_bg_img = np.full_like(orig_img_np, 255)
            white_bg_img[polyp_mask_rgb] = orig_img_np[polyp_mask_rgb]
            
            roi = self._extract_roi(white_bg_img, polyp_mask_bool)
            t_cls_start = time.perf_counter()
            cls_result, conf_score = self.classify(roi)
            t_cls_end = time.perf_counter()
            classify_ms = (t_cls_end - t_cls_start) * 1000
            
        t_post_end = time.perf_counter()
        t_end = time.perf_counter()
        
        # Calculate durations in ms
        total_ms = (t_end - t_start) * 1000
        pre_ms = (t_pre_end - t_pre_start) * 1000
        inf_ms = (t_inf_end - t_inf_start) * 1000
        post_ms = (t_post_end - t_post_start) * 1000
        
        # Log to CSV
        try:
            with open(PERFORMANCE_LOG_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"), 
                    self.current_model_type, "ResNet50V2",
                    f"{total_ms:.2f}", 
                    f"{pre_ms:.2f}", 
                    f"{inf_ms:.2f}", 
                    f"{post_ms:.2f}",
                    f"{classify_ms:.2f}"
                ])
            print(f"Logged Performance: Total={total_ms:.2f}ms (Pre={pre_ms:.2f}, Seg={inf_ms:.2f}, Post={post_ms:.2f}, Cls={classify_ms:.2f})")
        except Exception as e:
            print(f"Error logging performance: {e}")

        return segmented_polyp, cls_result, conf_score, trans_img_vis, seg_conf

# --- 3. GUI Application (CustomTkinter – modern kinézet, tkinter háttér) ---
class App:
    def __init__(self, root, window_title="HDMI Live Polyp Segmentation"):
        self.root = root
        self.root.title(window_title)
        self.root.geometry("1600x600")
        self.root.minsize(1200, 520)

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

        # SQLite: trigger-eseményenként audit-log (maszk BLOB + metaadatok)
        try:
            self._db_conn = _init_predictions_db()
        except sqlite3.Error as e:
            print(f"SQLite init error: {e}")
            self._db_conn = None
        self._db_lock = threading.Lock()

        self.main_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.top_panel = ctk.CTkFrame(self.main_frame, height=44, corner_radius=0)
        self.top_panel.grid(row=0, column=0, columnspan=3, sticky="ew")

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

        self.lbl_cam = ctk.CTkLabel(self.top_panel, text="Camera:")
        self.lbl_cam.pack(side=tk.LEFT, padx=5)

        self.combo_cam = ctk.CTkComboBox(
            self.top_panel, values=self.camera_list, width=280, state="readonly"
        )
        self.combo_cam.set(default_cam)
        self.combo_cam.pack(side=tk.LEFT, padx=5)

        self.btn_hdmi = ctk.CTkButton(
            self.top_panel, text="Start Live Input",
            command=lambda: self.start_source(sim=False),
            fg_color=("#2980b9", "#2980b9"), width=140,
        )
        self.btn_hdmi.pack(side=tk.LEFT, padx=10, pady=5)

        self.btn_sim = ctk.CTkButton(
            self.top_panel, text="Simulation (Images)",
            command=lambda: self.start_source(sim=True),
            fg_color=("#27ae60", "#27ae60"), width=160,
        )
        self.btn_sim.pack(side=tk.LEFT, padx=10, pady=5)

        self.model_var = tk.StringVar(value="deeplab")
        self.btn_model = ctk.CTkButton(
            self.top_panel, text="Model: DeepLabV3+", command=self.toggle_model,
            fg_color=("#8e44ad", "#8e44ad"), width=200,
        )
        self.btn_model.pack(side=tk.LEFT, padx=10, pady=5)

        self.controls_panel = ctk.CTkFrame(self.top_panel, fg_color="transparent")
        self.controls_panel.pack(side=tk.LEFT, padx=20, fill=tk.X, expand=True)

        self.check_live_analysis_var = tk.BooleanVar(value=False)
        self.check_live_analysis = ctk.CTkCheckBox(
            self.controls_panel, text="Live Analysis", variable=self.check_live_analysis_var
        )
        self.check_live_analysis.pack(side=tk.TOP, anchor="w", pady=(0, 4))

        def _panel_frame(parent, title):
            f = ctk.CTkFrame(parent, corner_radius=8)
            ctk.CTkLabel(f, text=title, font=ctk.CTkFont(size=12, weight="bold")).pack(
                anchor="w", padx=6, pady=(4, 2)
            )
            return f

        self.live_panel_frame = _panel_frame(self.main_frame, "Live Feed")
        self.live_panel_frame.grid(row=1, column=0, padx=4, pady=2, sticky="new")
        self._preview_sz = 360
        self._footer_row_h = 28
        self._live_video_column = ctk.CTkFrame(self.live_panel_frame, fg_color="transparent")
        self._live_video_column.pack(anchor="n", fill=tk.NONE, expand=False)
        sz, fh = self._preview_sz, self._footer_row_h
        self._live_stack = ctk.CTkFrame(self._live_video_column, fg_color="transparent", width=sz, height=sz)
        self._live_stack.pack(anchor="n", padx=2, pady=(0, 2))
        self._live_stack.pack_propagate(False)
        self._live_stack.grid_columnconfigure(0, weight=1)
        self._live_stack.grid_rowconfigure(0, weight=0, minsize=sz - fh)
        self._live_stack.grid_rowconfigure(1, weight=0, minsize=fh)
        self.live_panel = ctk.CTkLabel(self._live_stack, text="Waiting...")
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

        self.snapshot_panel_frame = _panel_frame(self.main_frame, "Network Input (Center Crop)")
        self.snapshot_panel_frame.grid(row=1, column=1, padx=4, pady=2, sticky="new")
        self._snapshot_inner = ctk.CTkFrame(self.snapshot_panel_frame, fg_color="transparent", width=sz, height=sz)
        self._snapshot_inner.pack(anchor="n", padx=2, pady=(0, 2))
        self._snapshot_inner.pack_propagate(False)
        self._snapshot_inner.grid_columnconfigure(0, weight=1)
        self._snapshot_inner.grid_rowconfigure(0, weight=0, minsize=sz - fh)
        self._snapshot_inner.grid_rowconfigure(1, weight=0, minsize=fh)
        self.snapshot_panel = ctk.CTkLabel(self._snapshot_inner, text="No Capture")
        self.snapshot_panel.grid(row=0, column=0, sticky="nsew")

        self.result_panel_frame = _panel_frame(self.main_frame, "Polyp Segmentation")
        self.result_panel_frame.grid(row=1, column=2, padx=4, pady=2, sticky="new")
        self._result_inner = ctk.CTkFrame(self.result_panel_frame, fg_color="transparent", width=sz, height=sz)
        self._result_inner.pack(anchor="n", padx=2, pady=(0, 2))
        self._result_inner.pack_propagate(False)
        self._result_inner.grid_columnconfigure(0, weight=1)
        self._result_inner.grid_rowconfigure(0, weight=0, minsize=sz - fh)
        self._result_inner.grid_rowconfigure(1, weight=0, minsize=fh)
        self.result_panel = ctk.CTkLabel(self._result_inner, text="Result")
        self.result_panel.grid(row=0, column=0, sticky="nsew")
        self.lbl_result = ctk.CTkLabel(self._result_inner, text="—", font=ctk.CTkFont(size=12, weight="bold"))
        self.lbl_result.grid(row=1, column=0, sticky="w", padx=4, pady=2)

        # Alsó vezérlőblokk a grid 3. sorában — így nincs üres rés a panelek és a gombok között
        self.controls_frame = ctk.CTkFrame(self.main_frame, corner_radius=0, height=88)
        self.controls_frame.grid(row=2, column=0, columnspan=3, padx=4, pady=(2, 4), sticky="ew")
        self.controls_frame.grid_propagate(False)

        self.btn_analyze = ctk.CTkButton(
            self.controls_frame, text="Capture & Analyze (Space)", command=self.capture_and_analyze,
            font=ctk.CTkFont(size=15, weight="bold"), fg_color=("#e74c3c", "#c0392b"), height=40,
        )
        self.btn_analyze.pack(pady=4)

        save_frame = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        save_frame.pack(pady=2)
        self.save_mode_var = tk.StringVar(value="manual")
        self.save_mode_var.trace_add("write", lambda *_: self._update_save_button_state())
        ctk.CTkRadioButton(
            save_frame, text="Manuális mentés", variable=self.save_mode_var, value="manual"
        ).pack(side=tk.LEFT, padx=10)
        ctk.CTkRadioButton(
            save_frame, text="Auto mentés", variable=self.save_mode_var, value="auto"
        ).pack(side=tk.LEFT, padx=10)
        self.btn_save = ctk.CTkButton(
            save_frame, text="Mentés", command=self._manual_save,
            fg_color=("#27ae60", "#27ae60"), width=100,
        )
        self.btn_save.pack(side=tk.LEFT, padx=10)

        self.lbl_status = ctk.CTkLabel(
            self.controls_frame, text="Rendszer inicializálása...", font=ctk.CTkFont(size=12)
        )
        self.lbl_status.pack(side=tk.BOTTOM, pady=2)

        # Grid: sorok nem expandálnak — tartalom magassága, nincs felesleges rés
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.columnconfigure(2, weight=1)
        self.main_frame.rowconfigure(0, weight=0)
        self.main_frame.rowconfigure(1, weight=0)
        self.main_frame.rowconfigure(2, weight=0)

        self.root.bind('<space>', lambda e: self.capture_and_analyze())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.main_frame.bind('<Configure>', lambda e: self.root.after(50, self._apply_preview_size))

        self._update_save_button_state()

        # Start Async Loading
        self.root.after(100, self._start_async_loading)
        # Méretezés megnyitáskor — adaptív az oszlopszélességhez
        self.root.after(150, self._apply_preview_size)

        # Start Loop
        self.update_live_feed()

    def _apply_preview_size(self):
        """Megnyitáskor / átméretezés: oszlopszélesség alapján adaptív _preview_sz."""
        if not hasattr(self, "_snapshot_inner"):
            return
        self.root.update_idletasks()
        try:
            w = self.main_frame.winfo_width()
            if w > 100:
                col = max(280, (w - 40) // 3)
                sz = min(420, col - 24)
                sz = max(280, sz)
                if sz != self._preview_sz:
                    self._preview_sz = sz
                    fh = self._footer_row_h
                    self._live_stack.configure(width=sz, height=sz)
                    self._live_stack.grid_rowconfigure(0, minsize=sz - fh)
                    self._snapshot_inner.configure(width=sz, height=sz)
                    self._snapshot_inner.grid_rowconfigure(0, minsize=sz - fh)
                    self._result_inner.configure(width=sz, height=sz)
                    self._result_inner.grid_rowconfigure(0, minsize=sz - fh)
        except tk.TclError:
            pass

    def _start_async_loading(self):
        """Modellek betöltésének indítása külön szálon."""
        self._set_ui_state("disabled")
        threading.Thread(target=self.analyzer.load_models, args=(self._update_loading_status,), daemon=True).start()

    def _update_loading_status(self, text):
        """Státusz frissítése a GUI szálon."""
        self.root.after(0, lambda t=text: self.lbl_status.configure(text=t))
        if text == "Rendszer kész." or (isinstance(text, str) and text.startswith("HIBA:")):
            self.root.after(500, lambda: self._set_ui_state("normal"))

    def _set_ui_state(self, state):
        """Gombok tiltása/engedélyezése (CustomTkinter: 'normal' / 'disabled')."""
        self.btn_hdmi.configure(state=state)
        self.btn_sim.configure(state=state)
        self.btn_model.configure(state=state)
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

    def toggle_model(self):
        if self.model_var.get() == "deeplab":
            self.model_var.set("unet")
            self.analyzer.switch_model("unet")
            self.btn_model.configure(text="Model: U-Net", fg_color=("#d35400", "#d35400"))
            self.lbl_status.configure(text="Aktív modell: U-Net (ResNet34, epoch 22)")
        else:
            self.model_var.set("deeplab")
            self.analyzer.switch_model("deeplab")
            self.btn_model.configure(text="Model: DeepLabV3+", fg_color=("#8e44ad", "#8e44ad"))
            self.lbl_status.configure(text="Aktív modell: DeepLabV3+ (MobileNet, epoch 38)")

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
        self._live_stack.grid_rowconfigure(1, minsize=self._footer_row_h)
        self.video_slider.grid(row=1, column=0, sticky="ew", padx=2, pady=(4, 0))

    def _hide_video_slider(self):
        """Csúszka elrejtése (kamera / szimuláció / nincs videó)."""
        self._slider_updating = True
        try:
            self.video_slider.set(0)
            self.video_slider.configure(state="disabled")
            if self._video_slider_packed:
                self.video_slider.grid_remove()
                self._live_stack.grid_rowconfigure(1, minsize=0)
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
        
        if self.video_thread:
            ret, frame = self.video_thread.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                tgt = self._preview_sz
                row_h = self._footer_row_h
                vf = self.video_thread.video_file_mode
                # Fájl + csúszka: a kép kisebb, a csúszka külön sorban; összmagasság ≈ tgt → nem nő az ablak
                if vf and self._video_slider_packed:
                    img_h = max(1, tgt - row_h)
                else:
                    img_h = tgt
                scale = min(img_h / h, tgt / w)
                display_w = max(1, min(tgt, int(w * scale)))
                display_h = max(1, min(img_h, int(h * scale)))

                frame_resized = cv2.resize(frame, (display_w, display_h))

                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                self._live_image_ref = ctk.CTkImage(
                    light_image=img, dark_image=img, size=(max(1, display_w), max(1, display_h))
                )
                self.live_panel.configure(image=self._live_image_ref, text="")
                self._live_stack.configure(width=tgt, height=tgt)

                # Live Analysis Trigger
                if self.check_live_analysis_var.get() and self.analyzer.models_loaded:
                    if not getattr(self, '_is_analyzing_live', False):
                        self._is_analyzing_live = True
                        threading.Thread(target=self._async_live_analyze, args=(frame, display_w, display_h), daemon=True).start()

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
        
        self.root.after(delay, self.update_live_feed)

    def _async_live_analyze(self, frame, display_w, display_h):
        # Háttérben lefut az elemzés, majd a főszálra ütemezi a megjelenítést
        try:
            results = self.analyzer.analyze(frame)
            self.root.after(0, lambda: self._apply_live_analysis_result(frame, display_w, display_h, results))
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
        sz = self._preview_sz - self._footer_row_h
        trans_img_pil = Image.fromarray(trans_img_vis)
        trans_img_resized = trans_img_pil.resize((sz, sz), Image.NEAREST)
        self._snap_image_ref = ctk.CTkImage(
            light_image=trans_img_resized, dark_image=trans_img_resized, size=(sz, sz)
        )
        self.snapshot_panel.configure(image=self._snap_image_ref, text="")
        
        # 2. Update Segmentation Panel
        seg_img = Image.fromarray(segmented_polyp)
        seg_img_resized = seg_img.resize((sz, sz), Image.NEAREST)
        self._seg_image_ref = ctk.CTkImage(
            light_image=seg_img_resized, dark_image=seg_img_resized, size=(sz, sz)
        )
        self.result_panel.configure(image=self._seg_image_ref, text="")

        self.lbl_result.configure(
            text=f"Szegm.: {seg_conf*100:.1f}% | {cls_res} ({conf*100:.1f}%)"
        )

    def capture_and_analyze(self):
        if not self.video_thread:
             self.lbl_status.configure(text="Please select a source first!")
             return

        self.lbl_status.configure(text="Analyzing... Please wait.")
        self.btn_analyze.configure(state="disabled") # Gomb letiltása amíg dolgozik
        self.root.update_idletasks() # Force UI update
        
        # Get snapshot
        ret, frame = self.video_thread.read()
        if not ret or frame is None:
            self.lbl_status.configure(text="Error: No video frame available.")
            self.btn_analyze.configure(state="normal")
            return

        h, w = frame.shape[:2]
        display_h = self._preview_sz - self._footer_row_h
        scale = display_h / h
        display_w = int(w * scale)
        
        # Külön szálon futtatjuk a hálózatot, hogy ne fagyjon a GUI
        threading.Thread(target=self._async_analyze, args=(frame, display_w, display_h), daemon=True).start()

    def _async_analyze(self, frame, display_w, display_h):
        start_time = time.time()
        # Elemzés lefut a háttérszálon...
        results = self.analyzer.analyze(frame)
        elapsed = time.time() - start_time
        
        # UI frissítések a main szálra ütemezve
        self.root.after(0, lambda: self._apply_analysis_result(frame, display_w, display_h, results, elapsed))

    def _apply_analysis_result(self, frame, display_w, display_h, results, elapsed):
        segmented_polyp, cls_res, conf, trans_img_vis, seg_conf = results
        # Store for manual save
        self._last_result = (frame, trans_img_vis, segmented_polyp, cls_res, conf, seg_conf)
        
        # Update UI
        self.run_analysis_on_frame(frame, display_w, display_h, results=results)
        
        # Auto save if enabled
        if self.save_mode_var.get() == "auto":
            self._save_results(*self._last_result)
        
        self.lbl_status.configure(text=f"Result: {cls_res} ({conf*100:.1f}%) | Processing Time: {elapsed:.3f}s")
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
