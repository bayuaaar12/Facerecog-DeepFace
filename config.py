import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")

import platform
from pathlib import Path

# Path Konfigurasi
BASE_DIR = Path(__file__).resolve().parent
CASCADE_PATH = BASE_DIR / "face_ref.xml"
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
EMBEDDINGS_PATH = BASE_DIR / "known_faces_embeddings.json"

# API Endpoints Default
DEFAULT_API_URL = "http://127.0.0.1:8000/api/customers/register-face"
DEFAULT_DETECTION_API_URL = "http://127.0.0.1:8000/api/customers/detect-member"

# Palette Warna UI (BGR Format untuk OpenCV & RGB Hex untuk Tkinter)
COLOR_PRIMARY = (238, 112, 35)
COLOR_PRIMARY_SOFT = (255, 241, 229)
COLOR_ACCENT = (75, 167, 88)
COLOR_ACCENT_SOFT = (226, 247, 231)
COLOR_TEXT = (28, 31, 36)
COLOR_MUTED = (116, 124, 138)
COLOR_BORDER = (220, 226, 235)
COLOR_BG = (246, 248, 251)
COLOR_PANEL = (255, 255, 255)
COLOR_SURFACE = (236, 240, 246)
COLOR_BLUE = COLOR_PRIMARY
COLOR_BLUE_SOFT = COLOR_PRIMARY_SOFT

# Model & Parameter AI/ML Inference
FACENET_MODEL_NAME = "Facenet512"
# Referensi Threshold: 0.40 berasal dari rekomendasi default DeepFace untuk Facenet512 dengan Cosine Metric,
# dan telah dikonfirmasi melalui hasil eksperimen empiris (hasil_evaluasi_full.txt) untuk keseimbangan Precision-Recall.
COSINE_DISTANCE_THRESHOLD = 0.40
EMBEDDING_FILE_VERSION = 1

# Parameter Deteksi & Preprocessing Citra Klasik (Non-AI)
LOW_LIGHT_MEAN_LIMIT = 95
LOW_LIGHT_TARGET_MEAN = 125
LOW_LIGHT_BRIGHTNESS_BONUS = 18
FACE_DETECTION_ANGLES = (0, -20, 20, -35, 35)
FACE_DETECTION_IOU_LIMIT = 0.35
REGISTER_SAMPLE_COUNT = 5
REGISTER_SAMPLE_INTERVAL = 0.35
REGISTER_SAMPLE_TIMEOUT = 6.0
