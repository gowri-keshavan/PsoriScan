
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


# --- Folders -----------------------------------------------------------------
RAW_DATA_DIR = _path("RAW_DATA_DIR", BASE_DIR / "dataset")  # input of prepare_dataset.py
DATA_DIR = _path("DATA_DIR", BASE_DIR / "data")  # output: train/ val/ test/
MODEL_DIR = _path("MODEL_DIR", BASE_DIR / "models")
REPORT_DIR = _path("REPORT_DIR", BASE_DIR / "reports")

# --- Data / model contract (shared by training, evaluation and serving) ------
IMG_SIZE = 224
CLASS_NAMES = ("no_psoriasis", "psoriasis")  # index 1 is the positive class
SEED = 42
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TF_LAST_CONV_LAYER = "last_conv"  # layer used for Grad-CAM in the TensorFlow CNN
MODEL_FILES = {"tf": "psoriasis_cnn.keras", "torch": "psoriasis_resnet18.pt"}
MODEL_NAMES = {"tf": "Custom CNN (TensorFlow/Keras)", "torch": "ResNet18 transfer learning (PyTorch)"}

# --- Serving -----------------------------------------------------------------
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "torch")  # "torch" or "tf"
MODEL_URL = os.getenv("MODEL_URL", "")  # optional https URL, downloaded if the file is missing
MODEL_SHA256 = os.getenv("MODEL_SHA256", "")  # optional checksum for the download
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "8"))
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]

DISCLAIMER = (
    "PsoriScan is a screening aid built as a machine learning project. It is not a medical "
    "device and cannot diagnose psoriasis. See a dermatologist for any skin concern."
)
