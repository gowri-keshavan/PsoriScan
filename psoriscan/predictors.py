"""Inference wrappers. Training, evaluation and the web API all go through these classes."""
import hashlib
import logging
import threading
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image

from . import config
from .imaging import to_array

log = logging.getLogger(__name__)


class ModelNotFoundError(FileNotFoundError):
    pass


class Predictor(ABC):
    """Common interface: probability of psoriasis plus a Grad-CAM explanation map."""

    backend: str

    def __init__(self) -> None:
        self._lock = threading.Lock()  # frameworks are not guaranteed thread-safe for one model

    @property
    def model_name(self) -> str:
        return config.MODEL_NAMES[self.backend]

    def predict_proba(self, img: Image.Image) -> float:
        with self._lock:
            return self._predict(to_array(img))

    def analyze(self, img: Image.Image) -> tuple[float, np.ndarray]:
        """Return (P(psoriasis), Grad-CAM map scaled to 0-1)."""
        with self._lock:
            prob, cam = self._analyze(to_array(img))
        cam = np.maximum(cam, 0.0)
        return prob, cam / (cam.max() + 1e-8)

    @abstractmethod
    def _predict(self, arr: np.ndarray) -> float: ...

    @abstractmethod
    def _analyze(self, arr: np.ndarray) -> tuple[float, np.ndarray]: ...


class TFPredictor(Predictor):
    backend = "tf"

    def __init__(self, path: Path) -> None:
        super().__init__()
        import tensorflow as tf

        self._tf = tf
        self.model = tf.keras.models.load_model(path, compile=False)
        last_conv = self.model.get_layer(config.TF_LAST_CONV_LAYER).output
        self._grad_model = tf.keras.Model(self.model.inputs, [last_conv, self.model.output])

    def _predict(self, arr):
        out = self.model(arr[None], training=False)
        return float(out.numpy()[0, 0])

    def _analyze(self, arr):
        tf = self._tf
        x = tf.convert_to_tensor(arr[None])
        with tf.GradientTape() as tape:
            conv, pred = self._grad_model(x, training=False)
            score = pred[:, 0]
        grads = tape.gradient(score, conv)
        weights = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.reduce_sum(conv * weights[:, None, None, :], axis=-1)[0]
        return float(pred.numpy()[0, 0]), cam.numpy()


class TorchPredictor(Predictor):
    backend = "torch"

    def __init__(self, path: Path) -> None:
        super().__init__()
        import torch

        from .torch_model import build_resnet18

        self._torch = torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.model = build_resnet18(pretrained=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self._mean = torch.tensor(config.IMAGENET_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(config.IMAGENET_STD).view(1, 3, 1, 1)

    def _tensor(self, arr):
        x = self._torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 255.0
        return (x - self._mean) / self._std

    def _predict(self, arr):
        with self._torch.inference_mode():
            return float(self._torch.sigmoid(self.model(self._tensor(arr)))[0, 0])

    def _analyze(self, arr):
        torch = self._torch
        captured = {}
        handle = self.model.layer4.register_forward_hook(lambda m, i, o: captured.update(act=o))
        try:
            with torch.enable_grad():
                logit = self.model(self._tensor(arr))[0, 0]
                (grads,) = torch.autograd.grad(logit, captured["act"])
        finally:
            handle.remove()
        act = captured["act"][0].detach()
        weights = grads[0].mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * act).sum(0))
        return float(torch.sigmoid(logit.detach())), cam.numpy()


def ensure_model_file(path: Path) -> Path:
    """Return `path`, downloading it from MODEL_URL first if it is missing."""
    path = Path(path)
    if path.exists():
        return path
    if not config.MODEL_URL:
        raise ModelNotFoundError(
            f"Model file {path} not found. Train one (train_torch.py / train_tf.py) "
            "or set MODEL_URL to a downloadable https link."
        )
    if not config.MODEL_URL.startswith("https://"):
        raise ModelNotFoundError("MODEL_URL must start with https://")
    log.info("Downloading model from %s", config.MODEL_URL)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(config.MODEL_URL, tmp)  # noqa: S310 (https enforced above)
    if config.MODEL_SHA256:
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if digest != config.MODEL_SHA256.lower():
            tmp.unlink()
            raise ModelNotFoundError("Downloaded model failed the SHA-256 check.")
    tmp.replace(path)
    return path


def load_predictor(backend: str | None = None) -> Predictor:
    backend = backend or config.MODEL_BACKEND
    if backend not in config.MODEL_FILES:
        raise ValueError(f"Unknown backend {backend!r}. Use one of {sorted(config.MODEL_FILES)}.")
    path = ensure_model_file(config.MODEL_DIR / config.MODEL_FILES[backend])
    return TFPredictor(path) if backend == "tf" else TorchPredictor(path)
