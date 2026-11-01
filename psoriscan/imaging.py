"""Image loading, preprocessing and heatmap rendering.

The same functions are used for training, evaluation and serving, so the model always
sees identically preprocessed pixels.
"""
import base64
import io

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import IMG_SIZE

ALLOWED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP", "BMP"}  # MPO = multi-picture JPEG from many phones
MAX_PIXELS = 50_000_000  # guards against decompression bombs
MIN_SIDE = 32


class InvalidImageError(ValueError):
    """Raised when an upload is not a usable image."""


def load_image(source) -> Image.Image:
    """Open a path or file-like object as an RGB image with EXIF rotation applied."""
    try:
        with Image.open(source) as im:
            if im.format not in ALLOWED_FORMATS:
                raise InvalidImageError("Unsupported image type. Use a JPEG, PNG, WEBP or BMP file.")
            width, height = im.size
            if width * height > MAX_PIXELS:
                raise InvalidImageError("Image is too large. Use one under 50 megapixels.")
            if min(width, height) < MIN_SIDE:
                raise InvalidImageError(f"Image is too small. Use one at least {MIN_SIDE}px on each side.")
            im.load()
            return ImageOps.exif_transpose(im).convert("RGB")
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("The file is not a readable image.") from exc


def resize_for_model(img: Image.Image) -> Image.Image:
    """Resize to the model's square input. Aspect ratio is not preserved (same as training)."""
    return img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)


def to_array(img: Image.Image) -> np.ndarray:
    """RGB image -> float32 array of shape (224, 224, 3) with values in 0-255."""
    return np.asarray(resize_for_model(img), dtype=np.float32)


def cam_to_rgba(cam: np.ndarray) -> Image.Image:
    """Turn a 0-1 Grad-CAM map into a transparent RGBA heatmap the size of the model input.

    Low-intensity areas are transparent so the original photo shows through.
    """
    cam = cv2.resize(cam.astype(np.float32), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    cam = np.clip(cam, 0.0, 1.0)
    colored = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]  # BGR -> RGB
    alpha = (np.clip(cam * 1.6 - 0.2, 0.0, 1.0) * 215).astype(np.uint8)
    return Image.fromarray(np.dstack([colored, alpha]), "RGBA")


def png_data_url(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
