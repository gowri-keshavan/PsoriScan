import io

import numpy as np
import pytest
from conftest import make_image_bytes
from PIL import Image

from psoriscan.imaging import IMG_SIZE, InvalidImageError, cam_to_rgba, load_image, to_array


def test_load_image_returns_rgb():
    img = load_image(io.BytesIO(make_image_bytes()))
    assert img.mode == "RGB"


def test_load_image_rejects_garbage():
    with pytest.raises(InvalidImageError):
        load_image(io.BytesIO(b"nope"))


def test_load_image_rejects_tiny_images():
    with pytest.raises(InvalidImageError):
        load_image(io.BytesIO(make_image_bytes(size=(20, 20))))


def test_load_image_applies_exif_rotation():
    img = Image.new("RGB", (60, 40), "red")
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise when displayed
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    assert load_image(io.BytesIO(buf.getvalue())).size == (40, 60)


def test_to_array_shape_and_range():
    arr = to_array(Image.new("RGB", (300, 120), (255, 0, 0)))
    assert arr.shape == (IMG_SIZE, IMG_SIZE, 3) and arr.dtype == np.float32 and arr.max() <= 255


def test_cam_to_rgba_low_values_are_transparent():
    cam = np.zeros((7, 7), dtype=np.float32)
    cam[3, 3] = 1.0
    alpha = np.asarray(cam_to_rgba(cam))[..., 3]
    assert alpha[0, 0] == 0 and alpha.max() > 150
