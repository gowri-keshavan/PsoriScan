import io

import numpy as np
import pytest
from PIL import Image

from psoriscan.predictors import Predictor


class StubPredictor(Predictor):
    """Fixed-output predictor so API tests need no model and no ML framework."""

    backend = "tf"

    def __init__(self, prob: float = 0.9):
        super().__init__()
        self.prob = prob

    def _predict(self, arr):
        return self.prob

    def _analyze(self, arr):
        cam = np.zeros((7, 7), dtype=np.float32)
        cam[3, 3] = 1.0
        return self.prob, cam


def make_image_bytes(fmt="PNG", size=(96, 80), seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def stub():
    return StubPredictor()


@pytest.fixture
def client(stub):
    from psoriscan.api import create_app

    return create_app(stub).test_client()
