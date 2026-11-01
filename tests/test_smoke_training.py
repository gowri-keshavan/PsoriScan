"""Tiny end-to-end runs: prepare data -> train 1 epoch -> evaluate -> serve one prediction.

"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from prepare_dataset import prepare
from psoriscan.config import CLASS_NAMES

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.slow

TRAIN_CMD = {
    "tf": ["train_tf.py", "--epochs", "1", "--batch-size", "8"],
    "torch": ["train_torch.py", "--epochs-head", "1", "--epochs-finetune", "1", "--batch-size", "8",
              "--no-pretrained", "--num-workers", "0"],
}


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    base = tmp_path_factory.mktemp("smoke")
    rng = np.random.default_rng(0)
    for ci, cls in enumerate(CLASS_NAMES):
        folder = base / "raw" / cls
        folder.mkdir(parents=True)
        for i in range(30):
            pixels = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            pixels[..., 0] = np.clip(pixels[..., 0] + ci * 60, 0, 255)  # class 1 is redder
            Image.fromarray(pixels).save(folder / f"{i}.png")
    prepare(base / "raw", base / "data", base / "reports", {"train": 0.6, "val": 0.2, "test": 0.2})
    return base


def run(workspace, *args):
    env = {**os.environ, "DATA_DIR": str(workspace / "data"), "MODEL_DIR": str(workspace / "models"),
           "REPORT_DIR": str(workspace / "reports"), "TF_CPP_MIN_LOG_LEVEL": "3"}
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)


@pytest.mark.parametrize("backend", ["tf", "torch"])
def test_train_evaluate_and_serve(workspace, backend):
    pytest.importorskip("tensorflow" if backend == "tf" else "torch")
    trained = run(workspace, *TRAIN_CMD[backend])
    assert trained.returncode == 0, trained.stderr[-2000:]

    evaluated = run(workspace, "evaluate.py", "--backend", backend, "--bootstrap", "20")
    assert evaluated.returncode == 0, evaluated.stderr[-2000:]
    assert (workspace / "reports" / backend / "metrics_test.json").exists()

    sample = next((workspace / "data" / "test" / "psoriasis").glob("*"))
    script = textwrap.dedent(f"""
        from psoriscan.api import create_app
        from psoriscan.predictors import load_predictor
        client = create_app(load_predictor("{backend}")).test_client()
        r = client.post("/predict", data={{"image": (open(r"{sample}", "rb"), "x.png")}}, content_type="multipart/form-data")
        body = r.get_json()
        assert r.status_code == 200 and body["model"]["backend"] == "{backend}", body
        assert 0 <= body["probability_psoriasis"] <= 1 and body["heatmap"].startswith("data:image/png")
        print("OK")
    """)
    served = run(workspace, "-c", script)
    assert served.returncode == 0 and "OK" in served.stdout, served.stderr[-2000:]
