import base64
import io

from conftest import StubPredictor, make_image_bytes
from PIL import Image

from psoriscan.api import create_app


def post_image(client, data: bytes, name="skin.png"):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)}, content_type="multipart/form-data")


def test_health(client):
    body = client.get("/health").get_json()
    assert body["status"] == "ok" and body["backend"] == "tf"


def test_predict_positive(client):
    response = post_image(client, make_image_bytes())
    body = response.get_json()
    assert response.status_code == 200
    assert body["label"] == "Psoriasis" and body["is_psoriasis"] is True
    assert body["probability_psoriasis"] == 0.9 and body["confidence"] == 0.9
    assert "not a medical device" in body["disclaimer"]
    assert response.headers["Cache-Control"] == "no-store"


def test_predict_negative_reports_confidence_in_predicted_class():
    client = create_app(StubPredictor(prob=0.04)).test_client()
    body = post_image(client, make_image_bytes()).get_json()
    assert body["label"] == "Not psoriasis"
    assert body["probability_psoriasis"] == 0.04 and body["confidence"] == 0.96


def test_heatmap_is_transparent_png(client):
    body = post_image(client, make_image_bytes()).get_json()
    assert body["heatmap"].startswith("data:image/png;base64,")
    heatmap = Image.open(io.BytesIO(base64.b64decode(body["heatmap"].split(",")[1])))
    assert heatmap.mode == "RGBA" and heatmap.size == (224, 224)


def test_accepts_jpeg(client):
    assert post_image(client, make_image_bytes("JPEG"), "photo.jpg").status_code == 200


def test_missing_file(client):
    response = client.post("/predict")
    assert response.status_code == 400 and response.get_json()["error"]["code"] == "no_image"


def test_not_an_image(client):
    response = post_image(client, b"definitely not an image", "x.jpg")
    assert response.status_code == 415 and response.get_json()["error"]["code"] == "invalid_image"


def test_unsupported_format(client):
    response = post_image(client, make_image_bytes("GIF"), "x.gif")
    assert response.status_code == 415


def test_too_large(client):
    response = post_image(client, b"0" * (9 * 1024 * 1024))
    assert response.status_code == 413 and response.get_json()["error"]["code"] == "file_too_large"


def test_unknown_route_returns_json_404(client):
    response = client.get("/nope")
    assert response.status_code == 404 and "error" in response.get_json()
