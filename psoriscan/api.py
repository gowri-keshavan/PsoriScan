"""Flask API. Images are processed in memory and never written to disk."""
import logging

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from . import __version__, config
from .imaging import InvalidImageError, cam_to_rgba, load_image, png_data_url
from .predictors import Predictor, load_predictor

LABEL_POSITIVE = "Psoriasis"
LABEL_NEGATIVE = "Not psoriasis"


def _error(status: int, code: str, message: str):
    return jsonify({"error": {"code": code, "message": message}}), status


def create_app(predictor: Predictor | None = None) -> Flask:
    """App factory. Production: gunicorn 'app:create_app()'. Tests pass a stub predictor."""
    logging.basicConfig(level=logging.INFO)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
    CORS(app, resources={r"/(predict|health)": {"origins": config.CORS_ORIGINS}})
    predictor = predictor or load_predictor()

    @app.after_request
    def no_store(response):
        response.headers["Cache-Control"] = "no-store"  # responses contain medical images
        return response

    @app.get("/health")
    def health():
        return jsonify(status="ok", version=__version__, backend=predictor.backend, model=predictor.model_name)

    @app.post("/predict")
    def predict():
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return _error(400, "no_image", "Attach an image in the 'image' field.")
        try:
            image = load_image(upload.stream)
        except InvalidImageError as exc:
            return _error(415, "invalid_image", str(exc))

        prob, cam = predictor.analyze(image)
        is_positive = prob >= config.DECISION_THRESHOLD
        return jsonify(
            label=LABEL_POSITIVE if is_positive else LABEL_NEGATIVE,
            is_psoriasis=bool(is_positive),
            probability_psoriasis=round(prob, 4),
            confidence=round(prob if is_positive else 1.0 - prob, 4),
            threshold=config.DECISION_THRESHOLD,
            heatmap=png_data_url(cam_to_rgba(cam)),
            model={"backend": predictor.backend, "name": predictor.model_name},
            disclaimer=config.DISCLAIMER,
        )

    @app.errorhandler(413)
    def too_large(_):
        return _error(413, "file_too_large", f"Image is larger than {config.MAX_UPLOAD_MB} MB.")

    @app.errorhandler(Exception)
    def unexpected(exc):
        if isinstance(exc, HTTPException):
            return _error(exc.code or 500, exc.name.lower().replace(" ", "_"), exc.description or exc.name)
        app.logger.exception("Unhandled error")
        return _error(500, "server_error", "Something went wrong while analysing the image.")

    return app
