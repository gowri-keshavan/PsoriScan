"""Local development entry point:  python app.py
Production:  gunicorn 'app:create_app()'  (see Dockerfile)."""
import os

from psoriscan.api import create_app

__all__ = ["create_app"]

if __name__ == "__main__":
    create_app().run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG") == "1",
    )
