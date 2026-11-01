"""Access to the prepared train/val/test folders created by prepare_dataset.py."""
from pathlib import Path

import numpy as np

from .config import CLASS_NAMES, DATA_DIR
from .imaging import load_image, to_array

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_split(split: str, root: Path = DATA_DIR) -> list[tuple[Path, int]]:
    """Return (path, label) pairs for a split. Label 1 means psoriasis."""
    items = []
    for label, cls in enumerate(CLASS_NAMES):
        folder = Path(root) / split / cls
        items += [(p, label) for p in sorted(folder.glob("*")) if p.suffix.lower() in IMAGE_EXTS]
    if not items:
        raise FileNotFoundError(f"No images in {Path(root) / split}. Run `python prepare_dataset.py` first.")
    return items


def load_arrays(split: str, root: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Load a split into memory as uint8 images (N, 224, 224, 3) and float32 labels (N,)."""
    items = list_split(split, root)
    images = np.stack([to_array(load_image(p)).astype(np.uint8) for p, _ in items])
    labels = np.array([label for _, label in items], dtype=np.float32)
    return images, labels
