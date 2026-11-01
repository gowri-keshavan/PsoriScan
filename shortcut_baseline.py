
import argparse
import json

import numpy as np
from PIL import Image
from sklearn.tree import DecisionTreeClassifier

from psoriscan.config import DATA_DIR, REPORT_DIR, SEED
from psoriscan.data import list_split
from psoriscan.reporting import compute_metrics


def features(path) -> list[float]:
    with Image.open(path) as im:  # reads the header only, not the pixels
        width, height = im.size
    size = path.stat().st_size
    return [width, height, width / height, size, size / (width * height), float(path.suffix.lower() == ".png")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    train, held_out = list_split("train", args.data_dir), list_split(args.split, args.data_dir)
    x_train = np.array([features(p) for p, _ in train])
    y_train = [label for _, label in train]
    x_eval = np.array([features(p) for p, _ in held_out])
    y_eval = [label for _, label in held_out]

    tree = DecisionTreeClassifier(max_depth=4, random_state=SEED).fit(x_train, y_train)
    result = compute_metrics(y_eval, tree.predict_proba(x_eval)[:, 1], 0.5, args.bootstrap, SEED)
    m, ci = result["metrics"], result["ci95"]
    print(f"Metadata-only decision tree on {args.split} (n={result['n']}):")
    print(f"  accuracy {m['accuracy']:.3f} ({ci['accuracy'][0]:.3f}-{ci['accuracy'][1]:.3f}), "
          f"ROC-AUC {m['roc_auc']:.3f} ({ci['roc_auc'][0]:.3f}-{ci['roc_auc'][1]:.3f})")
    if m["accuracy"] > 0.75:
        print("  WARNING: file properties alone separate the classes. Treat model scores with caution.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result.update(model="Metadata-only decision tree (no pixels)", split=args.split)
    (REPORT_DIR / f"shortcut_baseline_{args.split}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()