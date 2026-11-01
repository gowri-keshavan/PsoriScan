"""Plots and metrics shared by the training and evaluation scripts."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)


def save_history_plot(history: dict, path: Path, title: str) -> None:
    """Loss and accuracy curves. Expects keys loss/val_loss and accuracy/val_accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key in zip(axes, ("loss", "accuracy")):
        ax.plot(history[key], label="train")
        ax.plot(history["val_" + key], label="validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compute_metrics(y_true, probs, threshold: float = 0.5, n_boot: int = 1000, seed: int = 42) -> dict:
    """Classification metrics at `threshold`, with bootstrap 95% confidence intervals."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    def point(y, p, hard):
        has_both = len(set(y)) == 2
        return {
            "accuracy": accuracy_score(y, hard),
            "precision": precision_score(y, hard, zero_division=0),
            "recall": recall_score(y, hard, zero_division=0),
            "f1": f1_score(y, hard, zero_division=0),
            "roc_auc": roc_auc_score(y, p) if has_both else float("nan"),
        }

    metrics = point(y_true, probs, pred)
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) else float("nan")
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in ("accuracy", "f1", "roc_auc")}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        stats = point(y_true[idx], probs[idx], pred[idx])
        for k in samples:
            samples[k].append(stats[k])
    ci = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in samples.items()}
    return {
        "n": int(len(y_true)), "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "metrics": {k: float(v) for k, v in metrics.items()},
        "ci95": ci,
    }


def save_confusion_matrix(cm: dict, path: Path, title: str) -> None:
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(mat, cmap="Blues")
    for (i, j), value in np.ndenumerate(mat):
        ax.text(j, i, str(value), ha="center", va="center", color="white" if value > mat.max() / 2 else "black")
    labels = ["no psoriasis", "psoriasis"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_roc_curve(y_true, probs, auc: float, path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, probs)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
