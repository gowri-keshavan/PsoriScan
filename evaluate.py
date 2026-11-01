
import argparse
import csv
import json

from psoriscan.config import DATA_DIR, DECISION_THRESHOLD, REPORT_DIR, SEED
from psoriscan.data import list_split
from psoriscan.imaging import load_image
from psoriscan.predictors import load_predictor
from psoriscan.reporting import compute_metrics, save_confusion_matrix, save_roc_curve


def evaluate_backend(backend: str, split: str, threshold: float, n_boot: int,
                     data_dir=DATA_DIR, tag: str = "") -> dict:
    predictor = load_predictor(backend)
    items = list_split(split, data_dir)
    name = f"{split}_{tag}" if tag else split
    probs = [predictor.predict_proba(load_image(path)) for path, _ in items]
    labels = [label for _, label in items]
    result = compute_metrics(labels, probs, threshold, n_boot, SEED)
    result.update(backend=backend, model=predictor.model_name, split=name)

    out = REPORT_DIR / backend
    out.mkdir(parents=True, exist_ok=True)
    (out / f"metrics_{name}.json").write_text(json.dumps(result, indent=2))
    with open(out / f"predictions_{name}.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "label", "probability_psoriasis"])
        writer.writerows((p.name, y, f"{pr:.4f}") for (p, y), pr in zip(items, probs))
    save_confusion_matrix(result["confusion_matrix"], out / f"confusion_matrix_{name}.png", result["model"])
    save_roc_curve(labels, probs, result["metrics"]["roc_auc"], out / f"roc_curve_{name}.png", result["model"])
    return result


def markdown_table(results: list[dict]) -> str:
    head = "| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |\n|---|---|---|---|---|---|---|\n"
    rows = []
    for r in results:
        m, ci = r["metrics"], r["ci95"]
        acc = f"{m['accuracy']:.3f} ({ci['accuracy'][0]:.3f}-{ci['accuracy'][1]:.3f})"
        auc = f"{m['roc_auc']:.3f} ({ci['roc_auc'][0]:.3f}-{ci['roc_auc'][1]:.3f})"
        rows.append(f"| {r['model']} | {acc} | {m['precision']:.3f} | {m['recall']:.3f} | "
                    f"{m['specificity']:.3f} | {m['f1']:.3f} | {auc} |")
    return head + "\n".join(rows) + f"\n\nn = {results[0]['n']} {results[0]['split']} images; brackets are bootstrap 95% CIs.\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["tf", "torch", "both"], default="both")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--threshold", type=float, default=DECISION_THRESHOLD)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="folder holding the split folders")
    parser.add_argument("--tag", default="", help="suffix for output files, e.g. external")
    args = parser.parse_args()

    backends = ["tf", "torch"] if args.backend == "both" else [args.backend]
    results = [evaluate_backend(b, args.split, args.threshold, args.bootstrap, args.data_dir, args.tag) for b in backends]
    table = markdown_table(results)
    (REPORT_DIR / f"comparison_{results[0]['split']}.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()