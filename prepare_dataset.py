
import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from psoriscan.config import CLASS_NAMES, DATA_DIR, RAW_DATA_DIR, REPORT_DIR, SEED
from psoriscan.data import IMAGE_EXTS
from psoriscan.imaging import InvalidImageError, load_image

SPLITS = ("train", "val", "test")


def dhash(img: Image.Image, size: int = 16) -> np.ndarray:
    """256-bit difference hash. Near-identical photos differ in only a few bits."""
    gray = np.asarray(img.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    return (gray[:, 1:] > gray[:, :-1]).flatten()


def find_clusters(hashes: np.ndarray, threshold: int, groups: list[str | None] | None = None) -> list[int]:
    """Union-find over images whose hashes differ in <= `threshold` bits. Returns a cluster id per image.

    `groups` optionally gives a key per image (or None); images sharing a key are always merged.
    """
    n = len(hashes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n - 1):
        distances = (hashes[i + 1 :] != hashes[i]).sum(axis=1)
        for j in np.where(distances <= threshold)[0]:
            parent[find(i + 1 + int(j))] = find(i)
    if groups:
        first_seen: dict[str, int] = {}
        for i, key in enumerate(groups):
            if key is None:
                continue
            if key in first_seen:
                parent[find(i)] = find(first_seen[key])
            else:
                first_seen[key] = i
    return [find(i) for i in range(n)]


def assign_splits(clusters: dict[int, int], fractions: dict[str, float], seed: int) -> dict[int, str]:
    """Greedily assign whole clusters (id -> size) to the split furthest below its target size."""
    total = sum(clusters.values())
    target = {s: fractions[s] * total for s in SPLITS}
    filled = dict.fromkeys(SPLITS, 0)
    ids = sorted(clusters)
    random.Random(seed).shuffle(ids)
    ids.sort(key=lambda c: -clusters[c])  # place big clusters first, ties keep the shuffled order
    assignment = {}
    for cid in ids:
        split = max(SPLITS, key=lambda s: target[s] - filled[s])
        assignment[cid] = split
        filled[split] += clusters[cid]
    return assignment


def scan(raw_dir: Path, positive_keyword: str | None = None) -> tuple[list[tuple[Path, int]], list[Path]]:
    """Return (kept, excluded) image lists.

    With `positive_keyword`, images in the positive class whose file name lacks that keyword
    (case-insensitive) are excluded. Use it when the positive folder also holds look-alike
    conditions (e.g. lichen planus) that are only identifiable from the file name.
    """
    items, excluded = [], []
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and path.parent.name in CLASS_NAMES:
            label = CLASS_NAMES.index(path.parent.name)
            if positive_keyword and label == 1 and positive_keyword.lower() not in path.stem.lower():
                excluded.append(path)
            else:
                items.append((path, label))
    return items, excluded


def prepare(raw_dir: Path, out_dir: Path, report_dir: Path, fractions: dict[str, float],
            near_threshold: int = 6, seed: int = SEED, force: bool = False,
            positive_keyword: str | None = None, group_regex: str | None = None) -> dict:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            raise SystemExit(f"{out_dir} already has files. Re-run with --force to overwrite it.")
        shutil.rmtree(out_dir)

    found, name_excluded = scan(raw_dir, positive_keyword)
    if not found:
        raise SystemExit(f"No images found under {raw_dir}/**/{{{','.join(CLASS_NAMES)}}}/")

    records, unreadable, exact_dupes = [], [], []
    seen_md5: dict[str, Path] = {}
    for path, label in found:
        try:
            image = load_image(path)
        except InvalidImageError as exc:
            unreadable.append(f"{path}: {exc}")
            continue
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        if md5 in seen_md5:
            exact_dupes.append(str(path))
            continue
        seen_md5[md5] = path
        records.append({"path": path, "label": label, "md5": md5, "hash": dhash(image)})

    group_keys = None
    if group_regex:
        pattern = re.compile(group_regex)
        group_keys = []
        for r in records:
            match = pattern.match(r["path"].stem)
            group_keys.append(f"{r['label']}:{match.group(1)}" if match else None)
    cluster_of = find_clusters(np.stack([r["hash"] for r in records]), near_threshold, group_keys)
    members = defaultdict(list)
    for record, cid in zip(records, cluster_of):
        record["cluster"] = cid
        members[cid].append(record)

    # Near-duplicates with conflicting labels cannot be trusted: drop the whole cluster.
    conflicting = [cid for cid, rs in members.items() if len({r["label"] for r in rs}) > 1]
    for cid in conflicting:
        del members[cid]

    split_of = {}
    for label in range(len(CLASS_NAMES)):  # stratify: split each class separately
        sizes = {cid: len(rs) for cid, rs in members.items() if rs[0]["label"] == label}
        split_of.update(assign_splits(sizes, fractions, seed + label))

    rows = []
    for cid, rs in members.items():
        for r in rs:
            split = split_of[cid]
            name = f"{r['md5'][:10]}_{r['path'].name}"
            dest = out_dir / split / CLASS_NAMES[r["label"]] / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(r["path"], dest)
            rows.append({"file": dest.relative_to(out_dir).as_posix(), "class": CLASS_NAMES[r["label"]],
                         "split": split, "md5": r["md5"], "cluster": cid, "source": r["path"].as_posix()})

    with open(out_dir / "manifest.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Leakage self-check: no file content or near-duplicate cluster may appear in two splits.
    for key in ("md5", "cluster"):
        owner = defaultdict(set)
        for row in rows:
            owner[row[key]].add(row["split"])
        assert all(len(s) == 1 for s in owner.values()), f"leakage detected via {key}"

    counts = {s: {c: sum(1 for r in rows if r["split"] == s and r["class"] == c) for c in CLASS_NAMES} for s in SPLITS}
    summary = {
        "images_found": len(found) + len(name_excluded), "images_kept": len(rows),
        "positive_keyword": positive_keyword, "group_regex": group_regex,
        "excluded_by_positive_keyword": len(name_excluded),
        "unreadable_skipped": len(unreadable), "exact_duplicates_removed": len(exact_dupes),
        "near_duplicate_clusters_with_conflicting_labels_dropped": len(conflicting),
        "near_duplicate_threshold_bits_of_256": near_threshold,
        "split_counts": counts, "seed": seed, "leakage_check": "passed",
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--near-threshold", type=int, default=6, help="max differing bits (of 256) to call two images near-duplicates")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--positive-keyword", default=None,
                        help="keep only positive-class images whose file name contains this text (case-insensitive)")
    parser.add_argument("--group-regex", default=None,
                        help="regex with one capture group; images with the same captured value (and class) are kept in the same split")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    fractions = {"train": args.train, "val": args.val, "test": args.test}
    if abs(sum(fractions.values()) - 1) > 1e-6:
        raise SystemExit("--train, --val and --test must add up to 1.")
    print(json.dumps(prepare(args.raw_dir, args.out_dir, args.report_dir, fractions,
                             args.near_threshold, args.seed, args.force, args.positive_keyword,
                             args.group_regex), indent=2))


if __name__ == "__main__":
    main()