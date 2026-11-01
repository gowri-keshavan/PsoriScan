import csv
import io
import shutil

import numpy as np
from PIL import Image

from prepare_dataset import assign_splits, find_clusters, prepare
from psoriscan.config import CLASS_NAMES


def save_noise(path, seed, quality=95):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(path, quality=quality)


def build_raw(root):
    for ci, cls in enumerate(CLASS_NAMES):
        folder = root / "train" / cls  # nested like the original dataset
        folder.mkdir(parents=True)
        for i in range(40):
            save_noise(folder / f"{i}.jpg", seed=ci * 1000 + i)
        shutil.copy(folder / "0.jpg", folder / "exact_copy.jpg")  # exact duplicate
        save_noise(folder / "0_recompressed.jpg", seed=ci * 1000, quality=60)  # near-duplicate of 0.jpg
    (root / "train" / CLASS_NAMES[0] / "broken.jpg").write_bytes(b"not an image")


def test_find_clusters_groups_close_hashes():
    base = np.zeros(256, dtype=bool)
    near = base.copy(); near[:3] = True
    far = ~base
    clusters = find_clusters(np.stack([base, near, far]), threshold=6)
    assert clusters[0] == clusters[1] != clusters[2]


def test_assign_splits_keeps_clusters_whole_and_close_to_fractions():
    sizes = {i: 1 for i in range(100)} | {1000: 5, 1001: 5}
    fractions = {"train": 0.7, "val": 0.15, "test": 0.15}
    split = assign_splits(sizes, fractions, seed=1)
    totals = {s: sum(sizes[c] for c, sp in split.items() if sp == s) for s in fractions}
    assert abs(totals["train"] / 110 - 0.7) < 0.05 and abs(totals["test"] / 110 - 0.15) < 0.05


def test_prepare_removes_duplicates_and_prevents_leakage(tmp_path):
    build_raw(tmp_path / "raw")
    summary = prepare(tmp_path / "raw", tmp_path / "data", tmp_path / "reports",
                      {"train": 0.7, "val": 0.15, "test": 0.15})
    assert summary["exact_duplicates_removed"] == 2
    assert summary["unreadable_skipped"] == 1
    assert summary["leakage_check"] == "passed"
    rows = list(csv.DictReader(open(tmp_path / "data" / "manifest.csv")))
    cluster_splits = {}
    for row in rows:
        cluster_splits.setdefault(row["cluster"], set()).add(row["split"])
    assert all(len(s) == 1 for s in cluster_splits.values())
    # the recompressed near-duplicate must sit in the same split as its original
    original = next(r for r in rows if r["source"].endswith("no_psoriasis/0.jpg"))
    near = next(r for r in rows if r["source"].endswith("no_psoriasis/0_recompressed.jpg"))
    assert original["split"] == near["split"]


def test_prepare_refuses_to_overwrite_without_force(tmp_path):
    build_raw(tmp_path / "raw")
    args = (tmp_path / "raw", tmp_path / "data", tmp_path / "reports", {"train": 0.7, "val": 0.15, "test": 0.15})
    prepare(*args)
    try:
        prepare(*args)
    except SystemExit:
        return
    raise AssertionError("expected SystemExit")


def test_positive_keyword_excludes_look_alike_conditions(tmp_path):
    pos = tmp_path / "raw" / "psoriasis"
    neg = tmp_path / "raw" / "no_psoriasis"
    pos.mkdir(parents=True)
    neg.mkdir(parents=True)
    for i in range(12):
        save_noise(pos / f"Psoriasis-scalp-{i}.jpg", seed=i)
        save_noise(neg / f"{i}.jpg", seed=500 + i)
    for i in range(5):
        save_noise(pos / f"lichen-planus-{i}.jpg", seed=900 + i)

    summary = prepare(tmp_path / "raw", tmp_path / "data", tmp_path / "reports",
                      {"train": 0.7, "val": 0.15, "test": 0.15}, positive_keyword="psoria")

    assert summary["excluded_by_positive_keyword"] == 5
    assert summary["images_kept"] == 24 and summary["images_found"] == 29
    rows = list(csv.DictReader(open(tmp_path / "data" / "manifest.csv")))
    assert not any("lichen" in r["source"] for r in rows)
    assert sum(r["class"] == "psoriasis" for r in rows) == 12


def test_group_regex_keeps_numbered_siblings_together(tmp_path):
    neg = tmp_path / "raw" / "no_psoriasis"
    pos = tmp_path / "raw" / "psoriasis"
    neg.mkdir(parents=True)
    pos.mkdir(parents=True)
    for group in range(1, 9):  # files 1.1 ... 8.5: five variants of each source photo
        for k in range(1, 6):
            save_noise(neg / f"{group}.{k}.jpg", seed=group * 100 + k)
    for i in range(30):
        save_noise(pos / f"psoriasis-{i}.jpg", seed=5000 + i)

    prepare(tmp_path / "raw", tmp_path / "data", tmp_path / "reports",
            {"train": 0.6, "val": 0.2, "test": 0.2}, group_regex=r"^(\d+)[.,]\d+$")

    rows = [r for r in csv.DictReader(open(tmp_path / "data" / "manifest.csv")) if r["class"] == "no_psoriasis"]
    splits_per_group = {}
    for r in rows:
        group = r["source"].rsplit("/", 1)[-1].split(".")[0]
        splits_per_group.setdefault(group, set()).add(r["split"])
    assert len(splits_per_group) == 8
    assert all(len(s) == 1 for s in splits_per_group.values())