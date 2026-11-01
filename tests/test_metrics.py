import math

from psoriscan.reporting import compute_metrics


def test_perfect_classifier():
    result = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], n_boot=50)
    assert result["metrics"]["accuracy"] == 1.0 and result["metrics"]["roc_auc"] == 1.0
    assert result["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_known_confusion_matrix():
    # one false positive, one false negative
    result = compute_metrics([0, 0, 1, 1], [0.6, 0.1, 0.4, 0.9], n_boot=50)
    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    m = result["metrics"]
    assert math.isclose(m["precision"], 0.5) and math.isclose(m["recall"], 0.5) and math.isclose(m["specificity"], 0.5)


def test_threshold_changes_predictions():
    low = compute_metrics([0, 1], [0.3, 0.4], threshold=0.35, n_boot=10)["confusion_matrix"]
    high = compute_metrics([0, 1], [0.3, 0.4], threshold=0.5, n_boot=10)["confusion_matrix"]
    assert low["tp"] == 1 and high["fn"] == 1


def test_confidence_intervals_bracket_the_estimate():
    probs = [0.2, 0.7, 0.4, 0.9, 0.1, 0.8, 0.6, 0.3]
    result = compute_metrics([0, 1, 1, 1, 0, 1, 0, 0], probs, n_boot=200)
    low, high = result["ci95"]["accuracy"]
    assert low <= result["metrics"]["accuracy"] <= high
