import numpy as np
import pytest

from src.dashboard_data import class_test_profile


def test_letter_profile_uses_selected_test_row():
    matrix = np.zeros((26, 26), dtype=np.int64)
    matrix[1, 1] = 4
    matrix[1, 0] = 2
    matrix[1, 3] = 4
    metrics = {
        "classification_report": {
            "B": {"precision": 0.8, "recall": 0.4, "f1-score": 0.533, "support": 10}
        },
        "confusion_matrix": matrix.tolist(),
    }

    profile = class_test_profile(metrics, "b")

    assert profile["letter"] == "B"
    assert profile["correct"] == 4
    assert profile["support"] == 10
    assert [(row["predicted_letter"], row["count"]) for row in profile["confusions"]] == [("D", 4), ("A", 2)]


def test_letter_profile_rejects_invalid_letter():
    with pytest.raises(ValueError, match="A to Z"):
        class_test_profile({}, "AA")