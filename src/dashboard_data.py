"""Derived per-letter views over saved official-test metrics."""

from __future__ import annotations

import numpy as np


def class_test_profile(metrics: dict, letter: str, top_k: int = 3) -> dict:
    """Return genuine per-class scores and most common wrong labels."""
    selected = letter.upper()
    if len(selected) != 1 or selected not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        raise ValueError("letter must be one character from A to Z")

    report = metrics["classification_report"].get(selected)
    if report is None:
        raise ValueError(f"No test report is available for letter {selected}")
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    if matrix.shape != (26, 26):
        raise ValueError(f"Expected a 26 by 26 confusion matrix, received {matrix.shape}")

    class_index = ord(selected) - ord("A")
    mistakes = matrix[class_index].copy()
    mistakes[class_index] = 0
    ranked_indices = sorted(range(26), key=lambda index: (-int(mistakes[index]), index))
    support = int(report["support"])
    confusions = [
        {
            "predicted_letter": chr(ord("A") + index),
            "count": int(mistakes[index]),
            "share": float(mistakes[index] / support) if support else 0.0,
        }
        for index in ranked_indices[:top_k]
        if mistakes[index] > 0
    ]
    return {
        "letter": selected,
        "precision": float(report["precision"]),
        "recall": float(report["recall"]),
        "f1": float(report["f1-score"]),
        "support": support,
        "correct": int(matrix[class_index, class_index]),
        "confusions": confusions,
    }