"""Held-out test evaluation and machine-readable artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from config import CLASS_NAMES, METRICS_PATH, REPORT_DIR
from src.inference import predict_probabilities


def evaluate_model(model, images: np.ndarray, labels: np.ndarray) -> dict:
    """Evaluate against supplied official test data and persist genuine metrics."""
    test_images = images.astype(np.float32) / 255.0
    probabilities = predict_probabilities(model, test_images[..., np.newaxis])
    predicted = probabilities.argmax(axis=1)
    report = classification_report(
        labels,
        predicted,
        labels=np.arange(26),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predicted, labels=np.arange(26))
    misclassified = np.flatnonzero(predicted != labels)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    examples_path = REPORT_DIR / "misclassified_examples.npz"
    np.savez_compressed(
        examples_path,
        images=images[misclassified[:100]],
        true_labels=labels[misclassified[:100]],
        predicted_labels=predicted[misclassified[:100]],
    )

    metrics = {
        "evaluation_split": "official_emnist_letters_test",
        "sample_count": int(len(labels)),
        "accuracy": float(report["accuracy"]),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "misclassified_examples": str(examples_path.name),
    }
    Path(METRICS_PATH).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics