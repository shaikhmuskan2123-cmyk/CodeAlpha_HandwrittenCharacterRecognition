"""Model loading and probability-validated character inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config import CLASS_NAMES, MODEL_PATH


def load_model(model_path: str | Path = MODEL_PATH):
    """Load a trained Keras model, reporting a clear error when it is unavailable."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Download EMNIST and run scripts/train_model.py first."
        )
    import tensorflow as tf

    return tf.keras.models.load_model(path)


def predict_probabilities(model, image_batch: np.ndarray) -> np.ndarray:
    """Return validated (batch, 26) softmax output from a Keras-compatible model."""
    batch = np.asarray(image_batch, dtype=np.float32)
    if batch.ndim != 4 or batch.shape[1:] != (28, 28, 1):
        raise ValueError(f"Expected input shape (batch, 28, 28, 1), received {batch.shape}")
    probabilities = np.asarray(model.predict(batch, verbose=0), dtype=np.float32)
    if probabilities.shape != (len(batch), len(CLASS_NAMES)):
        raise ValueError(f"Expected model output shape (batch, 26), received {probabilities.shape}")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Model returned invalid class probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError("Model output probabilities do not sum to one")
    return probabilities


def predict_top_k(model, image: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
    """Return the highest-probability classes for one preprocessed character."""
    if not 1 <= k <= len(CLASS_NAMES):
        raise ValueError("k must be between 1 and 26")
    probabilities = predict_probabilities(model, np.asarray(image)[np.newaxis, ...])[0]
    indices = np.argsort(probabilities)[::-1][:k]
    return [(CLASS_NAMES[index], float(probabilities[index])) for index in indices]