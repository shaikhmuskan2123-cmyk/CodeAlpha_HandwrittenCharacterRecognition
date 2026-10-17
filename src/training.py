"""Reproducible full and smoke-test model training."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from config import (
    BATCH_SIZE,
        CLASS_MAP_PATH,
    CONFIG_PATH,
    EPOCHS,
    HISTORY_PATH,
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_SEED,
    RAW_DATA_DIR,
)
from src.dataset import load_emnist, split_train_validation
from src.model import create_model


def serialize_history(history: dict) -> str:
    """Convert Keras history values to JSON-compatible native floats."""
    serializable_history = {
        metric: [float(value) for value in values]
        for metric, values in history.items()
    }
    return json.dumps(serializable_history, indent=2)


def train(smoke_test: bool = False) -> Path:
    """Train on official EMNIST training data; smoke tests use a separate artifact."""
    import tensorflow as tf

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    train_images, train_labels, _, _ = load_emnist(RAW_DATA_DIR)
    if smoke_test:
        rng = np.random.default_rng(RANDOM_SEED)
        selected = np.concatenate(
            [rng.choice(np.flatnonzero(train_labels == label), size=40, replace=False) for label in range(26)]
        )
        train_images, train_labels = train_images[selected], train_labels[selected]

    x_train, x_validation, y_train, y_validation = split_train_validation(
        train_images, train_labels, validation_size=0.2 if smoke_test else 0.1
    )
    x_train = x_train.astype(np.float32)[..., np.newaxis] / 255.0
    x_validation = x_validation.astype(np.float32)[..., np.newaxis] / 255.0

    model_path = MODEL_DIR / "glyphsense_smoke_test.keras" if smoke_test else MODEL_PATH
    epochs = 2 if smoke_test else EPOCHS
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_loss", save_best_only=True),
    ]
    model = create_model()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    if not model_path.exists():
        model.save(model_path)
    if not smoke_test:
        HISTORY_PATH.write_text(serialize_history(history.history), encoding="utf-8")
        CLASS_MAP_PATH.write_text(
            json.dumps({str(index): letter for index, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}, indent=2),
            encoding="utf-8",
        )
        CONFIG_PATH.write_text(
            json.dumps(
                {
                    "architecture": "Conv2D-BatchNorm-ReLU-MaxPool x 2, Conv2D-BatchNorm-ReLU, GlobalAveragePooling, Dense",
                    "input_shape": [28, 28, 1],
                    "class_count": 26,
                    "seed": RANDOM_SEED,
                    "batch_size": BATCH_SIZE,
                    "epochs_requested": EPOCHS,
                    "epochs_completed": len(history.epoch),
                    "augmentation": ["random rotation", "random translation"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return Path(model_path)