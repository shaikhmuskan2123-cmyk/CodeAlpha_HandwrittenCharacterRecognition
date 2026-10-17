"""EMNIST Letters IDX loading and reproducible data splitting."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from config import RANDOM_SEED, VALIDATION_SIZE

IMAGE_MAGIC = 2051
LABEL_MAGIC = 2049
IMAGE_SHAPE = (28, 28)


def read_idx_images(path: str | Path) -> np.ndarray:
    """Read gzip-compressed IDX image data and correct EMNIST orientation."""
    with gzip.open(path, "rb") as source:
        header = source.read(16)
        if len(header) != 16:
            raise ValueError(f"Invalid IDX image header in {path}")
        magic, count, rows, columns = struct.unpack(">IIII", header)
        if magic != IMAGE_MAGIC or (rows, columns) != IMAGE_SHAPE:
            raise ValueError(f"Unexpected IDX image dimensions or magic number in {path}")
        pixels = source.read()

    expected_bytes = count * rows * columns
    if len(pixels) != expected_bytes:
        raise ValueError(f"Expected {expected_bytes} image bytes, received {len(pixels)}")

    images = np.frombuffer(pixels, dtype=np.uint8).reshape(count, rows, columns)
    return np.transpose(images, (0, 2, 1)).copy()


def read_idx_labels(path: str | Path) -> np.ndarray:
    """Read gzip-compressed EMNIST Letters labels and map 1..26 to 0..25."""
    with gzip.open(path, "rb") as source:
        header = source.read(8)
        if len(header) != 8:
            raise ValueError(f"Invalid IDX label header in {path}")
        magic, count = struct.unpack(">II", header)
        if magic != LABEL_MAGIC:
            raise ValueError(f"Unexpected IDX label magic number in {path}")
        raw_labels = source.read()

    if len(raw_labels) != count:
        raise ValueError(f"Expected {count} labels, received {len(raw_labels)}")
    labels = np.frombuffer(raw_labels, dtype=np.uint8).astype(np.int64) - 1
    if np.any((labels < 0) | (labels > 25)):
        raise ValueError("EMNIST Letters labels must be in the range 1..26")
    return labels


def load_emnist(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load official EMNIST Letters train and test IDX files from ``data_dir``."""
    root = Path(data_dir)
    train_images = read_idx_images(root / "emnist-letters-train-images-idx3-ubyte.gz")
    train_labels = read_idx_labels(root / "emnist-letters-train-labels-idx1-ubyte.gz")
    test_images = read_idx_images(root / "emnist-letters-test-images-idx3-ubyte.gz")
    test_labels = read_idx_labels(root / "emnist-letters-test-labels-idx1-ubyte.gz")

    if len(train_images) != len(train_labels) or len(test_images) != len(test_labels):
        raise ValueError("EMNIST image and label counts do not match")
    return train_images, train_labels, test_images, test_labels


def split_train_validation(
    images: np.ndarray,
    labels: np.ndarray,
    validation_size: float = VALIDATION_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create stratified train/validation partitions; never split official test data."""
    if len(images) != len(labels):
        raise ValueError("Image and label counts do not match")
    return train_test_split(
        images,
        labels,
        test_size=validation_size,
        random_state=seed,
        stratify=labels,
    )