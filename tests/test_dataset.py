import gzip
import struct

import numpy as np
import pytest

from src.dataset import read_idx_images, read_idx_labels


def test_idx_image_dimensions_and_orientation(tmp_path):
    path = tmp_path / "images.gz"
    image = np.arange(28 * 28, dtype=np.uint8).reshape(28, 28)
    with gzip.open(path, "wb") as output:
        output.write(struct.pack(">IIII", 2051, 1, 28, 28))
        output.write(image.tobytes())

    result = read_idx_images(path)

    assert result.shape == (1, 28, 28)
    np.testing.assert_array_equal(result[0], image.T)


def test_labels_are_zero_based(tmp_path):
    path = tmp_path / "labels.gz"
    with gzip.open(path, "wb") as output:
        output.write(struct.pack(">II", 2049, 3))
        output.write(bytes((1, 13, 26)))

    np.testing.assert_array_equal(read_idx_labels(path), np.array((0, 12, 25)))


def test_invalid_emnist_label_is_rejected(tmp_path):
    path = tmp_path / "labels.gz"
    with gzip.open(path, "wb") as output:
        output.write(struct.pack(">II", 2049, 1))
        output.write(bytes((0,)))

    with pytest.raises(ValueError, match="1..26"):
        read_idx_labels(path)