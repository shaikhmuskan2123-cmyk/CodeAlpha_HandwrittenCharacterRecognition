import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.preprocessing import INPUT_SHAPE, preprocess_image


def test_preprocessing_returns_normalized_model_input():
    image = Image.new("RGB", (80, 80), "white")
    ImageDraw.Draw(image).line((20, 60, 60, 20), fill="black", width=8)

    result = preprocess_image(image)

    assert result.shape == INPUT_SHAPE
    assert result.dtype == np.float32
    assert 0.0 <= result.min() <= result.max() <= 1.0
    assert result.max() > 0.5


def test_preprocessing_rejects_empty_canvas():
    with pytest.raises(ValueError, match="No handwritten character"):
        preprocess_image(Image.new("RGB", (64, 64), "white"))