"""Shared normalization for uploaded images and drawing-canvas input."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

INPUT_SHAPE = (28, 28, 1)


def preprocess_image(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert one dark-on-light or light-on-dark character to a normalized tensor."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a Pillow image or NumPy array")

    grayscale = ImageOps.grayscale(image)
    pixels = np.asarray(grayscale, dtype=np.uint8)
    if pixels.size == 0:
        raise ValueError("Image contains no pixels")

    # Canonical input uses a black background and a bright character, like EMNIST.
    if float(pixels.mean()) > 127:
        pixels = 255 - pixels
    foreground = pixels > max(20, int(pixels.max() * 0.15))
    if not np.any(foreground):
        raise ValueError("No handwritten character was detected")

    ys, xs = np.where(foreground)
    cropped = Image.fromarray(pixels[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    width, height = cropped.size
    scale = 20.0 / max(width, height)
    resized = cropped.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("L", (28, 28), color=0)
    canvas.paste(resized, ((28 - resized.width) // 2, (28 - resized.height) // 2))
    normalized = np.asarray(canvas, dtype=np.float32) / 255.0
    return normalized[..., np.newaxis]