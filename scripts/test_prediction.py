"""Load a saved model and run one genuine sample prediction."""

import numpy as np

from config import RAW_DATA_DIR
from src.dataset import load_emnist
from src.inference import load_model, predict_top_k


if __name__ == "__main__":
    _, _, test_images, _ = load_emnist(RAW_DATA_DIR)
    sample = test_images[0].astype(np.float32)[..., np.newaxis] / 255.0
    print(predict_top_k(load_model(), sample))