"""Evaluate the trained model on official held-out EMNIST test data."""

from config import RAW_DATA_DIR
from src.dataset import load_emnist
from src.evaluation import evaluate_model
from src.inference import load_model


if __name__ == "__main__":
    _, _, test_images, test_labels = load_emnist(RAW_DATA_DIR)
    metrics = evaluate_model(load_model(), test_images, test_labels)
    print(f"Official test accuracy: {metrics['accuracy']:.4f}")