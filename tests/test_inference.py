import numpy as np
import pytest

from src.inference import load_model, predict_probabilities, predict_top_k


class FixedModel:
    def predict(self, batch, verbose=0):
        result = np.zeros((len(batch), 26), dtype=np.float32)
        result[:, 3] = 0.7
        result[:, 4] = 0.2
        result[:, 5] = 0.1
        return result


def test_inference_returns_top_three_valid_scores():
    result = predict_top_k(FixedModel(), np.zeros((28, 28, 1), dtype=np.float32))

    assert [label for label, _ in result] == ["D", "E", "F"]
    assert all(0.0 <= confidence <= 1.0 for _, confidence in result)


def test_inference_rejects_wrong_input_dimensions():
    with pytest.raises(ValueError, match="Expected input shape"):
        predict_probabilities(FixedModel(), np.zeros((1, 32, 32, 1), dtype=np.float32))


def test_missing_model_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="Trained model not found"):
        load_model(tmp_path / "missing.keras")