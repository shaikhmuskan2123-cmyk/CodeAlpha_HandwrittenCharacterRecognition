import json

import numpy as np

from src.training import serialize_history


def test_training_history_values_are_json_serializable():
    history = {
        "accuracy": [np.float32(0.75), np.float64(0.8)],
        "loss": [np.float32(1.2), np.float64(0.9)],
    }
    restored = json.loads(serialize_history(history))

    assert restored["accuracy"] == [0.75, 0.8]
    assert restored["loss"] == [1.2000000476837158, 0.9]