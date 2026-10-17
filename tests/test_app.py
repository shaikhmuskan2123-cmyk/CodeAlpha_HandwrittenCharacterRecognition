from pathlib import Path

from streamlit.testing.v1 import AppTest

from config import METRICS_PATH
from src.utils import load_json


def test_all_navigation_pages_render_without_exceptions():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=60)
    assert not app.exception

    pages = [
        "Draw & Predict",
        "Upload & Predict",
        "Batch Recognition",
        "Model Analytics",
        "Prediction History",
        "About the Project",
        "Overview",
    ]
    for page in pages:
        app.radio[0].set_value(page).run(timeout=60)
        assert not app.exception, f"Page {page!r} raised: {app.exception}"


def test_analytics_letter_selector_changes_to_real_class_scores():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=60)
    app.radio[0].set_value("Model Analytics").run(timeout=60)

    selector = next(item for item in app.selectbox if item.label == "Explore test results by letter")
    expected = load_json(METRICS_PATH)["classification_report"]
    values = {}
    for letter in ("A", "I"):
        selector.set_value(letter).run(timeout=60)
        score = next(item.value for item in app.metric if item.label == f"{letter} F1")
        values[letter] = float(score.rstrip("%")) / 100
        assert abs(values[letter] - expected[letter]["f1-score"]) < 0.001

    assert values["A"] != values["I"]