import csv
import io

import pytest
from PIL import Image, ImageDraw

from pages.common import reject_multiple_characters
from src.utils import rows_to_csv


def test_csv_export_includes_headers_and_values():
    content = rows_to_csv([{"letter": "A", "confidence": 0.9}])
    rows = list(csv.DictReader(io.StringIO(content)))

    assert rows == [{"letter": "A", "confidence": "0.9"}]


def test_multiple_characters_are_rejected():
    image = Image.new("RGB", (160, 50), "white")
    drawing = ImageDraw.Draw(image)
    drawing.rectangle((20, 10, 40, 40), fill="black")
    drawing.rectangle((100, 10, 120, 40), fill="black")

    with pytest.raises(ValueError, match="multiple characters"):
        reject_multiple_characters(image)