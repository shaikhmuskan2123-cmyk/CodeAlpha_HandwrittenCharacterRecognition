"""Small shared helpers for report loading, timestamps, and CSV downloads."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: str | Path) -> dict | None:
    """Load a JSON artifact, returning None when it has not been created yet."""
    artifact = Path(path)
    if not artifact.is_file():
        return None
    with artifact.open(encoding="utf-8") as source:
        return json.load(source)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rows_to_csv(rows: list[dict]) -> str:
    """Serialize dictionaries to CSV for Streamlit download controls."""
    if not rows:
        return ""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()