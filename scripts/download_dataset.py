"""Download and verify the official NIST EMNIST Letters IDX files."""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

from config import RAW_DATA_DIR
from src.dataset import load_emnist

DATASET_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
REQUIRED_FILES = (
    "emnist-letters-train-images-idx3-ubyte.gz",
    "emnist-letters-train-labels-idx1-ubyte.gz",
    "emnist-letters-test-images-idx3-ubyte.gz",
    "emnist-letters-test-labels-idx1-ubyte.gz",
)


def download_dataset(force: bool = False) -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if all((RAW_DATA_DIR / name).is_file() for name in REQUIRED_FILES) and not force:
        load_emnist(RAW_DATA_DIR)
        print(f"Verified existing EMNIST Letters files in {RAW_DATA_DIR}")
        return

    archive_path = RAW_DATA_DIR / "emnist-gzip.zip.part"
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "GlyphSenseAI/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, archive_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        with zipfile.ZipFile(archive_path) as archive:
            members = {Path(info.filename).name: info for info in archive.infolist()}
            missing = [name for name in REQUIRED_FILES if name not in members]
            if missing:
                raise ValueError(f"Official archive is missing required files: {missing}")
            for name in REQUIRED_FILES:
                destination = RAW_DATA_DIR / name
                with archive.open(members[name]) as source, destination.open("wb") as output:
                    output.write(source.read())
    except Exception:
        for name in REQUIRED_FILES:
            (RAW_DATA_DIR / name).unlink(missing_ok=True)
        raise
    finally:
        archive_path.unlink(missing_ok=True)

    train_images, train_labels, test_images, test_labels = load_emnist(RAW_DATA_DIR)
    print(
        "Verified official EMNIST Letters: "
        f"train={train_images.shape}/{train_labels.shape}, "
        f"test={test_images.shape}/{test_labels.shape}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Download again even if data already exists")
    download_dataset(parser.parse_args().force)