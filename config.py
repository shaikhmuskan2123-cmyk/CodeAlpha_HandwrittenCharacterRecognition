"""Project paths and reproducible training defaults."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
ASSET_DIR = PROJECT_ROOT / "assets"

MODEL_PATH = MODEL_DIR / "glyphsense_emnist_letters.keras"
METRICS_PATH = REPORT_DIR / "evaluation_metrics.json"
HISTORY_PATH = REPORT_DIR / "training_history.json"
CONFIG_PATH = REPORT_DIR / "model_config.json"
CLASS_MAP_PATH = REPORT_DIR / "class_mapping.json"

CLASS_NAMES = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
RANDOM_SEED = 42
VALIDATION_SIZE = 0.1
BATCH_SIZE = 128
EPOCHS = 30

for directory in (DATA_DIR, RAW_DATA_DIR, MODEL_DIR, REPORT_DIR, ASSET_DIR):
    directory.mkdir(parents=True, exist_ok=True)