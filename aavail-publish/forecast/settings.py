"""Deployment paths and runtime settings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def settings():
    state = Path(os.environ.get("FORECAST_STATE", ROOT / "runtime"))
    return {
        "TRAIN_DATA": Path(os.environ.get("FORECAST_TRAIN_DATA", ROOT / "data/cs-train")),
        "OBSERVATIONS": Path(os.environ["FORECAST_OBSERVATIONS"]) if os.environ.get("FORECAST_OBSERVATIONS") else None,
        "MODEL_SEED": Path(os.environ.get("FORECAST_MODEL_SEED", ROOT / "artifacts/model.joblib")),
        "MODEL_PATH": state / "model.joblib",
        "LOG_PATH": state / "events.sqlite3",
        "ADMIN_TOKEN": os.environ.get("FORECAST_ADMIN_TOKEN", ""),
        "MAX_CONTENT_LENGTH": 16 * 1024,
    }
