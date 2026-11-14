
import numpy as np
import pandas as pd

from forecast.ingest import ALL


def daily_fixture():
    dates = pd.date_range("2018-01-01", periods=405)
    rows = []
    for name, multiplier in [(ALL, 3), ("Canada", 1), ("United Kingdom", 2)]:
        activity = np.arange(len(dates)) % 7 + 3
        rows.append(pd.DataFrame({
            "date": dates, "country": name, "revenue": activity * 10. * multiplier,
            "views": activity * multiplier, "lines": activity * multiplier,
            "invoices": activity * multiplier, "streams": activity * multiplier,
        }))
    return pd.concat(rows).set_index(["date", "country"]).sort_index()


def test_config(root):
    return {
        "TESTING": True, "ADMIN_TOKEN": "fixture-token", "TRAIN_DATA": root / "train",
        "MODEL_SEED": root / "seed.joblib", "MODEL_PATH": root / "state/model.joblib",
        "LOG_PATH": root / "state/events.sqlite3", "OBSERVATIONS": None,
    }
