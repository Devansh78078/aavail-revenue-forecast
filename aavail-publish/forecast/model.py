"""Fit and serve direct 30-day revenue forecasts."""
import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .ingest import ALL

HORIZON = 30
WARMUP = 90
DAY = pd.Timedelta(days=1)
BASELINE = "trailing_30"
CANDIDATES = (BASELINE, "seasonal_naive", "ridge", "ridge_activity", "forest", "forest_activity")
ACTIVITY = ["lines_30", "views_30", "invoices_30"]


class ForecastError(ValueError):
    """A request is outside the supported forecast range or scopes."""


def design(series):
    """Index rows by forecast start. Every predictor uses earlier observations."""
    origin = series.reindex(series.index.union([series.index[-1] + DAY]))
    revenue = origin["revenue"]
    features = pd.DataFrame(index=origin.index)
    for window in (7, 14, 30, 60, 90):
        features[f"revenue_{window}"] = revenue.shift(1).rolling(window).sum()
    for column in ("lines", "views", "invoices"):
        features[f"{column}_30"] = origin[column].shift(1).rolling(30).sum()
    seasonal = features["revenue_30"].shift(335)
    features["seasonal"] = seasonal.fillna(features["revenue_30"])
    angle = 2 * np.pi * (origin.index.dayofyear - 1) / 365.25
    features["year_sin"] = np.sin(angle)
    features["year_cos"] = np.cos(angle)
    features["target"] = revenue.rolling(HORIZON).sum().shift(1-HORIZON)
    return features.iloc[WARMUP:]


def fit_candidate(name, rows):
    if name in (BASELINE, "seasonal_naive"):
        return None
    if name.startswith("ridge"):
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=12.0))
    else:
        estimator = RandomForestRegressor(
            n_estimators=64, max_depth=5, min_samples_leaf=8, random_state=29, n_jobs=1
        )
    return estimator.fit(_inputs(name, rows), rows["target"])


def _inputs(name, rows):
    drop = ["target"] + ([] if name.endswith("_activity") else ACTIVITY)
    return rows.drop(columns=drop)


def estimate(name, estimator, rows):
    if name in (BASELINE, "seasonal_naive"):
        return rows["revenue_30" if name == BASELINE else "seasonal"].to_numpy()
    return estimator.predict(_inputs(name, rows))


def summarize(frame):
    actual = frame["actual"].to_numpy()
    residual = (frame["prediction"] - frame["actual"]).to_numpy()
    absolute = float(np.abs(residual).sum())
    magnitude = float(np.abs(actual).sum())
    return {
        "origins": len(frame),
        "actual_abs_total": magnitude,
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "wape": (absolute / magnitude) if magnitude else None,
        "bias": float(residual.mean()),
    }


def fit(daily):
    """Choose a model per scope with expanding-window validation."""
    histories = {name: group.droplevel("country") for name, group in daily.groupby(level="country")}
    end = daily.index.get_level_values("date").max()
    if len(histories[ALL]) < 360:
        raise ValueError("training requires at least 360 calendar days")
    # Each fold evaluates a 30-day block of daily prediction starts.
    anchors = [end - pd.Timedelta(days=offset) for offset in (148, 103, 58)]
    results, sales = [], {}
    for country, series in histories.items():
        rows = design(series).dropna(subset=["target"])
        settings = CANDIDATES if series["revenue"].ne(0).sum() >= 25 else (BASELINE, "seasonal_naive")
        for name in settings:
            for number, start in enumerate(anchors, 1):
                training = rows.loc[:start - HORIZON * DAY]
                validation = rows.loc[start:start + 29 * DAY]
                estimator = fit_candidate(name, training)
                values = estimate(name, estimator, validation)
                results.append(pd.DataFrame({
                    "date": validation.index, "country": country, "model": name, "fold": number,
                    "prediction": values, "actual": validation["target"].to_numpy(),
                }))
        trials = pd.concat(results[-len(settings) * len(anchors):])
        mae = (trials["prediction"] - trials["actual"]).abs().groupby(trials["model"]).mean()
        winner = min(settings, key=mae.__getitem__)
        sales[country] = {
            "name": winner, "estimator": fit_candidate(winner, rows),
            "validation_mae": float(mae[winner]), "baseline_mae": float(mae[BASELINE]),
        }
    digest = hashlib.sha256(pd.util.hash_pandas_object(daily).to_numpy().tobytes()).hexdigest()[:12]
    trained_at = datetime.now(timezone.utc).isoformat()
    version = f"{end:%Y%m%d}-{digest}-{datetime.fromisoformat(trained_at):%Y%m%dT%H%M%S%fZ}"
    return {
        "version": version, "trained_at": trained_at, "through": end, "models": sales, "daily": daily,
    }, pd.concat(results, ignore_index=True)


def save_model(bundle, path):
    """Publish atomically. Readers see the old or new complete bundle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temporary = Path(temp.name)
    try:
        joblib.dump(bundle, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_model(path):
    return joblib.load(path)


def resolve_country(bundle, country):
    key = " ".join(country.strip().replace("_", " ").casefold().split())
    if key in ("all", "worldwide"):
        return ALL
    lookup = {name.casefold(): name for name in bundle["models"]}
    if key not in lookup:
        raise ForecastError("unknown country; use /countries for supported names")
    return lookup[key]


def combined_history(bundle, observations=None):
    daily = bundle["daily"]
    if observations is None:
        return daily
    future = observations.loc[observations.index.get_level_values("date") > bundle["through"]]
    if future.empty:
        return daily
    dates = future.index.get_level_values("date")
    if dates.min() != bundle["through"] + DAY or (dates.max() - dates.min()).days + 1 != dates.nunique():
        raise ValueError("observations must cover consecutive dates immediately after training coverage")
    return pd.concat([daily, future]).sort_index()


def forecast(bundle, history, country, start=None):
    scope = resolve_country(bundle, country)
    available = history.index.get_level_values("date").max() + DAY
    origin = pd.Timestamp(start) if start else available
    earliest = bundle["through"] + DAY
    if not earliest <= origin <= available:
        raise ForecastError(f"date must be between {earliest.date()} and {available.date()}")
    series = history.xs(scope, level="country").loc[:origin - DAY]
    calendar = pd.date_range(series.index[0], origin - DAY)
    row = design(series.reindex(calendar, fill_value=0)).loc[[origin]]
    model = bundle["models"][scope]
    predicted = estimate(model["name"], model["estimator"], row)[0]
    return {
        "country": scope, "date": str(origin.date()),
        "window_end": str((origin + (HORIZON - 1) * DAY).date()),
        "days": HORIZON, "prediction": round(float(predicted), 2),
        "baseline_prediction": round(float(estimate(BASELINE, None, row)[0]), 2),
        "model": model["name"], "version": bundle["version"],
        "trained_through": str(bundle["through"].date()),
        "observed_through": str((origin - DAY).date()),
    }
