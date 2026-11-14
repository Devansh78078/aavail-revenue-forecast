"""Flask endpoints for 30-day revenue forecasts and operations."""
from datetime import date
from functools import wraps
import hmac
from pathlib import Path
from threading import Lock
from time import perf_counter

from flask import Flask, g, jsonify, request
import numpy as np
from werkzeug.exceptions import HTTPException

from .events import EventLog
from .ingest import ingest
from .model import ForecastError, combined_history, fit, forecast, load_model, save_model
from .settings import settings


def create_app(overrides=None):
    app = Flask(__name__)
    app.config.update(settings())
    app.config.update(overrides or {})
    log = EventLog(app.config["LOG_PATH"])
    publication_lock, training_lock = Lock(), Lock()
    active_path = Path(app.config["MODEL_PATH"])
    seed = Path(app.config["MODEL_SEED"])
    cache = {"signature": None, "bundle": None, "history": None}

    def load_current():
        with publication_lock:
            path = active_path if active_path.is_file() else seed
            if not path.is_file():
                return None, None
            stat = path.stat()
            configured = app.config["OBSERVATIONS"]
            files = []
            if configured:
                for source in sorted(Path(configured).glob("*.json")):
                    revision = source.stat()
                    files.append((source.name, revision.st_ino, revision.st_mtime_ns, revision.st_size))
            signature = (path, stat.st_ino, stat.st_mtime_ns, stat.st_size, tuple(files))
            if signature != cache["signature"]:
                bundle = load_model(path)
                observations = ingest(configured) if configured else None
                history = combined_history(bundle, observations)
                cache.update(signature=signature, bundle=bundle, history=history)
            return cache["bundle"], cache["history"]

    def operator_only(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            expected = app.config["ADMIN_TOKEN"]
            if not expected:
                return jsonify(error="operator endpoints are disabled"), 503
            given = request.headers.get("Authorization", "")
            if not hmac.compare_digest(given.encode(), f"Bearer {expected}".encode()):
                return jsonify(error="operator token required"), 401
            return function(*args, **kwargs)
        return wrapped

    @app.before_request
    def start_timer():
        g.started = perf_counter()

    @app.after_request
    def record_request(response):
        if request.endpoint not in ("health", "metrics", "logfile"):
            log.write(
                "request", endpoint=request.endpoint or "unmatched", method=request.method,
                status=response.status_code, elapsed_ms=round((perf_counter()-g.started)*1000, 3),
            )
        return response

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    @app.get("/")
    def index():
        return jsonify(
            service="AAVAIL Revenue Forecast", horizon_days=30,
            endpoints=["GET /health", "GET /countries", "GET|POST /predict", "POST /train",
                       "GET /metrics", "GET /logfile"],
        )

    @app.get("/health")
    def health():
        bundle, history = load_current()
        if bundle is None:
            return jsonify(status="unavailable"), 503
        return jsonify(status="ready", version=bundle["version"],
                       trained_through=str(bundle["through"].date()),
                       observed_through=str(history.index.get_level_values("date").max().date()))

    @app.get("/countries")
    def countries():
        bundle, _ = load_current()
        if bundle is None:
            return jsonify(error="model unavailable"), 503
        return jsonify(countries=sorted(bundle["models"]))

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        inputs = request.args if request.method == "GET" else request.get_json()
        if request.method == "POST" and not isinstance(inputs, dict):
            return jsonify(error="expected a JSON object"), 400
        country = inputs.get("country", "all")
        start = inputs.get("date")
        if not isinstance(country, str) or not country.strip():
            return jsonify(error="country must be a nonempty string"), 400
        if start is not None:
            try:
                if date.fromisoformat(start).isoformat() != start:
                    raise ValueError
            except (TypeError, ValueError):
                return jsonify(error="date must use YYYY-MM-DD"), 400
        bundle, history = load_current()
        if bundle is None:
            return jsonify(error="model unavailable"), 503
        try:
            result = forecast(bundle, history, country, start)
        except ForecastError as exc:
            return jsonify(error=str(exc)), 422
        log.write("prediction", **result)
        return jsonify(result)

    @app.post("/train")
    @operator_only
    def train():
        if not training_lock.acquire(blocking=False):
            return jsonify(error="training already in progress"), 409
        try:
            bundle, _ = fit(ingest(app.config["TRAIN_DATA"]))
            configured = app.config["OBSERVATIONS"]
            if configured:
                combined_history(bundle, ingest(configured))
            save_model(bundle, active_path)
            log.write("training", model_version=bundle["version"], countries=len(bundle["models"]))
        except ValueError as exc:
            log.write("training_error", reason=str(exc))
            return jsonify(error="training data failed validation; see the operator log"), 422
        finally:
            training_lock.release()
        return jsonify(version=bundle["version"], trained_through=str(bundle["through"].date()),
                       countries=len(bundle["models"]))

    @app.get("/logfile")
    @operator_only
    def logfile():
        limit = request.args.get("limit", "100")
        try:
            limit = int(limit)
            if not 1 <= limit <= 1000:
                raise ValueError
        except ValueError:
            return jsonify(error="limit must be between 1 and 1000"), 400
        return jsonify(events=log.read(kind=request.args.get("kind"), limit=limit))

    @app.get("/metrics")
    @operator_only
    def metrics():
        entries = log.read(kind="request", limit=1000)
        timings = [e["elapsed_ms"] for e in entries]
        return jsonify(
            sample_size=len(entries),
            error_5xx=sum(e["status"] >= 500 for e in entries),
            error_4xx=sum(400 <= e["status"] < 500 for e in entries),
            p95_elapsed_ms=round(float(np.percentile(timings, 95)), 3) if timings else None,
        )

    # Fail at startup if a configured source or artifact is invalid.
    load_current()
    return app
