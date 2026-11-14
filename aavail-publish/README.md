# AAVAIL 30-Day Revenue Forecasting

A Flask service that forecasts revenue for the 30 days starting on a chosen date, for an individual country or the worldwide total. The package includes the implementation, a public data download script, trained model, isolated tests, model comparison charts, and a historical production replay.

## Run with Docker

From the repository root, fetch the course dataset, build the image, and start the service.

```bash
python scripts/fetch_data.py
docker build -t aavail-forecast:1.0 .
docker run --rm -p 127.0.0.1:8000:8000 \
  --mount source=aavail-state,target=/var/lib/forecast \
  aavail-forecast:1.0
```

If you downloaded the separate prebuilt image for Linux x86-64, replace the build step with `docker load -i aavail-forecast-linux-amd64.tar`. The image starts with the included model. The named volume retains the active model and SQLite journal between container runs.

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/countries
curl -fsS 'http://127.0.0.1:8000/predict?country=all&date=2019-08-01'
curl -fsS -G http://127.0.0.1:8000/predict \
  --data-urlencode 'country=United Kingdom' --data-urlencode 'date=2019-08-01'
curl -fsS http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"country":"France","date":"2019-08-01"}'
```

The default country is `all`. Country names are case-insensitive and accept underscores in place of spaces. Dates use `YYYY-MM-DD`, and omitting the date starts the forecast on the day after the latest observation. The requested date is the first day in the 30-day window. The seed covers history through July 31, 2019. Its first forecast starts August 1. Append later observations to enable additional dates, including mid-month starts. A forecast uses transactions through the day before its start. The worldwide forecast is trained on aggregate daily revenue and can differ from the sum of country forecasts.

### Operator endpoints

Set `FORECAST_ADMIN_TOKEN` to a secret in your environment and pass `-e FORECAST_ADMIN_TOKEN` in the Docker command. Operator calls use a bearer token.

| Endpoint | Purpose |
| --- | --- |
| `POST /train` | Retrain from `FORECAST_TRAIN_DATA` and atomically publish a new version. |
| `GET /logfile?kind=prediction&limit=10` | Retrieve recent structured events. Omit `kind` to include all event types. |
| `GET /metrics` | Read p95 latency and 4xx/5xx counts over the latest 1,000 recorded requests. |

```bash
curl -fsS -X POST http://127.0.0.1:8000/train \
  -H "Authorization: Bearer $FORECAST_ADMIN_TOKEN"
curl -fsS 'http://127.0.0.1:8000/logfile?kind=prediction&limit=10' \
  -H "Authorization: Bearer $FORECAST_ADMIN_TOKEN"
```

Mount directories of trusted monthly JSON exports read-only and point `FORECAST_TRAIN_DATA` and optionally `FORECAST_OBSERVATIONS` at their container paths. Stage each complete export under a temporary extension, then rename it to `.json`. Each source file must have a unique name and a non-overlapping set of dates. Observation coverage must start immediately after training coverage. The service reloads observations when files change. Treat data and the serialized model as operator-managed inputs. The default deployment binds on loopback and uses one Gunicorn worker with four threads, allowing prediction traffic during training. For a shared deployment, place the service behind your authenticated TLS ingress and use a persistent local volume for the SQLite journal.

## Develop and reproduce

Python 3.12 is used by the image. The analysis extras add Matplotlib.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python scripts/fetch_data.py
sh run_tests.sh
python -m forecast.ingest data/cs-train --output runtime/daily_training.csv
python -m forecast.train data/cs-train --model artifacts/model.joblib \
  --validation reports/validation_predictions.csv
sh scripts/replay_demo.sh reports
python -m forecast.figures
```

`sh run_tests.sh` runs all 21 tests. The API tests use a Flask client, synthetic transaction histories, and temporary model and journal paths. Model and journal persistence tests use their own temporary directories. The same suite runs during the Docker build and can be rerun with `docker run --rm aavail-forecast:1.0 ./run_tests.sh`. The included CI workflow also runs `./scripts/docker_smoke.sh` on a Docker host.

`sh scripts/replay_demo.sh` starts a temporary Gunicorn instance and generates a prediction for each of six scopes on every day from August 1 to December 7, 2019. It uses the seed trained through July 31, supplies only the preceding observations to each forecast, and deletes its temporary journal after exporting the reports. To run monitoring against an existing service journal and new actuals, use:

```bash
python -m forecast.monitor --logs runtime/events.sqlite3 \
  --actuals data/cs-train data/cs-production --output runtime/monitoring
```

For Docker, mount actuals at `/actuals` and run the same command inside the container using `/var/lib/forecast/events.sqlite3` and an output directory under `/var/lib/forecast`. Automate the command after each actuals refresh. It scores only complete 30-day windows, de-duplicates forecast retries, and writes per-version and per-country metrics. Use `--version` to filter a single deployment. The JSON flags scopes where MAE exceeds the baseline by over 10% after at least 30 mature forecasts. Monitor request errors and latency with `/metrics`.

## Peer review checklist

| Rubric | Evidence |
| --- | --- |
| API unit tests | [tests/test_api.py](tests/test_api.py) |
| Model unit tests | [tests/test_model.py](tests/test_model.py) |
| Logging unit tests | [tests/test_events.py](tests/test_events.py) |
| All tests via one script | [run_tests.sh](run_tests.sh) and [test output](reports/image_tests.txt) |
| Performance monitoring | [forecast/monitor.py](forecast/monitor.py), `GET /metrics`, [replay metrics](reports/monitoring_metrics.json) |
| Isolated reads and writes | Test fixtures and temporary paths in [tests/](tests/) |
| Country and worldwide API | `GET/POST /predict` and [HTTP verification](reports/image_http_smoke.json) |
| Automated data ingestion | [forecast/ingest.py](forecast/ingest.py) |
| Multiple models compared | [validation results](reports/validation_predictions.csv) |
| Visual EDA | [training EDA](reports/training_eda.png) |
| Container with model, service and tests | [Dockerfile](Dockerfile), [container CI](.github/workflows/container.yml), and the prebuilt image |
| Visual comparison against baseline | [validation comparison](reports/validation_comparison.png) and [production replay](reports/production_replay.png) |

See the [capstone report](reports/CAPSTONE_REPORT.md) for the business case, model selection, and holdout results. The original invoice files are available from the [public IBM AI Enterprise Workflow capstone repository](https://github.com/aavail/ai-workflow-capstone). The download script pins the source revision and leaves the raw exports in the ignored local `data/` folder.
