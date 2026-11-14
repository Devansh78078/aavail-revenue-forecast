# AAVAIL 30-Day Revenue Forecasting

IBM AI Enterprise Workflow capstone. The Flask API forecasts the next 30 days of revenue for individual countries and for all countries combined. The project includes a trained model, a Docker build, isolated unit tests, performance monitoring, and a production replay.

## Peer review

| Requirement | Evidence |
| --- | --- |
| Setup and API examples | [Project README](aavail-publish/README.md) |
| API, model, logging, and ingestion tests | [Test suite](aavail-publish/tests) and [test output](aavail-publish/reports/image_tests.txt) |
| Automated ingestion and data setup | [Ingestion module](aavail-publish/forecast/ingest.py) and [course dataset download](aavail-publish/scripts/fetch_data.py) |
| Country and worldwide prediction | [HTTP smoke test evidence](aavail-publish/reports/image_http_smoke.json) |
| Performance monitoring | [Monitoring script](aavail-publish/forecast/monitor.py) and [replay metrics](aavail-publish/reports/monitoring_metrics.json) |
| EDA and model comparisons | [Capstone report](aavail-publish/reports/CAPSTONE_REPORT.md), [validation comparison](aavail-publish/reports/validation_comparison.png), and [production replay](aavail-publish/reports/production_replay.png) |
| Container deployment | [Dockerfile](aavail-publish/Dockerfile) and [smoke test](aavail-publish/scripts/docker_smoke.sh) |

## Run the service

From a checkout of this repository, use the project directory as the Docker build context.

```bash
cd aavail-publish
python scripts/fetch_data.py
docker build -t aavail-forecast:1.0 .
docker run --rm -p 127.0.0.1:8000:8000 aavail-forecast:1.0
```

The [original IBM course data](https://github.com/aavail/ai-workflow-capstone) is downloaded on demand at a pinned revision. All 21 tests passed. The image filesystem and service were exercised in isolated Linux namespaces. Docker daemon execution remains to be verified on a Docker host.
