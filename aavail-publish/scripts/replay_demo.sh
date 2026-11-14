#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
out="${1:-reports}"
work="$(mktemp -d)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export FORECAST_STATE="$work/state" FORECAST_OBSERVATIONS="$PWD/data/cs-production"
export FORECAST_TRAIN_DATA="$PWD/data/cs-train" FORECAST_MODEL_SEED="$PWD/artifacts/model.joblib"
export FORECAST_ADMIN_TOKEN=replay-local NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
port="${REPLAY_PORT:-8765}"
python -m gunicorn --bind "127.0.0.1:$port" --workers 1 --threads 4 --timeout 180 'forecast.api:create_app()' > "$work/server.log" 2>&1 &
pid=$!
cleanup() {
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    rm -rf "$work"
}
trap cleanup EXIT
url="http://127.0.0.1:$port"
python - "$url" "$pid" "$work/server.log" <<'PY'
import os
from pathlib import Path
import sys
import time
from urllib.request import urlopen

url, pid, logs = sys.argv[1:]
for _ in range(45):
    try:
        os.kill(int(pid), 0)
        with urlopen(url + "/health", timeout=2) as response:
            if response.status == 200:
                break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit(Path(logs).read_text())
PY
python -m forecast.replay --url "$url" --from 2019-08-01 --to 2019-12-07 \
    --countries all 'United Kingdom' EIRE Germany France Norway
python -m forecast.monitor --logs "$work/state/events.sqlite3" --actuals data/cs-production --output "$out"
python - "$url" "$out" <<'PY'
from pathlib import Path
import sys
from urllib.request import Request, urlopen
request = Request(sys.argv[1] + "/metrics", headers={"Authorization": "Bearer replay-local"})
with urlopen(request) as response:
    (Path(sys.argv[2]) / "replay_api_metrics.json").write_bytes(response.read())
PY
