#!/bin/sh
set -eu
image="${1:-aavail-forecast:1.0}"
cid="$(docker run -d -p 127.0.0.1::8000 -e FORECAST_ADMIN_TOKEN=smoke-test "$image")"
trap 'docker rm -fv "$cid" >/dev/null' EXIT
address="$(docker port "$cid" 8000/tcp)"
python3 - "$address" <<'PY'
import json
import sys
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

origin = "http://" + sys.argv[1]
for _ in range(45):
    try:
        with urlopen(origin + "/health", timeout=2) as response:
            assert json.load(response)["status"] == "ready"
        break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit("container failed readiness")

for country, expected in (("all", "All Countries"), ("United Kingdom", "United Kingdom")):
    with urlopen(origin + "/predict?" + urlencode({"country": country, "date": "2019-08-01"})) as response:
        result = json.load(response)
    assert result["country"] == expected and result["days"] == 30
    assert result["window_end"] == "2019-08-30"

request = Request(origin + "/logfile?kind=prediction", headers={"Authorization": "Bearer smoke-test"})
with urlopen(request) as response:
    assert len(json.load(response)["events"]) == 2
print("Container readiness, country and aggregate predictions, and journal verified")
PY
