"""Match logged forecasts to newly matured actuals, then report model error."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .events import EventLog
from .ingest import ingest
from .model import DAY, HORIZON, summarize


def score(log, actuals, version=None):
    end = actuals.index.get_level_values("date").max()
    start = actuals.index.get_level_values("date").min()
    revenue = actuals["revenue"].unstack("country")
    # Retried requests share one model version, date and scope.
    seen, matched = set(), []
    pending, missing = 0, 0
    for event in log.read(kind="prediction"):
        key = (event["version"], event["country"], event["date"])
        if key in seen or (version and event["version"] != version):
            continue
        seen.add(key)
        origin = pd.Timestamp(event["date"])
        window_end = origin + (HORIZON - 1) * DAY
        if window_end > end:
            pending += 1
        elif origin < start or event["country"] not in revenue:
            missing += 1
        else:
            actual = revenue.loc[origin:window_end, event["country"]]
            if len(actual) != HORIZON or actual.isna().any():
                missing += 1
                continue
            matched.append({
                "date": event["date"], "window_end": str(window_end.date()), "country": event["country"],
                "version": event["version"], "model": event["model"], "prediction": event["prediction"],
                "baseline": event["baseline_prediction"], "actual": round(float(actual.sum()), 2),
            })
    columns = ["date", "window_end", "country", "version", "model", "prediction", "baseline", "actual"]
    return pd.DataFrame(matched, columns=columns).sort_values(["version", "country", "date"]), {
        "actuals_through": str(end.date()), "matched": len(matched), "pending": pending, "missing": missing,
    }


def metrics_report(matched, coverage, threshold):
    results = []
    for (version, country), group in matched.groupby(["version", "country"]):
        selected = summarize(group)
        baseline = summarize(group.assign(prediction=group["baseline"]))
        # A zero baseline MAE is a valid perfect forecast.
        exceeds_baseline = len(group) >= 30 and selected["mae"] > (1 + threshold) * baseline["mae"]
        results.append(dict(
            version=version, country=country, model=group["model"].iloc[0],
            selected=selected, baseline=baseline, exceeds_baseline=exceeds_baseline,
        ))
    return {**coverage, "relative_mae_alert_threshold": threshold, "scopes": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--actuals", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version")
    parser.add_argument("--mae-alert-threshold", type=float, default=0.10)
    args = parser.parse_args()
    if not args.logs.is_file():
        parser.error("log database does not exist")
    if not 0 <= args.mae_alert_threshold < float("inf"):
        parser.error("alert threshold must be finite and nonnegative")
    matched, coverage = score(EventLog(args.logs), ingest(*args.actuals), args.version)
    args.output.mkdir(parents=True, exist_ok=True)
    matched.to_csv(args.output / "scored_predictions.csv", index=False)
    result = metrics_report(matched, coverage, args.mae_alert_threshold)
    (args.output / "monitoring_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(coverage))


if __name__ == "__main__":
    main()
