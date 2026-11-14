"""Normalize monthly invoice JSON files into daily metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ALL = "All Countries"
METRICS = ["revenue", "lines", "invoices", "streams", "views"]
ALIASES = {"total_price": "price", "StreamID": "stream_id", "TimesViewed": "times_viewed"}
REQUIRED = {"year", "month", "day", "country", "invoice", "price", "stream_id", "times_viewed"}


def _normalize(path):
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list) or not records or not all(isinstance(r, dict) for r in records):
            raise ValueError("expected a nonempty JSON array of records")
        frame = pd.DataFrame(records)
        for alias, canonical in ALIASES.items():
            if alias in frame:
                source = frame.pop(alias)
                if canonical in frame:
                    overlap = frame[canonical].notna() & source.notna()
                    if (frame.loc[overlap, canonical] != source.loc[overlap]).any():
                        raise ValueError(f"conflicting {alias} and {canonical} values")
                    frame[canonical] = frame[canonical].combine_first(source)
                else:
                    frame[canonical] = source
        missing = REQUIRED - set(frame)
        if missing:
            raise ValueError(f"missing columns {sorted(missing)}")
        frame["date"] = pd.to_datetime(frame[["year", "month", "day"]].apply(pd.to_numeric))
        frame["country"] = frame["country"].astype("string").str.strip()
        frame["invoice"] = frame["invoice"].astype("string").str.replace(r"\D+", "", regex=True)
        frame["stream_id"] = frame["stream_id"].astype("string").str.strip()
        for column in ("price", "times_viewed"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if not np.isfinite(frame[column]).all():
                raise ValueError(f"missing or nonfinite {column}")
        if frame["date"].isna().any() or (frame["times_viewed"] < 0).any():
            raise ValueError("missing date or negative view count")
        for column in ("country", "invoice", "stream_id"):
            if frame[column].isna().any() or frame[column].eq("").any():
                raise ValueError(f"missing or blank {column}")
        if frame["country"].str.casefold().eq(ALL.casefold()).any():
            raise ValueError("country uses the reserved aggregate name")
        return frame[["date", "country", "invoice", "stream_id", "price", "times_viewed"]]
    except (ValueError, TypeError, OSError) as exc:
        raise ValueError(f"{path.name}: {exc}") from exc


def _aggregate(frame, keys):
    return frame.groupby(keys).agg(
        revenue=("price", "sum"),
        lines=("price", "size"),
        invoices=("invoice", "nunique"),
        streams=("stream_id", "nunique"),
        views=("times_viewed", "sum"),
    )


def ingest(*directories):
    """Return a dense daily matrix indexed by date and country."""
    files = {}
    for directory in directories:
        paths = sorted(Path(directory).glob("*.json"))
        if not paths:
            raise ValueError(f"no JSON files in {directory}")
        for path in paths:
            if path.name in files:
                raise ValueError(f"duplicate source filename: {path.name}")
            files[path.name] = path
    if not files:
        raise ValueError("at least one source directory is required")
    daily_parts, global_parts = [], []
    dates_seen = set()
    for path in sorted(files.values()):
        frame = _normalize(path)
        dates = set(frame["date"].unique())
        if dates_seen & dates:
            raise ValueError(f"{path.name}: dates overlap another source file")
        dates_seen.update(dates)
        daily_parts.append(_aggregate(frame, ["date", "country"]))
        global_parts.append(_aggregate(frame, ["date"]).assign(country=ALL).set_index("country", append=True))
    daily = pd.concat(daily_parts + global_parts)
    dates = daily.index.get_level_values("date")
    countries = sorted(daily.index.get_level_values("country").unique())
    index = pd.MultiIndex.from_product(
        [pd.date_range(dates.min(), dates.max()), countries], names=["date", "country"]
    )
    return daily.reindex(index, fill_value=0)


def read_daily(path):
    return pd.read_csv(path, parse_dates=["date"]).set_index(["date", "country"]).sort_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    daily = ingest(*args.sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output)
    print(f"Wrote {len(daily):,} daily rows to {args.output}")


if __name__ == "__main__":
    main()
