"""Replay daily forecasts against a running service and its event journal."""
import argparse
from datetime import date, timedelta
import json
from urllib.parse import urlencode
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--from", dest="start", type=date.fromisoformat, required=True)
    parser.add_argument("--to", dest="end", type=date.fromisoformat, required=True)
    parser.add_argument("--countries", nargs="+", default=["all", "United Kingdom"])
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("start must precede or equal end")
    current = args.start
    count = 0
    while current <= args.end:
        for country in args.countries:
            query = urlencode({"date": str(current), "country": country})
            with urlopen(f"{args.url.rstrip('/')}/predict?{query}", timeout=15) as response:
                result = json.load(response)
            if result["observed_through"] != str(current - timedelta(days=1)):
                raise RuntimeError(f"unexpected forecast history cutoff for {country} on {current}")
            count += 1
        current += timedelta(days=1)
    print(f"Logged {count} forecasts via HTTP")


if __name__ == "__main__":
    main()
