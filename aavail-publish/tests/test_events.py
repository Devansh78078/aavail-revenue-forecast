from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


from forecast.events import EventLog
from forecast.ingest import ALL
from forecast.monitor import metrics_report, score
from tests.fixtures import daily_fixture


class LogTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.log = EventLog(self.root / "events.sqlite3")

    def test_persists_and_filters_events(self):
        self.log.write("prediction", country="Canada", prediction=42.5)
        self.log.write("training", model_version="v1")
        reopened = EventLog(self.log.path)
        self.assertEqual(reopened.read(kind="prediction")[0]["prediction"], 42.5)
        self.assertEqual(reopened.read(limit=1)[0]["kind"], "training")
        self.assertIn("+00:00", reopened.read()[0]["timestamp"])
        self.assertFalse((self.root / "unrelated.sqlite3").exists())

    def test_concurrent_writes(self):
        def record(number):
            self.log.write("request", number=number)

        with ThreadPoolExecutor(max_workers=4) as workers:
            list(workers.map(record, range(30)))
        self.assertEqual({event["number"] for event in self.log.read()}, set(range(30)))

    def test_monitor_uses_mature_forecasts_and_deduplicates(self):
        actuals = daily_fixture()
        for origin in ["2018-03-01", "2018-03-01", "2019-02-08"]:
            self.log.write("prediction", country=ALL, date=origin, version="v1",
                           model="ridge", prediction=42, baseline_prediction=300)
        matched, coverage = score(self.log, actuals)
        self.assertEqual(coverage["matched"], 1)
        self.assertEqual(coverage["pending"], 1)
        actual = actuals.xs(ALL, level="country").loc["2018-03-01":"2018-03-30", "revenue"].sum()
        self.assertEqual(matched.iloc[0]["actual"], actual)
        self.assertAlmostEqual(metrics_report(matched, coverage, 0.1)["scopes"][0]["selected"]["mae"],
                               abs(actual - 42))
