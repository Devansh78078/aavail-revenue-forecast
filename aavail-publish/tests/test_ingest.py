import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from forecast.ingest import ALL, ingest


class IngestTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.row = dict(country="Canada", total_price=12.5, StreamID="x",
                        TimesViewed=5, invoice="A-42", year=2018, month=1, day=1)

    def write(self, filename, rows):
        (self.root / filename).write_text(json.dumps(rows))

    def test_aliases_zero_days_and_aggregate(self):
        self.write("first.json", [self.row, dict(self.row, country="France", invoice="42")])
        self.write("second.json", [dict(self.row, day=3, invoice="43")])
        daily = ingest(self.root)
        self.assertEqual(daily.loc[(pd.Timestamp("2018-01-01"), ALL), "revenue"], 25)
        self.assertEqual(daily.loc[(pd.Timestamp("2018-01-01"), "Canada"), "revenue"], 12.5)
        self.assertEqual(daily.loc[(pd.Timestamp("2018-01-01"), ALL), "invoices"], 1)
        self.assertEqual(daily.loc[(pd.Timestamp("2018-01-02"), ALL), "revenue"], 0)

    def test_invalid_date_reports_filename(self):
        self.write("invalid.json", [dict(self.row, month=25)])
        with self.assertRaisesRegex(ValueError, "invalid.json"):
            ingest(self.root)

    def test_overlap_is_detected(self):
        self.write("first.json", [self.row])
        self.write("second.json", [self.row])
        with self.assertRaisesRegex(ValueError, "dates overlap"):
            ingest(self.root)
