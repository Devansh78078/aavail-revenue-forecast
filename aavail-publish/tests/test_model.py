from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from forecast.ingest import ALL
from forecast.model import DAY, ForecastError, combined_history, design, fit, forecast, load_model, save_model
from tests.fixtures import daily_fixture


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.daily = daily_fixture()
        cls.bundle, cls.validation = fit(cls.daily)

    def test_future_changes_only_targets(self):
        series = self.daily.xs(ALL, level="country").copy()
        date = series.index[200]
        original = design(series).loc[date]
        series.loc[date:, "revenue"] += 1000
        modified = design(series).loc[date]
        pd.testing.assert_series_equal(original.drop("target"), modified.drop("target"))
        self.assertAlmostEqual(modified["target"] - original["target"], 30000)

    def test_target_is_inclusive_thirty_days(self):
        series = self.daily.xs(ALL, level="country")
        date = series.index[200]
        self.assertEqual(design(series).loc[date, "target"], series.loc[date:date+29*DAY, "revenue"].sum())

    def test_deserialized_model_matches_original(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            save_model(self.bundle, path)
            self.assertEqual(forecast(self.bundle, self.daily, "Canada"),
                             forecast(load_model(path), self.daily, "Canada"))

    def test_rejects_dates_seen_by_fit_and_unobserved_dates(self):
        cutoff = self.bundle["through"]
        for date in (cutoff, cutoff+2*DAY):
            with self.subTest(date=date), self.assertRaises(ForecastError):
                forecast(self.bundle, self.daily, "Canada", str(date.date()))

    def test_new_observations_do_not_leak_into_earlier_forecast(self):
        series = self.daily.iloc[-9:].reset_index()
        series["date"] += 3*DAY
        series["revenue"] *= 1000
        replay = combined_history(self.bundle, series.set_index(["date", "country"]))
        origin = str((self.bundle["through"] + DAY).date())
        self.assertEqual(forecast(self.bundle, replay, "all", origin),
                         forecast(self.bundle, self.daily, "all"))
        self.assertNotEqual(forecast(self.bundle, replay, "all")["date"], origin)

    def test_gap_in_appended_observations_is_rejected(self):
        series = self.daily.iloc[-3:].reset_index()
        series["date"] += 2*DAY
        with self.assertRaises(ValueError):
            combined_history(self.bundle, series.set_index(["date", "country"]))

    def test_validates_multiple_models_and_folds(self):
        self.assertEqual(self.validation["model"].nunique(), 6)
        self.assertEqual(self.validation["fold"].nunique(), 3)
        self.assertEqual(len(self.validation.query("country == 'Canada' and model == 'ridge'")), 90)
