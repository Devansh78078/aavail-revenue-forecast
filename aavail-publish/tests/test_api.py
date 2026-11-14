import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from forecast.api import create_app
from forecast.ingest import ALL
from forecast.model import DAY, fit, save_model
from tests.fixtures import daily_fixture, test_config

AUTH = {"Authorization": "Bearer fixture-token"}


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.daily = daily_fixture()
        cls.bundle, _ = fit(cls.daily)
        cls.date = str((cls.bundle["through"] + DAY).date())

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.config = test_config(Path(directory.name))
        save_model(self.bundle, self.config["MODEL_SEED"])
        self.client = create_app(self.config).test_client()

    def test_predict_country_and_aggregate(self):
        local = self.client.post("/predict", json={"country": "united_kingdom", "date": self.date})
        public = self.client.get("/predict?country=all")
        self.assertEqual((local.status_code, public.status_code), (200, 200))
        self.assertEqual(local.json["country"], "United Kingdom")
        self.assertEqual(local.json["date"], self.date)
        self.assertEqual(public.json["country"], ALL)
        self.assertGreater(public.json["prediction"], local.json["prediction"])
        self.assertEqual(public.json["days"], 30)

    def test_health_and_countries(self):
        self.assertEqual(self.client.get("/health").json["status"], "ready")
        self.assertIn("Canada", self.client.get("/countries").json["countries"])

    def test_missing_model_is_unavailable(self):
        self.config["MODEL_SEED"].unlink()
        self.assertEqual(self.client.get("/health").status_code, 503)
        self.assertEqual(self.client.get("/predict").status_code, 503)

    def test_invalid_requests(self):
        for query, expected in [
            ("country=unknown", 422), ("country=", 400), ("date=bad", 400),
            ("date=2018-01-01", 422), ("date=8888-01-01", 422),
        ]:
            with self.subTest(query=query):
                self.assertEqual(self.client.get(f"/predict?{query}").status_code, expected)
        self.assertEqual(self.client.post("/predict", json=[1]).status_code, 400)
        self.assertEqual(self.client.post("/predict", json={"date": 2019}).status_code, 400)

    def test_operator_token_and_log_isolation(self):
        self.assertEqual(self.client.post("/train").status_code, 401)
        self.assertEqual(self.client.get("/logfile").status_code, 401)
        self.assertEqual(self.client.get("/metrics").status_code, 401)
        self.client.get("/predict?country=Canada")
        result = self.client.get("/logfile?kind=prediction&limit=1", headers=AUTH)
        self.assertEqual(result.json["events"][0]["country"], "Canada")
        self.assertEqual(self.client.get("/logfile?limit=0", headers=AUTH).status_code, 400)
        self.assertGreaterEqual(self.client.get("/metrics", headers=AUTH).json["sample_size"], 2)

    def test_observations_refresh_when_files_are_published(self):
        directory = self.config["MODEL_PATH"].parent / "observations"
        directory.mkdir()
        first = self.bundle["through"] + DAY
        for index in range(2):
            day = first + index * DAY
            row = dict(country="Canada", price=100, stream_id="s", invoice=str(10+index),
                       times_viewed=4, year=day.year, month=day.month, day=day.day)
            (directory / f"batch-{index}.json").write_text(json.dumps([row]))
            if index == 0:
                client = create_app({**self.config, "OBSERVATIONS": directory}).test_client()
            self.assertEqual(client.get("/health").json["observed_through"], str(day.date()))

    def test_training_publishes_new_model(self):
        with patch("forecast.api.ingest", return_value=self.daily):
            result = self.client.post("/train", headers=AUTH)
        self.assertEqual(result.status_code, 200)
        self.assertTrue(self.config["MODEL_PATH"].exists())
        self.assertEqual(self.client.get("/predict").json["version"], result.json["version"])
        self.assertNotEqual(result.json["version"], self.bundle["version"])

    def test_failed_training_keeps_served_model(self):
        result = self.client.post("/train", headers=AUTH)
        self.assertEqual(result.status_code, 422)
        self.assertFalse(self.config["MODEL_PATH"].exists())
        self.assertEqual(self.client.get("/predict").json["version"], self.bundle["version"])
