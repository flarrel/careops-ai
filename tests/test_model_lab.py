import importlib.util
import unittest

from careops.corpus import DEV_CLAIMS, read_jsonl

HAS_LAB = all(importlib.util.find_spec(name) for name in ("numpy", "PIL", "sklearn", "pydantic"))


@unittest.skipUnless(HAS_LAB, "Install the models extra")
class ModelLabTests(unittest.TestCase):
    def test_scan_splits_hold_entire_claims_and_balance_conditions(self):
        import numpy as np

        from careops.model_lab.fixtures import CLASSES, IMAGE_SIZE, make_scans, render_scan

        claims = read_jsonl(DEV_CLAIMS)
        scans = make_scans(claims)
        all_ids = set()
        for split in scans.values():
            claim_ids = {item.split(":")[0] for item in split.ids}
            self.assertTrue(all_ids.isdisjoint(claim_ids))
            all_ids.update(claim_ids)
            self.assertEqual(split.pixels.shape[1:], (IMAGE_SIZE, IMAGE_SIZE))
            self.assertEqual(len(split.ids), len(set(split.ids)))
            self.assertEqual(len(set(np.bincount(split.labels))), 1)
        self.assertEqual(all_ids, {row["claim_id"] for row in claims})
        sample = claims[0]
        rendered = [render_scan(sample, 0, label) for label in CLASSES]
        self.assertTrue(np.array_equal(rendered[0], render_scan(sample, 0, CLASSES[0])))
        self.assertEqual(len({image.tobytes() for image in rendered}), len(CLASSES))

    def test_duplicate_claim_cannot_leak_between_scan_splits(self):
        from careops.model_lab.fixtures import split_claims

        claims = read_jsonl(DEV_CLAIMS)
        with self.assertRaisesRegex(ValueError, "Duplicate development claim ID"):
            split_claims([*claims, claims[0]])

    def test_forecast_baseline_uses_the_prior_week(self):
        from careops.model_lab.tabular import DAYS, FORECAST_START, forecast, simulate_arrivals

        claims = read_jsonl(DEV_CLAIMS)
        series, flagged = simulate_arrivals(claims)
        result = forecast(claims)
        self.assertEqual(len(result["predictions"]), DAYS - FORECAST_START)
        for day, prediction in enumerate(result["predictions"], FORECAST_START):
            self.assertEqual(prediction["seasonal_naive"], series[day - 7])
            self.assertEqual(prediction["arrivals"], series[day])
            self.assertEqual(prediction["flagged"], flagged[day])
            self.assertLessEqual(flagged[day], series[day])
