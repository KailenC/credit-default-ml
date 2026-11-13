import unittest

import numpy as np
import pandas as pd

from src.fairness import group_rates, worst_gap
from src.model_experiments import choose_threshold, pipeline
from src.fairness import wilson_interval
from src.fairness_followup import calibration, training_weights, paired_gap_bootstrap
from sklearn.ensemble import RandomForestClassifier


class AuditTests(unittest.TestCase):
    def test_wilson_boundary_and_no_data(self):
        low, high = wilson_interval(0, 10)
        self.assertAlmostEqual(low, 0.)
        self.assertAlmostEqual(high, .2775328, places=6)
        self.assertTrue(np.isnan(wilson_interval(0, 0)[0]))

    def test_weights_balance_age_label_association_without_validation(self):
        # Symmetric cell counts: weights should equalize each age/label cell.
        y = pd.Series([0] * 30 + [1] * 10 + [0] * 10 + [1] * 30)
        X = pd.DataFrame({'sex': 1, 'education': 1, 'marriage': 1,
                          'age': [25] * 40 + [45] * 40})
        weights = training_weights(X, y)
        self.assertAlmostEqual(weights.mean(), 1.)
        self.assertGreater(weights[30], weights[0])
        self.assertAlmostEqual(weights[30], weights[40])
        self.assertTrue(np.isfinite(weights).all())
        with self.assertRaises(ValueError):
            training_weights(X.iloc[::-1], y)

    def test_calibration_keeps_probability_one_and_sparse_bins(self):
        y = pd.Series([0] * 20 + [1] * 20)
        sensitive = pd.DataFrame({'sex': 1, 'education': 1, 'marriage': 1, 'age': 35}, index=y.index)
        summary, bins = calibration(y, np.array([0.] * 20 + [1.] * 20), sensitive)
        self.assertTrue((summary.ece == 0).all())
        self.assertTrue(summary.assessable.all())
        self.assertEqual(set(bins.bin), {0, 4})
        probabilities = np.array([0.] * 19 + [.5] + [1.] * 20)
        summary, _ = calibration(y, probabilities, sensitive)
        self.assertFalse(summary.assessable.any())

    def test_identical_predictions_have_zero_paired_gap_change(self):
        y = pd.Series(([0] * 20 + [1] * 20) * 2)
        sensitive = pd.DataFrame({'sex': [1] * 40 + [2] * 40,
                                  'education': 1, 'marriage': 1, 'age': 35})
        predictions = np.array([False] * 40 + [True] * 40)
        result = paired_gap_bootstrap(y, sensitive, predictions, predictions, repetitions=100)
        self.assertEqual(result['invalid_replicates'], 0)
        self.assertEqual(result['change_worst_gap_interval'], [0., 0.])

    def test_rates_and_sparse_exclusion(self):
        y = pd.Series([0] * 20 + [1] * 20 + [0] * 20 + [1] * 20 + [1])
        sensitive = pd.DataFrame({'sex': [1] * 40 + [2] * 40 + [3],
                                  'education': 1, 'marriage': 1, 'age': 35})
        pred = np.array([0] * 40 + [1] * 41)
        rates = group_rates(y, pred, sensitive)
        self.assertEqual(worst_gap(rates), 1.)
        self.assertFalse(rates.query("attribute == 'sex' and group == '3'").eligible.iloc[0])
        with self.assertRaises(ValueError):
            group_rates(y, pred, sensitive.iloc[::-1])

    def test_threshold_uses_default_as_positive(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        probabilities = np.array([.1, .2, .3, .8, .4, .5, .6, .9])
        self.assertEqual(choose_threshold(y, probabilities), .4)
        self.assertIsNone(choose_threshold(y, np.full(8, .5)))

    def test_demographics_excluded_from_transformer(self):
        model = pipeline(RandomForestClassifier(), exclude_sensitive=True)
        columns = [c for _, _, cols in model['preprocess'].transformers for c in cols]
        self.assertTrue(columns)
        self.assertTrue(set(columns).isdisjoint({'sex', 'education', 'marriage', 'age'}))


if __name__ == '__main__':
    unittest.main()
