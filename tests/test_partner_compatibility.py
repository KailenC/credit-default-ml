"""Keep the shared modeling API compatible with the partner's notebook."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT
from src.models import (get_models, get_param_spaces, evaluate,
                        export_predictions, cross_validate_models,
                        soft_rank_models, tune_models)
from src.model_experiments import evaluate as evaluate_fairness


class PartnerCompatibilityTests(unittest.TestCase):
    def test_model_api_and_prediction_export(self):
        models = get_models()
        self.assertEqual(set(models), {'logreg', 'decision_tree', 'random_forest', 'hist_bg'})
        self.assertEqual(set(get_param_spaces()), set(models))
        self.assertTrue(all(callable(f) for f in (cross_validate_models, soft_rank_models, tune_models)))
        data = ROOT / 'data' / 'processed'
        X = pd.read_csv(data / 'X_train_raw.csv', index_col='source_row').iloc[:300]
        y = pd.read_csv(data / 'y_train.csv', index_col='source_row').squeeze('columns').loc[X.index]
        model = models['decision_tree'].fit(X, y)
        scores = evaluate(model, X, y)
        self.assertEqual(set(scores), {'auc', 'pr_auc', 'precision', 'recall', 'f1'})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.csv'
            export_predictions(model, X, y, path)
            exported = pd.read_csv(path, index_col='source_row')
            self.assertTrue(exported.index.equals(X.index))
            np.testing.assert_array_equal(exported.y_true, y)
            np.testing.assert_allclose(exported.y_prob, model.predict_proba(X)[:, 1])
        # The fairness evaluator coexists with the partner's different signature.
        metrics, groups = evaluate_fairness(y, model.predict_proba(X)[:, 1], X, .5)
        self.assertIn('worst_group_gap', metrics)
        self.assertFalse(groups.empty)


if __name__ == '__main__':
    unittest.main()
