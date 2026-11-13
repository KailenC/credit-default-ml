"""Train-only model selection and demographic ablation. Run python -m src.model_experiments."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.config import ROOT, SEED, SENSITIVE
from src.fairness import group_rates, worst_gap
from src.preprocess import build_preprocessor


def pipeline(model, exclude_sensitive=False):
    preprocessor = build_preprocessor()
    if exclude_sensitive:
        preprocessor = ColumnTransformer([
            (name, transformer, [c for c in columns if c not in SENSITIVE])
            for name, transformer, columns in preprocessor.transformers
            if any(c not in SENSITIVE for c in columns)
        ], verbose_feature_names_out=False)
    return Pipeline([('preprocess', preprocessor), ('model', model)])


def choose_threshold(y, probabilities):
    """Use train OOF scores only; exploratory constraints inherited from audit."""
    candidates = []
    for threshold in np.unique(np.r_[0., probabilities, 1.]):
        tn, fp, fn, tp = confusion_matrix(y, probabilities >= threshold, labels=[0, 1]).ravel()
        recall, fpr = tp / (tp + fn), fp / (fp + tn)
        if recall >= .60 and fpr <= .25:
            candidates.append((fpr, -recall, float(threshold)))
    return min(candidates)[2] if candidates else None


def evaluate(y, probabilities, sensitive, threshold):
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    rates = group_rates(y, predictions, sensitive)
    return dict(average_precision=float(average_precision_score(y, probabilities)),
                roc_auc=float(roc_auc_score(y, probabilities)),
                brier=float(brier_score_loss(y, probabilities)),
                threshold=float(threshold), recall=float(tp / (tp + fn)),
                fpr=float(fp / (fp + tn)), false_flags=int(fp), missed_defaults=int(fn),
                worst_group_gap=worst_gap(rates)), rates


def run(data_dir=ROOT / 'data' / 'processed', output_dir=ROOT / 'outputs' / 'model_improvement'):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    for split in ('train', 'val'):
        X = pd.read_csv(data_dir / f'X_{split}_raw.csv', index_col='source_row')
        y = pd.read_csv(data_dir / f'y_{split}.csv', index_col='source_row').squeeze('columns')
        if not X.index.is_unique or not X.index.equals(y.index):
            raise ValueError(f'{split}: feature and label rows must match uniquely.')
        frames[split] = X, y
    X, y = frames['train']
    X_val, y_val = frames['val']
    if len(X.index.intersection(X_val.index)):
        raise ValueError('Train and validation rows overlap.')
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    families = {
        'random_forest': (RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1),
                          {'model__min_samples_leaf': [3, 10], 'model__class_weight': [None, 'balanced']}),
        'hist_gradient_boosting': (HistGradientBoostingClassifier(random_state=SEED, early_stopping=False),
                                   {'model__max_leaf_nodes': [7, 15], 'model__l2_regularization': [1., 10.]}),
    }
    records, tables = [], []
    metadata = dict(seed=SEED, sklearn_version=sklearn.__version__, selection='training 3-fold average precision',
                    threshold_rule='train OOF: minimum FPR with recall >= .60 and FPR <= .25; fallback .5',
                    test_used=False, candidates={})
    for family, (estimator, grid) in families.items():
        for exclude in (False, True):
            name = family + ('_without_demographics' if exclude else '_all_features')
            print(f'Training {name}', flush=True)
            search = GridSearchCV(pipeline(estimator, exclude), grid, cv=cv,
                                  scoring='average_precision', n_jobs=1, error_score='raise')
            search.fit(X, y)
            pd.DataFrame(search.cv_results_).to_csv(output_dir / f'{name}_cv.csv', index=False)
            oof = cross_val_predict(search.best_estimator_, X, y, cv=cv,
                                     method='predict_proba', n_jobs=1)[:, 1]
            threshold = choose_threshold(y, oof)
            probabilities = search.best_estimator_.predict_proba(X_val)[:, 1]
            metadata['candidates'][name] = dict(params=search.best_params_, cv_ap=float(search.best_score_),
                                               oof_threshold=threshold)
            for policy, cutoff in [('default', .5), ('train_threshold', threshold if threshold is not None else .5)]:
                metrics, rates = evaluate(y_val, probabilities, X_val[SENSITIVE], cutoff)
                records.append(dict(candidate=name, policy=policy, cv_ap=search.best_score_, **metrics))
                tables.append(rates.assign(candidate=name, policy=policy))
            pd.DataFrame({'source_row': X_val.index, 'y_true': y_val.to_numpy(),
                          'probability': probabilities}).to_csv(output_dir / f'{name}_validation_predictions.csv', index=False)
            joblib.dump(search.best_estimator_, output_dir / f'{name}.joblib')
    results = pd.DataFrame(records)
    results.to_csv(output_dir / 'comparison.csv', index=False)
    pd.concat(tables, ignore_index=True).to_csv(output_dir / 'group_rates.csv', index=False)
    (output_dir / 'run.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(results.to_string(index=False))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'processed')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'model_improvement')
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)
