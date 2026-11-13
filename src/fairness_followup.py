"""Validation diagnostics and one training-only age/label reweighting experiment.

Run: python -m src.fairness_followup
"""
import json
import os
from pathlib import Path

import joblib
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.cache' / 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from src.config import ROOT, SEED, SENSITIVE
from src.fairness import demographic_groups, group_rates, wilson_interval, worst_gap
from src.model_experiments import choose_threshold, evaluate

BASE = 'random_forest_without_demographics'
OUT = ROOT / 'outputs' / 'fairness_followup'


def training_weights(X, y):
    """Smoothed P(age)P(label)/P(age,label), fitted only on the supplied fold.

    Clip at [0.5, 2] and normalize to mean 1 to limit sparse-cell influence.
    These sample weights supplement the fixed baseline's balanced class weights.
    """
    if not X.index.equals(y.index):
        raise ValueError('Weighting rows must align.')
    ages = demographic_groups(X[SENSITIVE]).age_band
    counts = pd.crosstab(ages, y).reindex(columns=[0, 1], fill_value=0).astype(float) + 1.
    joint = counts / counts.to_numpy().sum()
    ratio = np.outer(joint.sum(axis=1), joint.sum(axis=0)) / joint.to_numpy()
    lookup = pd.DataFrame(ratio, index=joint.index, columns=joint.columns)
    weights = np.array([lookup.loc[age, label] for age, label in zip(ages, y)])
    weights = np.clip(weights, .5, 2.)
    return weights / weights.mean()


def calibration(y, probabilities, sensitive):
    """Fixed five-bin calibration; retain small bins and flag insufficient support."""
    groups = demographic_groups(sensitive)
    if not y.index.equals(sensitive.index):
        raise ValueError('Calibration rows must align.')
    y, probabilities = np.asarray(y), np.asarray(probabilities)
    if probabilities.shape != y.shape or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Calibration requires one finite probability in [0, 1] per label.')
    bin_index = np.minimum((probabilities * 5).astype(int), 4)
    summaries, bins = [], []
    for attribute in groups:
        for group in sorted(groups[attribute].unique()):
            mask = (groups[attribute] == group).to_numpy(dtype=bool)
            n = int(mask.sum())
            positives = int(y[mask].sum())
            eligible = n >= 30 and positives >= 10 and n - positives >= 10
            error, min_count = 0., n
            for b in range(5):
                cell = mask & (bin_index == b)
                count = int(cell.sum())
                if not count:
                    continue
                observed, predicted = float(y[cell].mean()), float(probabilities[cell].mean())
                low, high = wilson_interval(int(y[cell].sum()), count)
                error += count / n * abs(observed - predicted)
                min_count = min(min_count, count)
                bins.append(dict(attribute=attribute, group=group, bin=b, n=count,
                                 predicted=predicted, observed=observed, low=low, high=high))
            summaries.append(dict(attribute=attribute, group=group, n=n, defaults=positives,
                                  ece=error, mean_prediction=float(probabilities[mask].mean()),
                                  default_rate=float(y[mask].mean()), min_occupied_bin=min_count,
                                  assessable=bool(eligible and min_count >= 20)))
    return pd.DataFrame(summaries), pd.DataFrame(bins)


def gap_table(rates):
    rows = []
    for attribute, subset in rates.groupby('attribute'):
        eligible = subset[subset.eligible]
        rows.append(dict(attribute=attribute, eligible_groups=len(eligible),
                         excluded_groups=int((~subset.eligible).sum()),
                         fpr_gap=float(eligible.fpr.max() - eligible.fpr.min()) if len(eligible) >= 2 else np.nan,
                         fnr_gap=float(eligible.fnr.max() - eligible.fnr.min()) if len(eligible) >= 2 else np.nan))
    return pd.DataFrame(rows)


def plot_rates(rates, output):
    for metric in ('fpr', 'fnr'):
        attributes = list(rates.attribute.unique())
        fig, axes = plt.subplots(1, len(attributes), figsize=(19, 7), sharex=True)
        for ax, attribute in zip(axes, attributes):
            subset = rates[rates.attribute == attribute].reset_index(drop=True)
            for i, row in subset.iterrows():
                if pd.notna(row[metric]):
                    ax.errorbar(row[metric], i,
                                xerr=[[max(0., row[metric] - row[metric + '_low'])],
                                      [max(0., row[metric + '_high'] - row[metric])]],
                                fmt='o', color='#176b8f' if row.eligible else '#999999')
            ax.set_yticks(range(len(subset)),
                          [f'{r.group} (n={r.n}; D={r.defaults}; ND={r.nondefaults})'
                           for r in subset.itertuples()], fontsize=8)
            ax.set_title(attribute)
            ax.set_xlim(-.03, 1.03)
            ax.set_xlabel(metric.upper())
            ax.grid(axis='x', alpha=.25)
            ax.invert_yaxis()
        fig.suptitle('Baseline validation: marginal 95% Wilson intervals; gray = sparse groups\n'
                     'D = defaults, ND = non-defaults; intervals are not for differences')
        fig.tight_layout()
        fig.savefig(output / f'baseline_{metric}.png', dpi=170, bbox_inches='tight')
        plt.close(fig)


def plot_calibration(bins, output):
    fig, axes = plt.subplots(1, 5, figsize=(19, 5), sharex=True, sharey=True)
    for ax, (attribute, subset) in zip(axes, bins.groupby('attribute', sort=False)):
        ax.plot([0, 1], [0, 1], '--', color='gray')
        for group, points in subset.groupby('group'):
            ax.plot(points.predicted, points.observed, 'o-', label=group, markersize=3)
        ax.set_title(attribute)
        ax.set_xlabel('Mean predicted default probability')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Observed default rate')
    fig.suptitle('Baseline validation calibration: five fixed bins, including sparse bins\n'
                 'Descriptive curves; bin counts and rate intervals are in calibration_bins.csv')
    fig.tight_layout()
    fig.savefig(output / 'baseline_calibration.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


def paired_gap_bootstrap(y, sensitive, baseline, candidate, repetitions=2000):
    """Exploratory paired interval for change in worst eligible error gap.

    Freeze original eligible groups; resample whole records. Invalid replicates
    are counted and prevent an interval claim. This is a marginal interval and
    does not implement the full simultaneous-interval project policy.
    """
    rates = group_rates(y, baseline, sensitive)
    groups = demographic_groups(sensitive)
    masks = []
    for attribute, frame in rates[rates.eligible].groupby('attribute'):
        if len(frame) >= 2:
            masks.append([(groups[attribute] == group).to_numpy(dtype=bool) for group in frame.group])
    truth = np.asarray(y)
    rng, changes = np.random.default_rng(SEED), []
    invalid = 0
    for _ in range(repetitions):
        indices = rng.integers(0, len(y), len(y))
        worst = [0., 0.]
        valid = True
        for attribute_masks in masks:
            for label in (0, 1):
                values = [[], []]
                for mask in attribute_masks:
                    selected = mask[indices] & (truth[indices] == label)
                    if not selected.any():
                        valid = False
                        break
                    for j, predictions in enumerate((baseline, candidate)):
                        values[j].append(float(predictions[indices][selected].mean()))
                if not valid:
                    break
                for j in range(2):
                    worst[j] = max(worst[j], max(values[j]) - min(values[j]))
            if not valid:
                break
        if valid and masks:
            changes.append(worst[1] - worst[0])
        else:
            invalid += 1
    interval = np.quantile(changes, [.025, .975]).tolist() if not invalid else None
    return dict(repetitions=repetitions, invalid_replicates=invalid,
                change_worst_gap_interval=interval,
                note='Marginal paired percentile interval; validation selection is not accounted for.')


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    data = ROOT / 'data' / 'processed'
    previous = ROOT / 'outputs' / 'model_improvement'
    frames = {}
    for split in ('train', 'val'):
        X = pd.read_csv(data / f'X_{split}_raw.csv', index_col='source_row')
        y = pd.read_csv(data / f'y_{split}.csv', index_col='source_row').squeeze('columns')
        if not X.index.is_unique or not X.index.equals(y.index):
            raise ValueError('Features and labels must align uniquely.')
        frames[split] = X, y
    X, y = frames['train']
    V, v = frames['val']
    if len(X.index.intersection(V.index)):
        raise ValueError('Train and validation overlap.')
    # Reuse the fixed fitted baseline configuration, avoiding a second parameter search.
    baseline = joblib.load(previous / f'{BASE}.joblib')
    baseline_probability = baseline.predict_proba(V)[:, 1]
    old_run = json.loads((previous / 'run.json').read_text())
    cutoff = old_run['candidates'][BASE]['oof_threshold']
    cutoff = .5 if cutoff is None else cutoff
    print('Training age/label reweighted random forest, with fold-local weights', flush=True)
    oof = np.empty(len(y))
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    for train, holdout in cv.split(X, y):
        fold = clone(baseline)
        fold.fit(X.iloc[train], y.iloc[train],
                 model__sample_weight=training_weights(X.iloc[train], y.iloc[train]))
        oof[holdout] = fold.predict_proba(X.iloc[holdout])[:, 1]
    selected = choose_threshold(y, oof)
    weighted_cutoff = .5 if selected is None else selected
    weighted = clone(baseline)
    weights = training_weights(X, y)
    weighted.fit(X, y, model__sample_weight=weights)
    probability = weighted.predict_proba(V)[:, 1]
    joblib.dump(weighted, OUT / 'reweighted_random_forest.joblib')
    pd.DataFrame({'source_row': X.index, 'age_band': demographic_groups(X[SENSITIVE]).age_band,
                  'label': y, 'sample_weight': weights, 'oof_probability': oof}).to_csv(OUT / 'training_weights.csv', index=False)
    pd.DataFrame({'source_row': V.index, 'y_true': v, 'baseline_probability': baseline_probability,
                  'reweighted_probability': probability}).to_csv(OUT / 'validation_predictions.csv', index=False)
    rows, rate_tables, gap_tables, calibration_tables, bin_tables = [], [], [], [], []
    policies = [('baseline_train_threshold', baseline_probability, cutoff),
                ('reweighted_fixed_baseline_threshold', probability, cutoff),
                ('reweighted_train_threshold', probability, weighted_cutoff)]
    for name, probabilities, threshold in policies:
        metrics, rates = evaluate(v, probabilities, V[SENSITIVE], threshold)
        rows.append(dict(candidate=name, **metrics))
        rate_tables.append(rates.assign(candidate=name))
        gap_tables.append(gap_table(rates).assign(candidate=name))
        if name != 'reweighted_fixed_baseline_threshold':
            summary, bins = calibration(v, probabilities, V[SENSITIVE])
            calibration_tables.append(summary.assign(candidate=name))
            bin_tables.append(bins.assign(candidate=name))
        if name == 'baseline_train_threshold':
            plot_rates(rates, OUT)
            _, bins = calibration(v, probabilities, V[SENSITIVE])
            plot_calibration(bins, OUT)
    results = pd.DataFrame(rows)
    results.to_csv(OUT / 'comparison.csv', index=False)
    pd.concat(rate_tables).to_csv(OUT / 'group_rates.csv', index=False)
    pd.concat(gap_tables).to_csv(OUT / 'group_gaps.csv', index=False)
    pd.concat(calibration_tables).to_csv(OUT / 'calibration_summary.csv', index=False)
    pd.concat(bin_tables).to_csv(OUT / 'calibration_bins.csv', index=False)
    uncertainty = paired_gap_bootstrap(v, V[SENSITIVE], baseline_probability >= cutoff,
                                       probability >= weighted_cutoff)
    metadata = dict(seed=SEED, test_used=False, baseline=BASE,
                    baseline_threshold=cutoff, reweighted_oof_threshold=selected,
                    weighting='Fold-local Laplace-smoothed age/label independence weights, clipped [0.5,2], normalized; balanced class_weight retained',
                    scope='Exploratory validation follow-up, not full FAIRNESS_POLICY acceptance evaluation',
                    uncertainty=uncertainty)
    (OUT / 'run.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(results.to_string(index=False))
    print(json.dumps(uncertainty, indent=2))
    return results


if __name__ == '__main__':
    run()
