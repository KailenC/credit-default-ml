"""Group error rates for validation audits; positive label means default."""
import numpy as np
import pandas as pd


def demographic_groups(sensitive):
    """Shared audit definitions, including explicit missing-value groups."""
    groups = sensitive[['sex', 'education', 'marriage']].astype('string').fillna('missing')
    age = pd.cut(sensitive.age, [-np.inf, 30, 40, 50, 60, np.inf],
                 right=False, labels=['<30', '30-39', '40-49', '50-59', '60+'])
    groups['age_band'] = age.astype('string').fillna('missing')
    groups['sex_age'] = groups.sex + ':' + groups.age_band
    return groups


def wilson_interval(successes, total, z=1.959963984540054):
    """Marginal 95% binomial Wilson interval; undefined without observations."""
    if total == 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total ** 2)) / denominator
    return max(0., center - radius), min(1., center + radius)


def group_rates(y, predictions, sensitive):
    """Return all groups, flagging those eligible for gap comparisons."""
    if not y.index.equals(sensitive.index):
        raise ValueError('Labels and sensitive attributes must have identical row order.')
    groups = demographic_groups(sensitive)
    truth, pred = np.asarray(y), np.asarray(predictions)
    if len(pred) != len(truth):
        raise ValueError('Predictions and labels must have equal lengths.')
    rows = []
    for attribute in groups:
        for group in sorted(groups[attribute].unique()):
            mask = (groups[attribute] == group).to_numpy(dtype=bool)
            positive, negative = mask & (truth == 1), mask & (truth == 0)
            n_pos, n_neg = int(positive.sum()), int(negative.sum())
            tp, fp = int(pred[positive].sum()), int(pred[negative].sum())
            fn = n_pos - tp
            fpr_low, fpr_high = wilson_interval(fp, n_neg)
            fnr_low, fnr_high = wilson_interval(fn, n_pos)
            rows.append(dict(attribute=attribute, group=group, n=int(mask.sum()),
                             defaults=n_pos, nondefaults=n_neg,
                             tp=tp, fp=fp, fn=fn, tn=n_neg - fp,
                             tpr=float(pred[positive].mean()) if n_pos else np.nan,
                             fpr=float(pred[negative].mean()) if n_neg else np.nan,
                             fnr=fn / n_pos if n_pos else np.nan,
                             fpr_low=fpr_low, fpr_high=fpr_high,
                             fnr_low=fnr_low, fnr_high=fnr_high,
                             eligible=bool(mask.sum() >= 30 and n_pos >= 10 and n_neg >= 10)))
    return pd.DataFrame(rows)


def worst_gap(rates):
    """Largest within-attribute TPR/FPR gap; exclude sparse groups."""
    gaps = []
    for _, group in rates[rates.eligible].groupby('attribute'):
        if len(group) >= 2:
            gaps.extend(group[metric].max() - group[metric].min() for metric in ('tpr', 'fpr'))
    return float(max(gaps)) if gaps else float('nan')
