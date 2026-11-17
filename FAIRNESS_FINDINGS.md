# Fairness findings and potential next steps

## Scope

We reviewed four models using saved predictions for 750 validation records: 166 defaults and 584 non-defaults. Predictions were joined to labels and demographic attributes by `source_row`. We checked sex codes, education, marriage, age bands, and sex-by-age intersections.

We also compared 1,330 model/threshold combinations for random forest and histogram gradient boosting. No models were retrained, no deployed settings were changed, and the test set was not used.

## Findings

At the existing 0.5 threshold:

| Model | Defaults detected (recall) | Non-defaulters incorrectly flagged (FPR) | Defaults missed (FNR) |
|---|---:|---:|---:|
| Logistic regression | 64.5% | 32.4% | 35.5% |
| Decision tree | 65.1% | 21.9% | 34.9% |
| Random forest | 63.9% | 21.9% | 36.1% |
| Histogram gradient boosting | 32.5% | 4.1% | 67.5% |

- Logistic regression showed the largest education-group FPR gap (30.3 percentage points), plus a 9.0-point sex-code FPR gap.
- Random forest had smaller education and sex FPR gaps (2.1 and 0.2 points), but age and intersectional disparities remained. Its maximum eligible-group TPR or FPR gap across the audited attributes was 27.0 points.
- Gradient boosting's small FPR gaps at 0.5 coincided with missing 67.5% of defaults. Low disparities alone do not establish useful performance.
- Sex, education, marriage, and age are currently model inputs. Their contribution to performance and disparities has not yet been isolated.

## Threshold comparison

At the same 63.9% default recall:

| Model | Threshold (rounded) | False flags among 584 non-defaulters | Missed defaults among 166 defaults | Worst eligible-group error gap |
|---|---:|---:|---:|---:|
| Random forest baseline | 0.5000 | 128 | 60 | 27.0 points |
| Histogram gradient boosting | 0.2284 | 130 | 60 | 32.1 points |

Lowering random forest's threshold to approximately 0.4833 detected five additional defaults but produced 13 additional false flags. Its worst group error gap decreased only slightly, to 26.6 points. This candidate minimized the worst error gap subject to exploratory recall >=60% and FPR <=25%; those constraints are analysis choices, not agreed business requirements.

Random forest remains a useful comparison baseline. Threshold tuning did not demonstrate a clear fairness improvement, and small numerical differences should not be treated as established gains.

## What these results mean

The project is a useful prototype, but this analysis does not establish readiness for real lending decisions. Acceptability depends on the costs of missed defaults and false flags, which have not been defined. The findings also do not establish that any model is fair or that observed disparities prove discrimination.

Important limits:

- These are exploratory validation results; validation was already used for model comparison and threshold exploration.
- Group gaps use groups with at least 30 records, 10 defaults, and 10 non-defaults. Sparse groups remain a concern. Age bands were <30, 30-39, 40-49, 50-59, and 60+.
- Marginal 95% Wilson intervals were calculated for group rates, but not for group differences or threshold-selection gains. No significance or multiple-comparison claims were made.
- Demographic code meanings are unverified. Predicted non-default is a hypothetical favorable-outcome proxy, not an observed approval decision.
- Data representativeness, historical label bias, unavailable protected attributes, and exact training provenance remain unresolved.

## Potential next steps

1. Confirm demographic code meanings and define acceptable false-flag and missed-default costs.
2. Retrain random forest and gradient boosting with and without sex, education, marriage, and age, using training-only fitting. Compare both performance and group errors on validation. Removing these inputs may leave proxy effects.
3. Check group calibration and stability under training-only resampling, with particular attention to sparse and intersectional groups.
4. Compare resulting candidates against random forest at threshold 0.5. Choose a model and threshold using explicit performance and fairness criteria rather than the smallest validation gap alone.
5. Freeze the model and decision policy, then evaluate once on the untouched test set and document remaining limitations.

This commit contains only this findings summary. Audit implementation and generated metric files are not included.

Integration note: the statement above describes the original summary commit.
The branch now includes follow-up audit code, metrics, and notebooks, together
with the partner's model configurations and original validation predictions.
See `FAIRNESS_FOLLOWUP.md` and `README.md` for the current workflows.
