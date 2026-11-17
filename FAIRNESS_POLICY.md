# Project fairness evaluation policy

## Purpose and status

This policy defines proposed research acceptance criteria for this credit-default
project. The numeric limits are project design choices, not established industry
standards, legal tests, or evidence of suitability for lending. They were written
after inspecting validation findings, so they are not a preregistered evaluation.
Passing means meeting these measured criteria on the evaluated population;
it does not establish fairness in every sense.

The outcome is default next month: 1 means default, and a positive prediction
means flagged as likely to default. We have no observed lending approval
decisions. Treat a false flag as a potential applicant harm if predictions are
used to restrict credit; treat a missed default as a risk-screening error.

## Metrics and acceptance limits

TP = flagged defaulter; FP = flagged non-defaulter; TN = unflagged
non-defaulter; FN = unflagged defaulter. Rates below are proportions;
a 0.10 gap means 10 percentage points, not a 10% relative difference.

| Criterion | Definition | Proposed limit |
|---|---|---|
| Overall recall | TP / (TP + FN), defaults detected | At least 0.60 |
| Overall false-positive rate (FPR) | FP / (FP + TN), non-defaulters flagged | At most 0.25 |
| Group FPR gap, primary fairness criterion | Maximum group FPR minus minimum group FPR, separately within each audited attribute | At most 0.10 for every attribute |
| Group false-negative rate (FNR) gap | Maximum group FNR minus minimum group FNR; FNR = FN / (TP + FN) | At most 0.10 for every attribute |
| Group calibration error | Support-weighted mean absolute difference between mean predicted probability and observed default rate across fixed risk bins, within each group | At most 0.05 for every assessable group |

The 60% recall and 25% FPR constraints continue the exploratory audit's useful
performance screen. The 10-point disparity and 5-point calibration limits are
explicit starting targets for the project. They require substantive justification
before real use. We have no monetary error costs and will not invent them.
The policy constrains both errors instead of assuming a dollar tradeoff.

FPR is the primary disparity metric because it measures erroneous risk flags
among non-defaulters. FNR prevents closing FPR gaps by missing defaults unevenly.
The FNR gap equals the recall/TPR gap. Track precision, average precision,
ROC AUC, Brier score, and predicted non-default rates as supporting measures.
Equal predicted non-default rates are descriptive rather than an acceptance
requirement, since observed default rates may differ across groups. Calibration
and equal error rates may conflict; report unresolved tradeoffs explicitly.

## Groups, support, and calibration bins

Audit original sex, education, and marriage codes; age bands <30, 30-39,
40-49, 50-59, and 60+; and sex-by-age intersections. Retain missing values as
an explicit group. Do not attach demographic meanings to undocumented codes.
Check the dataset codebook before making claims about named populations.

A group is eligible for error-gap comparison only with at least 30 records,
10 defaults, and 10 non-defaults, matching the audit. Report every excluded
group and its counts. An attribute needs at least two eligible groups to assess
its gap. Exclusion is insufficient evidence, not a fairness pass. Full coverage
remains unresolved when an observed group lacks sufficient support.

Use fixed probability bins [0, .2), [.2, .4), [.4, .6), [.6, .8), [.8, 1].
For calibration, report every occupied bin's count, mean probability, and
observed default rate. A group is assessable only if it meets the error-rate
support rules and every occupied bin has at least 20 records. Do not omit a
small occupied bin from the weighted error to obtain a pass. Mark calibration
as insufficient evidence instead. Also display signed errors: a weighted
absolute summary can hide local miscalibration and depends on the chosen bins.

## Uncertainty and decisions

Report 95% Wilson intervals for individual group error rates. To assess gaps
and calibration errors jointly, use 2,000 bootstrap resamples of whole records,
seed 42, retaining all labels, probabilities, and group memberships together.
Keep the original eligible groups and bin definitions fixed; do not reselect
groups in each replicate. If required rates cannot be computed in any
replicate, report incomplete uncertainty evidence rather than silently dropping
replicates or groups.

For the frozen set of assessable fairness statistics m, form conservative
simultaneous intervals using bootstrap quantiles alpha/(2m) and
1-alpha/(2m), with alpha=.05. If 2,000 replicates give fewer than 10 draws in
either requested tail, increase replicates until that condition is satisfied;
otherwise mark uncertainty evidence insufficient. These bootstrap intervals are
approximate and do not account for earlier validation selection. Individual
Wilson intervals are descriptive and are not simultaneous intervals for gaps.

Assign each fairness criterion one of three statuses:

- Meets criterion: its simultaneous interval's upper endpoint is at or below
  the acceptance limit, and support and uncertainty requirements are satisfied.
- Exceeds criterion: its point estimate exceeds the acceptance limit. This
  describes the observed sample; it is not proof of discrimination or a
  statistically established population violation.
- Inconclusive: the point estimate is within the limit but its upper interval
  exceeds it, or required support, calibration, or uncertainty evidence is absent.

A candidate meets the project policy only if both overall performance limits
hold and every required fairness criterion meets its limit with full observed
group coverage. Report overall recall and FPR intervals as well. Passing
numerical checks still requires review of label quality, population coverage,
demographic coding, and potential proxy effects before drawing broader claims.

## Model and threshold selection

1. Fit preprocessing and models using training data only. Use training CV for
   hyperparameters and training OOF predictions for initial shared thresholds.
   Use one threshold for all demographic groups in this experiment.
2. On validation, apply the performance screen, then the fairness criteria.
   Keep all failed and inconclusive candidates in the report. Do not tune to
   the test set or quietly relax limits when no candidate qualifies.
3. Among qualifying candidates, prefer the smallest worst group FPR gap;
   use average precision, then Brier score, as tie-breakers. Differences without
   adequate uncertainty evidence are exploratory and should not be called gains.
4. If no candidate qualifies, retain the baseline only as a research comparator.
   Investigate the groups driving disparities, calibration, and training data.
   Any policy revision must be documented before the next evaluation.
5. Freeze the chosen fitted model, preprocessing, threshold, group definitions,
   metrics, and limits. Evaluate once on the untouched test split. A failed or
   inconclusive test evaluation is reported as such; further development needs
   new independent confirmation data.

To claim a fairness improvement over a baseline, also require a paired
whole-record bootstrap comparison on the same records, with the candidate
and baseline predictions held fixed. The simultaneous interval for the change
in worst FPR gap must lie below zero, performance limits must hold, and no
other required fairness criterion may change from meeting to exceeding its
limit. Assessing this requires the original baseline predictions. These are
now available in `outputs/val_preds_*.csv` following integration of the partner's
branch; the full paired baseline acceptance comparison is still pending.
Selecting many candidates on validation
still limits confirmatory claims even with these intervals.

## Current project assessment

The original random forest's reported worst TPR/FPR gap was 27.0 points.
The new random forest candidates' worst gaps were about 33.7-33.9 points.
Both exceed the proposed 10-point maximum for at least one error-gap
criterion. They therefore do not meet this proposed policy on validation.
The combined maximum alone does not identify whether FPR, FNR, or both
exceed the limit: inspect per-attribute group rates for that diagnosis.

Calibration, simultaneous uncertainty intervals, complete group coverage,
and demographic code verification have not yet been completed under this
policy. The current training script computes point error rates and applies
the overall threshold constraints; it does not implement this full acceptance
policy. No existing model is certified as fair, and the test set remains unused.
