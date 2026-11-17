# Model comparison after the fairness audit

The local `david/branch1` branch was updated to the fairness findings commit
`105baab`. We then trained four new candidates on the saved 3,497 training
records and evaluated on the 750 validation records. No test files were loaded.

## Results

All thresholds below are selected from training out-of-fold predictions,
except the explicitly shown 0.5 policy. Hyperparameters were selected by
three-fold training average precision. Group gaps follow the original audit's
support rules and include sex, education, marriage, age bands, and sex-age groups.

| Candidate | Threshold | Default recall | Non-default FPR | False flags | Missed defaults | Worst eligible group gap |
|---|---:|---:|---:|---:|---:|---:|
| Original audit random forest (reported) | 0.5000 | 63.9% | 21.9% | 128 | 60 | 27.0 points |
| New random forest, all features | 0.2477 | 66.9% | 21.1% | 123 | 55 | 33.9 points |
| New random forest, no demographics | 0.5000 | 65.7% | 20.2% | 118 | 57 | 33.9 points |
| New random forest, no demographics | 0.4834 | 68.7% | 21.6% | 126 | 52 | 33.7 points |
| New gradient boosting, all features | 0.2362 | 62.0% | 20.4% | 119 | 63 | 34.1 points |
| New gradient boosting, no demographics | 0.2390 | 62.0% | 20.4% | 119 | 63 | 37.7 points |

The new random forest without demographics at 0.5 produced 10 fewer false
flags and missed three fewer defaults than the reported audit baseline.
Its training-selected threshold detected eight more defaults than that baseline
with two fewer false flags. These are observed validation differences, not
established generalization gains. The worst group gap increased: this run did
not demonstrate a fairness improvement, and no candidate is promoted as a
replacement for the audited baseline.

The no-demographics random forest had the highest training CV average
precision (0.498), narrowly above the all-feature random forest (0.497).
Validation average precision was 0.582 and 0.585, respectively. The two selected
forests use different class weights, so their differences cannot be attributed
solely to removing demographic inputs. The all-feature model's Brier score
was better (0.136 versus 0.173); thresholded errors alone do not describe
probability quality.

## Reproduce and inspect

Run `.venv\Scripts\python.exe -m src.model_experiments` from the repository root.
Open `notebooks/model_improvement.ipynb` to inspect the saved comparison and
group rates. `outputs/model_improvement/` includes:

- `comparison.csv` and `group_rates.csv`, including sparse groups.
- Per-candidate CV tables and validation predictions keyed by `source_row`.
- `run.json`, recording selected parameters, thresholds, seed, and sklearn version.
- Local fitted pipeline files (`*.joblib`, excluded from Git).

The three automated checks cover demographic exclusion, known group-rate
gaps with sparse-group exclusion, row alignment, and threshold feasibility.

## Limits and next decision

At the time of this experiment, the partner's modeling changes had not been
integrated, so the original baseline numbers came from `FAIRNESS_FINDINGS.md`.
The integrated branch now includes partner configurations in `src/models.py`
and original validation predictions in `outputs/val_preds_*.csv`. This change
does not retroactively reproduce or rerun the historical comparison. The fitted
original pipelines themselves are not committed.
Training OOF scores reuse the hyperparameter-selection folds and are tuning
data. Validation has been reused for exploration; no significance claims or
confidence intervals for differences are provided. Removing demographics can
leave proxies, and a worst-gap statistic is sensitive to group sample sizes.

Retain the audited baseline for comparison. Before freezing a candidate,
define acceptable error costs and group disparity criteria, then investigate
the affected groups and calibration. The test set remains reserved for a single
final evaluation after that decision.
