# Credit Default ML Project

## Setup

1. Clone the repo and go into the folder:
   
2. Create and activate a virtual environment:
   ```
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

3. Install the libraries:
   ```
   pip install -r requirements.txt
   ```

4. Put the dataset at `data/train.csv` (re-name to project dataset to train.csv)

5. Turn on notebook output stripping (keeps git diffs clean):
   ```
   nbstripout --install
   ```

6. Use Jupyter or some other viewer and open a notebook in `notebooks/`:
   ```
   jupyter lab
   ```
## Preprocess the credit default data

From the repository folder, with the virtual environment activated:

```powershell
python -m pip install -r requirements-preprocess.txt
python -m src.preprocess --input data/train.csv
```

Outputs go to `data/processed/` (shared in Git for model development): transformed `X_train`,
`X_val`, and `X_test` CSVs, corresponding labels, raw features, original
sensitive attributes, a fitted `preprocessor.joblib`, and `summary.json`.
`source_row` is the original zero-based CSV record index; exclude it from
model features when loading CSVs (`pd.read_csv(path, index_col='source_row')`).

Preprocessing removes identical records before a stratified 70/15/15 split.
Numeric values are median-imputed and standardized; sex, education, and
marriage are imputed and one-hot encoded. Observed category codes are retained,
including undocumented codes, without assuming their meaning. Negative bill
and repayment-status values are retained. Imputation, scaling, and category
learning use training data only. Sensitive attributes remain in model features
and are also exported unchanged for a later fairness audit.

For model comparison with cross-validation, use the **raw** training split and
`Pipeline([('preprocess', build_preprocessor()), ('model', estimator)])` so each
fold learns its own preprocessing. Use validation for model selection and keep
the test split untouched until final evaluation. No class balancing is applied.
`load_splits()` preserves the existing six-value notebook interface.


## Use the shared data for modeling

The processed splits are included on the preprocessing branch. Replace
`<preprocessing-branch>` below with its branch name. To bring them into your
own working branch, commit or stash local changes, then run:

```powershell
git fetch origin
git merge origin/<preprocessing-branch>
```

From a notebook in `notebooks/`, load features and labels with:

```python
import pandas as pd

X_train = pd.read_csv('../data/processed/X_train.csv', index_col='source_row')
y_train = pd.read_csv('../data/processed/y_train.csv', index_col='source_row').squeeze('columns')
X_val = pd.read_csv('../data/processed/X_val.csv', index_col='source_row')
y_val = pd.read_csv('../data/processed/y_val.csv', index_col='source_row').squeeze('columns')
```

These features are already imputed, encoded, and scaled. For cross-validation,
use `X_train_raw.csv` with `build_preprocessor()` in a model pipeline instead.
Leave the test split for final evaluation after model selection.

## Improve models after the fairness audit

See [FAIRNESS_POLICY.md](FAIRNESS_POLICY.md) for proposed project acceptance
criteria, group definitions, uncertainty checks, and the model selection rule.
The experiment below does not yet implement the full policy.

The completed diagnostic and reweighting follow-up is described in
[FAIRNESS_FOLLOWUP.md](FAIRNESS_FOLLOWUP.md). Open
`notebooks/fairness_followup.ipynb` for saved plots and tables, or run
`python -m src.fairness_followup` after building the baseline artifacts.

Run the comparison against the saved raw training and validation splits:

```powershell
.venv\Scripts\python.exe -m src.model_experiments
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The experiment compares random forest and histogram gradient boosting, each
with and without sex, education, marriage, and age. Each candidate uses a
training-only preprocessing pipeline and three-fold cross-validation to select
parameters by average precision. A shared threshold per candidate is chosen
using training out-of-fold predictions and the audit's exploratory recall >=60%
and FPR <=25% constraints. If no threshold qualifies, the run records `null`
and evaluates 0.5 as a fallback. These constraints still require stakeholder
agreement. Out-of-fold scores reuse folds used for parameter selection, so they
are tuning data, not an unbiased performance estimate.

`outputs/model_improvement/` contains fitted pipelines, CV results, validation
predictions, group rates, a comparison table, and run metadata. The test split
is never loaded. Group comparisons use the audit's minimum support rules.
Removing demographic inputs can leave proxy effects; smaller observed gaps do
not establish fairness. Validation results remain exploratory after repeated
model comparisons. These are independently configured candidates, not exact
reproductions of the original audit models. After integration, the partner's
model configurations are in `src/models.py` and original saved validation
predictions are in `outputs/val_preds_*.csv`.

The shared modeling API remains in `src.models` (`get_models`,
`cross_validate_models`, `tune_models`, `evaluate`, `soft_rank_models`, and
`export_predictions`). The partner's `src/models.py` and
`notebooks/models_and_interp.ipynb` are unchanged from the integrated `main`.
Their notebook uses `load_splits()` and requires the original dataset at
`data/train.csv`, as described in Setup. Our notebooks and experiments use
the saved raw training and validation splits. The separate experiment API is
in `src.model_experiments`; the two modules have different `evaluate`
signatures. The fairness follow-up imports the experiment evaluator explicitly.
