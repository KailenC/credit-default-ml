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
python -m src.preprocess --input "C:\Users\David\Downloads\05_credit_card_default_train.csv"
```

Outputs go to `data/processed/` (ignored by Git): transformed `X_train`,
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
