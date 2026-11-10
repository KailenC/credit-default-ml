"""Leakage-safe preprocessing. Run: python -m src.preprocess --input path/to/train.csv."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATA_PATH, ROOT, SEED, SENSITIVE, TARGET

MONTHS = ('sep', 'aug', 'jul', 'jun', 'may', 'apr')
CATEGORICAL = ['sex', 'education', 'marriage']
NUMERIC = ['limit_bal', 'age'] + [
    f'{prefix}_{month}'
    for prefix in ('pay', 'bill_amt', 'pay_amt') for month in MONTHS
]
FEATURES = CATEGORICAL + NUMERIC


def load_data(path=DATA_PATH):
    """Validate numeric schema; retain valid negative repayment/bill values."""
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    df = df[FEATURES + [TARGET]].apply(pd.to_numeric, errors='raise')
    if np.isinf(df.to_numpy(dtype=float)).any():
        raise ValueError('Input contains infinite values.')
    if df[TARGET].isna().any() or not df[TARGET].isin([0, 1]).all():
        raise ValueError('Target must contain only 0 and 1, without missing values.')
    # Identical records must not appear on opposite sides of a split.
    return df.drop_duplicates().copy()


def load_splits(path=DATA_PATH, seed=SEED):
    """Return raw 70/15/15 stratified splits, compatible with existing notebooks."""
    df = load_data(path)
    X, y = df.drop(columns=TARGET), df[TARGET].astype(int)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp)
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor():
    """Use inside a model Pipeline during cross-validation to avoid leakage."""
    return ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('scaler', StandardScaler()),
        ]), NUMERIC),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), CATEGORICAL),
    ], verbose_feature_names_out=False)


def preprocess(path=DATA_PATH, output_dir=ROOT / 'data' / 'processed', seed=SEED):
    """Save transformed splits, raw splits, labels, groups, and fitted transformer."""
    splits = load_splits(path, seed)
    transformer = build_preprocessor()
    transformer.fit(splits[0])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {'seed': seed, 'source': Path(path).name, 'splits': {}}
    for name, X, y in zip(('train', 'val', 'test'), splits[:3], splits[3:]):
        transformed = pd.DataFrame(
            transformer.transform(X), index=X.index,
            columns=transformer.get_feature_names_out())
        transformed.to_csv(output_dir / f'X_{name}.csv', index_label='source_row')
        X.to_csv(output_dir / f'X_{name}_raw.csv', index_label='source_row')
        y.to_csv(output_dir / f'y_{name}.csv', index_label='source_row')
        X[SENSITIVE].to_csv(output_dir / f'sensitive_{name}.csv', index_label='source_row')
        summary['splits'][name] = {
            'rows': len(X), 'features': transformed.shape[1],
            'default_rate': float(y.mean()),
        }
    joblib.dump(transformer, output_dir / 'preprocessor.joblib')
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DATA_PATH)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data' / 'processed')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    print(json.dumps(preprocess(args.input, args.output_dir, args.seed), indent=2))
