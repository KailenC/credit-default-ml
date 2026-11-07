import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import SEED, DATA_PATH, TARGET

def load_splits(path=DATA_PATH, seed=SEED):
    df = pd.read_csv(path)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp)
    return X_train, X_val, X_test, y_train, y_val, y_test