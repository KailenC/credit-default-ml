import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score

from src.config import SEED
from preprocess import build_preprocessor

def get_models():
    return {
        "logreg": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
        ]),
        "random_forest": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=1)),
        ]),
        "hist_bg": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", HistGradientBoostingClassifier(random_state=SEED)),
        ]),
    }

def cross_validate_models(models, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, model in models.items():
        res = cross_validate(model, X, y, cv=cv, n_jobs=1, scoring=["roc_auc", "average_precision", "f1"])

        rows.append({
            "model": name,
            "cv_auc": res["test_roc_auc"].mean(),
            "cv_pr_auc": res["test_average_precision"].mean(),
            "cv_f1": res["test_f1"].mean(),
        })
    return pd.DataFrame(rows).set_index("model").round(3)

def evaluate(model, X, y, threshold=0.5):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y, probs),
        "pr_auc": average_precision_score(y, probs),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds),
    }