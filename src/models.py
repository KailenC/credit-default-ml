import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score

from src.config import SEED
from src.preprocess import build_preprocessor

def get_param_spaces():
    return {
        "logreg": {
            "clf__C": loguniform(1e-3, 1e2),
        },
        "random_forest": {
            "clf__max_depth": [2, 3, 4, 6, 8, 12, None],
            "clf__min_samples_leaf": randint(1, 30),
            "clf__max_features": ["sqrt", 0.3, 0.5, 0.7, 1],
            "clf__n_estimators": randint(200, 500),
        },
        "hist_bg": {
            "clf__learning_rate": loguniform(0.003, 0.2),
            "clf__max_depth": [2, 3, 4, 6],
            "clf__min_samples_leaf": randint(20, 200),
            "clf__l2_regularization": loguniform(1e-3, 10),
            "clf__max_iter": randint(50, 400)
        },
        "decision_tree": {
            "clf__max_depth": randint(2, 13),
            "clf__min_samples_leaf": randint(10, 200),
        },
    }

def get_models():
    return {
        "logreg": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, C=0.005, class_weight="balanced", random_state=SEED)),
        ]),
        "random_forest": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", RandomForestClassifier(n_estimators=426, max_depth=4, max_features=0.5, min_samples_leaf=1, class_weight="balanced", random_state=SEED, n_jobs=1)),
        ]),
        "hist_bg": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", HistGradientBoostingClassifier(l2_regularization=1.64, learning_rate=0.0065, max_depth=3, max_iter=380, min_samples_leaf=147, random_state=SEED)),
        ]),
        "decision_tree": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", DecisionTreeClassifier(max_depth=6, min_samples_leaf=91, class_weight="balanced", random_state=SEED)),
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

def tune_models(models, X, y, n_iter=25, scoring="average_precision"):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    spaces = get_param_spaces()
    tuned, rows = {}, []
    for name, pipe in models.items():
        if name not in spaces:
            continue
        search = RandomizedSearchCV(
            pipe, spaces[name], n_iter=n_iter, scoring=scoring, cv=cv, random_state=SEED, n_jobs=1, refit=True, verbose=1
        )
        search.fit(X, y)
        tuned[name] = search.best_estimator_
        rows.append({"model": name, "best_cv_score": search.best_score_, "best_params": search.best_params_})
    return tuned, pd.DataFrame(rows).set_index("model")

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

def soft_rank_models(table):
    ranks = pd.DataFrame({
        "cv_pr_auc": table["cv_pr_auc"].rank(ascending=False),
        "cv_auc": table["cv_auc"].rank(ascending=False),
    })
    ranks["avg_rank"] = ranks.mean(axis=1)
    print(ranks.sort_values("avg_rank"))
    return ranks.sort_values("avg_rank")

def export_predictions(model, X, y, path, threshold=0.5):
    probs = model.predict_proba(X)[:,1]
    out = pd.DataFrame({"y_true": y, "y_prob": probs, "y_pred": (probs >= threshold).astype(int)}, index=X.index)
    out.index.name = "source_row"
    out.to_csv(path)