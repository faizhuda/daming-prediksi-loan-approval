"""Core, reusable pipeline logic for the KOM1338 Loan Approval task.

This module is the single source of truth for data loading, feature
engineering, cross-validation, hyperparameter tuning and out-of-fold (OOF)
ensembling. Both ``train_model.py`` (the CLI runner) and ``notebook.ipynb``
(the academic write-up) import from here so the two never drift apart.

Design decisions (justified empirically in the notebook):
  * No class weighting. AUC is a ranking metric; re-weighting the loss
    distorts probabilities away from what AUC measures and was shown to
    *lower* OOF AUC on this data (0.9314 balanced -> 0.9319 unweighted).
  * Minimal feature set. The signal is dominated by a few strong features
    (loan_grade, loan_int_rate, loan_percent_income, home ownership,
    prior default). Adding many engineered features overfits and lowered
    OOF AUC, so we keep only one non-redundant interaction.
  * Tree ensemble. LightGBM + XGBoost + CatBoost are blended; their errors
    are decorrelated enough that the blend edges out any single model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# ── Constants ───────────────────────────────────────────────────────────────────
SEED = 42
N_SPLITS = 5
TARGET = "loan_status"
ID_COL = "sample_id"

# loan_grade is genuinely ordinal (A is best credit, G is worst) and its default
# rate rises monotonically A->G, so an integer encoding is appropriate.
GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}

# Nominal categoricals — one-hot for LGBM/XGBoost, native handling for CatBoost.
CAT_COLS = ["person_home_ownership", "loan_intent", "cb_person_default_on_file"]


# ── Data loading & feature engineering ──────────────────────────────────────────
def load_data(data_dir: str | Path = "."):
    """Read train/test CSVs from ``data_dir``."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered columns and drop the raw ordinal string.

    We deliberately keep this lean: ``grade_ord`` (ordinal encoding) and a
    single interaction ``int_rate_x_grade``. A loan/income ratio was tested
    and dropped — it correlates 0.9995 with the existing
    ``loan_percent_income`` column, so it is pure redundancy.
    """
    df = df.copy()
    df["grade_ord"] = df["loan_grade"].map(GRADE_MAP)
    df["int_rate_x_grade"] = df["loan_int_rate"] * df["grade_ord"]
    return df.drop(columns=["loan_grade"])


def make_onehot(train: pd.DataFrame, test: pd.DataFrame):
    """Feature matrices with one-hot categoricals (for LightGBM / XGBoost)."""
    tr = pd.get_dummies(add_features(train), columns=CAT_COLS)
    te = pd.get_dummies(add_features(test), columns=CAT_COLS)
    feat_cols = tr.columns.drop([TARGET, ID_COL])
    te = te.reindex(columns=feat_cols, fill_value=0)  # align any missing dummies
    X = tr[feat_cols].copy()
    y = tr[TARGET].values
    return X, y, te


def make_native(train: pd.DataFrame, test: pd.DataFrame):
    """Feature matrices with raw string categoricals (for CatBoost)."""
    tr = add_features(train)
    te = add_features(test)
    feat_cols = [c for c in tr.columns if c not in (TARGET, ID_COL)]
    X = tr[feat_cols].copy()
    te = te[feat_cols].copy()
    for c in CAT_COLS:  # CatBoost wants categoricals as strings
        X[c] = X[c].astype(str)
        te[c] = te[c].astype(str)
    y = tr[TARGET].values
    return X, y, te, CAT_COLS


def get_folds():
    """The single, shared CV splitter — identical everywhere for comparability."""
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


# ── Cross-validated fit / predict ───────────────────────────────────────────────
def cv_fit_predict(make_model, X, y, X_test, cat_features=None, verbose=True):
    """Run stratified K-fold, returning OOF preds, averaged test preds, models.

    ``make_model`` is a zero-arg factory so every fold gets a fresh estimator.
    ``cat_features`` (CatBoost only) is forwarded to ``fit``.
    """
    folds = get_folds()
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    models = []
    for k, (tr_idx, va_idx) in enumerate(folds.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y[tr_idx]
        model = make_model()
        if cat_features is not None:
            model.fit(X_tr, y_tr, cat_features=cat_features)
        else:
            model.fit(X_tr, y_tr)
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / folds.n_splits
        models.append(model)
        if verbose:
            print(f"    fold {k}: AUC = {roc_auc_score(y[va_idx], oof[va_idx]):.5f}")
    if verbose:
        print(f"  OOF AUC = {roc_auc_score(y, oof):.5f}")
    return oof, test_pred, models


# ── Optuna objectives (X, y passed explicitly — no global state) ─────────────────
def tune_lightgbm(X, y, n_trials=50, seed=SEED):
    import lightgbm as lgb
    import optuna
    from sklearn.model_selection import cross_val_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = get_folds()

    def objective(trial):
        params = dict(
            objective="binary", metric="auc", verbosity=-1, boosting_type="gbdt",
            n_estimators=trial.suggest_int("n_estimators", 400, 1500),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 16, 128),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 120),
            feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            bagging_freq=trial.suggest_int("bagging_freq", 1, 7),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            random_state=seed,
        )
        model = lgb.LGBMClassifier(**params)
        return cross_val_score(model, X, y, cv=folds, scoring="roc_auc", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = dict(study.best_params, objective="binary", metric="auc",
                verbosity=-1, random_state=seed)
    return best, study.best_value


def tune_xgboost(X, y, n_trials=40, seed=SEED):
    import optuna
    import xgboost as xgb
    from sklearn.model_selection import cross_val_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = get_folds()

    def objective(trial):
        params = dict(
            objective="binary:logistic", eval_metric="auc", tree_method="hist",
            n_estimators=trial.suggest_int("n_estimators", 400, 1500),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            gamma=trial.suggest_float("gamma", 1e-4, 5.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            random_state=seed,
        )
        model = xgb.XGBClassifier(**params)
        return cross_val_score(model, X, y, cv=folds, scoring="roc_auc", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = dict(study.best_params, objective="binary:logistic",
                eval_metric="auc", tree_method="hist", random_state=seed)
    return best, study.best_value


def tune_catboost(X, y, cat_features, n_trials=30, seed=SEED):
    import optuna
    from catboost import CatBoostClassifier
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = get_folds()

    def objective(trial):
        # iterations capped tighter than LGB/XGB: CatBoost CV tuning is the runtime
        # bottleneck, and 300-900 trees already saturate AUC on this dataset.
        params = dict(
            iterations=trial.suggest_int("iterations", 300, 900),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            depth=trial.suggest_int("depth", 3, 7),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
            random_strength=trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
            random_seed=seed, verbose=0,
        )
        scores = []
        for tr_idx, va_idx in folds.split(X, y):
            m = CatBoostClassifier(**params)
            m.fit(X.iloc[tr_idx], y[tr_idx], cat_features=cat_features)
            scores.append(roc_auc_score(y[va_idx], m.predict_proba(X.iloc[va_idx])[:, 1]))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = dict(study.best_params, random_seed=seed, verbose=0)
    return best, study.best_value


# ── Blending ────────────────────────────────────────────────────────────────────
def rank_blend(oof_list, weights):
    """Weighted average of *rank-normalised* predictions (robust to scale)."""
    ranks = [w * (rankdata(o) / len(o)) for w, o in zip(weights, oof_list)]
    return np.sum(ranks, axis=0) / sum(weights)


def search_blend_weights(oof_list, y, grid=(0, 1, 2, 3)):
    """Small grid search over integer weights to maximise OOF AUC."""
    from itertools import product
    best_w, best_auc = None, -1.0
    for w in product(grid, repeat=len(oof_list)):
        if sum(w) == 0:
            continue
        auc = roc_auc_score(y, rank_blend(oof_list, w))
        if auc > best_auc:
            best_auc, best_w = auc, w
    return best_w, best_auc


# ── IO helpers ──────────────────────────────────────────────────────────────────
def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def make_submission(test_df, probs, path="submission.csv"):
    sub = pd.DataFrame({ID_COL: test_df[ID_COL].values, TARGET: probs})
    sub.to_csv(path, index=False)
    return sub
