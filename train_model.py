import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Load data ──────────────────────────────────────────────────────────────────
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

TARGET    = "loan_status"
ID_COL    = "sample_id"
GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}

def preprocess(df):
    df = df.copy()

    # Ordinal encode loan_grade
    df["loan_grade"] = df["loan_grade"].map(GRADE_MAP)

    # One-hot encode categoricals
    cat_cols = ["person_home_ownership", "loan_intent", "cb_person_default_on_file"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    # Feature engineering
    df["loan_income_ratio"]  = df["loan_amnt"] / (df["person_income"] + 1)
    df["int_rate_x_grade"]   = df["loan_int_rate"] * df["loan_grade"]

    return df

train_p = preprocess(train)
test_p  = preprocess(test)

# Align columns (test may be missing some dummies)
test_p = test_p.reindex(columns=train_p.columns.drop([TARGET, ID_COL]), fill_value=0)

X = train_p.drop(columns=[TARGET, ID_COL])
y = train_p[TARGET]
X_test = test_p

print(f"Train shape: {X.shape}  |  Test shape: {X_test.shape}")
print(f"Class distribution:\n{y.value_counts(normalize=True).round(3)}")

# ── Optuna tuning ──────────────────────────────────────────────────────────────
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = {
        "objective":       "binary",
        "metric":          "auc",
        "verbosity":       -1,
        "boosting_type":   "gbdt",
        "n_estimators":    trial.suggest_int("n_estimators", 300, 1200),
        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves":      trial.suggest_int("num_leaves", 31, 255),
        "max_depth":       trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq":    trial.suggest_int("bagging_freq", 1, 7),
        "reg_alpha":       trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":      trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "class_weight":    "balanced",
        "random_state":    42,
    }
    model = lgb.LGBMClassifier(**params)
    scores = cross_val_score(model, X, y, cv=SKF, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

print("\nRunning Optuna (50 trials)...")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params
best_params.update({"objective": "binary", "metric": "auc", "verbosity": -1,
                    "class_weight": "balanced", "random_state": 42})
print(f"\nBest CV AUC: {study.best_value:.5f}")
print(f"Best params: {best_params}")

# ── OOF predictions + final model ─────────────────────────────────────────────
oof_preds  = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (tr_idx, va_idx) in enumerate(SKF.split(X, y)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=-1)])

    oof_preds[va_idx]  = model.predict_proba(X_va)[:, 1]
    test_preds        += model.predict_proba(X_test)[:, 1] / SKF.n_splits

    fold_auc = roc_auc_score(y_va, oof_preds[va_idx])
    print(f"  Fold {fold+1}: AUC = {fold_auc:.5f}")

final_auc = roc_auc_score(y, oof_preds)
print(f"\nFinal OOF AUC: {final_auc:.5f}")

# ── Save submission ────────────────────────────────────────────────────────────
submission = pd.DataFrame({
    "sample_id":   test[ID_COL],
    "loan_status": test_preds,
})
submission.to_csv("submission.csv", index=False)
print(f"\nSubmission saved -> submission.csv  ({len(submission)} rows)")
print(submission.head())
