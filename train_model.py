"""End-to-end training pipeline for the KOM1338 Loan Approval competition.

Tunes LightGBM, XGBoost and CatBoost with Optuna, builds out-of-fold (OOF)
predictions for each, blends them by rank, and writes the submission plus all
reproducible artifacts (best params, OOF arrays, fitted-model count, metrics).

Run:  python train_model.py            # full tuning (slow, ~20-40 min)
      python train_model.py --fast     # skip tuning, use validated preset params
      python train_model.py --quick    # few trials, for a fast smoke test

Metric: AUC-ROC (the competition metric). Validation: StratifiedKFold(5).
"""
import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

# Windows consoles default to cp1252; force UTF-8 so progress/summary prints never crash.
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

import src.loan_pipeline as lp

warnings.filterwarnings("ignore")

MODELS_DIR = Path("models")

# Validated preset params (no class weighting; shallow trees). Used by --fast so the
# pipeline can be finalised in minutes; the full Optuna search refines these further.
PRESET_PARAMS = {
    "lightgbm": dict(objective="binary", metric="auc", verbosity=-1, n_estimators=964,
                     learning_rate=0.0101, num_leaves=79, max_depth=4, min_child_samples=69,
                     feature_fraction=0.797, bagging_fraction=0.734, bagging_freq=4,
                     reg_alpha=1.6e-4, reg_lambda=1.2e-3, random_state=lp.SEED),
    "xgboost": dict(objective="binary:logistic", eval_metric="auc", tree_method="hist",
                    n_estimators=1200, learning_rate=0.015, max_depth=4, min_child_weight=5,
                    subsample=0.75, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                    random_state=lp.SEED),
    "catboost": dict(iterations=800, learning_rate=0.03, depth=5, l2_leaf_reg=5.0,
                     random_seed=lp.SEED, verbose=0),
}


def main(quick: bool, fast: bool, no_cat_tune: bool = False):
    n_lgb, n_xgb, n_cat = (5, 5, 5) if quick else (50, 50, 25)
    MODELS_DIR.mkdir(exist_ok=True)

    train, test = lp.load_data(".")
    print(f"Train: {train.shape} | Test: {test.shape}")
    print(f"Positive rate: {train[lp.TARGET].mean():.4f}\n")

    # Two views of the data: one-hot (LGBM/XGB) and native-categorical (CatBoost).
    X_oh, y, X_oh_test = lp.make_onehot(train, test)
    X_nat, _, X_nat_test, cat_features = lp.make_native(train, test)

    # ── Obtain hyperparameters (tuned or preset) ─────────────────────────────────
    if fast:
        print("Fast mode: using validated preset params (skipping Optuna).\n")
        lgb_params, xgb_params, cat_params = (
            PRESET_PARAMS["lightgbm"], PRESET_PARAMS["xgboost"], PRESET_PARAMS["catboost"])
        lgb_cv = xgb_cv = cat_cv = None
    else:
        # Load existing params as fallback (in case process was interrupted mid-tune)
        existing = {}
        if (MODELS_DIR / "best_params.json").exists():
            existing = json.load(open(MODELS_DIR / "best_params.json"))

        print("Tuning LightGBM...")
        lgb_params, lgb_cv = lp.tune_lightgbm(X_oh, y, n_trials=n_lgb)
        print(f"  best CV AUC = {lgb_cv:.5f}\n")
        # Save immediately so progress is not lost if interrupted later
        lp.save_json({**existing, "lightgbm": lgb_params}, MODELS_DIR / "best_params.json")
        existing["lightgbm"] = lgb_params

        print("Tuning XGBoost...")
        xgb_params, xgb_cv = lp.tune_xgboost(X_oh, y, n_trials=n_xgb)
        print(f"  best CV AUC = {xgb_cv:.5f}\n")
        lp.save_json({**existing, "xgboost": xgb_params}, MODELS_DIR / "best_params.json")
        existing["xgboost"] = xgb_params

        if no_cat_tune:
            print("CatBoost: skipping tune, using preset params.\n")
            cat_params, cat_cv = PRESET_PARAMS["catboost"], None
        else:
            print("Tuning CatBoost...")
            cat_params, cat_cv = lp.tune_catboost(X_nat, y, cat_features, n_trials=n_cat)
            print(f"  best CV AUC = {cat_cv:.5f}\n")
        lp.save_json({**existing, "catboost": cat_params}, MODELS_DIR / "best_params.json")

    lp.save_json(
        {"lightgbm": lgb_params, "xgboost": xgb_params, "catboost": cat_params},
        MODELS_DIR / "best_params.json",
    )

    # ── Fit OOF + test predictions with the tuned params ─────────────────────────
    print("LightGBM OOF:")
    oof_lgb, test_lgb, lgb_models = lp.cv_fit_predict(
        lambda: lgb.LGBMClassifier(**lgb_params), X_oh, y, X_oh_test)

    print("XGBoost OOF:")
    oof_xgb, test_xgb, xgb_models = lp.cv_fit_predict(
        lambda: xgb.XGBClassifier(**xgb_params), X_oh, y, X_oh_test)

    print("CatBoost OOF:")
    oof_cat, test_cat, cat_models = lp.cv_fit_predict(
        lambda: CatBoostClassifier(**cat_params), X_nat, y, X_nat_test,
        cat_features=cat_features)

    # ── Blend ────────────────────────────────────────────────────────────────────
    oof_list = [oof_lgb, oof_xgb, oof_cat]
    test_list = [test_lgb, test_xgb, test_cat]
    weights, blend_auc = lp.search_blend_weights(oof_list, y)

    singles = {
        "lightgbm": roc_auc_score(y, oof_lgb),
        "xgboost": roc_auc_score(y, oof_xgb),
        "catboost": roc_auc_score(y, oof_cat),
    }
    print("\n── Results ──")
    for k, v in singles.items():
        print(f"  {k:10s} OOF AUC = {v:.5f}")
    print(f"  blend {weights} OOF AUC = {blend_auc:.5f}")

    # Final test probabilities use the same rank-blend recipe as OOF.
    test_blend = lp.rank_blend(test_list, weights)

    # ── Persist artifacts ────────────────────────────────────────────────────────
    for name, models in [("lgb", lgb_models), ("xgb", xgb_models), ("cat", cat_models)]:
        with open(MODELS_DIR / f"{name}_models.pkl", "wb") as f:
            pickle.dump(models, f)
    np.savez(MODELS_DIR / "oof_predictions.npz",
             lgb=oof_lgb, xgb=oof_xgb, cat=oof_cat, y=y)
    lp.save_json(
        {"single": singles, "blend_weights": list(weights),
         "blend_oof_auc": blend_auc,
         "cv_auc": {"lightgbm": lgb_cv, "xgboost": xgb_cv, "catboost": cat_cv}},
        MODELS_DIR / "metrics.json",
    )

    sub = lp.make_submission(test, test_blend, "submission.csv")
    print(f"\nSubmission saved -> submission.csv ({len(sub)} rows)")
    print(sub.head())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip tuning, use preset params")
    ap.add_argument("--quick", action="store_true", help="few trials, fast smoke test")
    ap.add_argument("--no-cat-tune", action="store_true", help="tune LGB+XGB only, CatBoost uses preset")
    args = ap.parse_args()
    main(args.quick, args.fast, args.no_cat_tune)
