"""Étape 3 — comparaison LightGBM (GBDT et GOSS) + XGBoost en 5-fold.

Hyperparamètres LightGBM repris du kernel jsaguiar (bayésiens). XGBoost adapté
avec des valeurs équivalentes. Early stopping sur l'AUC du fold de validation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import training

import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

training.setup_mlflow()
X, y = training.load_training_data()
print(f"Données : {X.shape}, y_mean={y.mean():.4f}", flush=True)

es_lgb = [lgb.early_stopping(stopping_rounds=200, verbose=False), lgb.log_evaluation(period=0)]

# --- 1/3 LightGBM GBDT (Aguiar) -----------------------------------------------
lgb_gbdt = LGBMClassifier(
    n_estimators=10000, learning_rate=0.02, num_leaves=34, max_depth=8,
    colsample_bytree=0.9497036, subsample=0.8715623,
    reg_alpha=0.041545473, reg_lambda=0.0735294,
    min_split_gain=0.0222415, min_child_weight=39.3259775,
    random_state=42, n_jobs=-1, verbose=-1,
)
print("\n[1/3] LightGBM GBDT", flush=True)
training.cv_run(
    lgb_gbdt, X, y, run_name="lgbm_gbdt", n_splits=5,
    use_eval_set=True, fit_kwargs={"eval_metric": "auc", "callbacks": es_lgb},
    extra_params={
        "model": "LightGBM", "boosting_type": "gbdt", "data_sample_strategy": "bagging",
        "n_estimators": 10000, "learning_rate": 0.02, "num_leaves": 34, "max_depth": 8,
        "early_stopping_rounds": 200,
    },
    extra_tags={"step": "3", "family": "boosting"},
)

# --- 2/3 LightGBM GOSS (mentor tip "best sampling") ---------------------------
lgb_goss = LGBMClassifier(
    n_estimators=10000, learning_rate=0.02, num_leaves=34, max_depth=8,
    colsample_bytree=0.9497036,
    data_sample_strategy="goss",  # remplace boosting_type='goss' (déprécié)
    reg_alpha=0.041545473, reg_lambda=0.0735294,
    min_split_gain=0.0222415, min_child_weight=39.3259775,
    random_state=42, n_jobs=-1, verbose=-1,
)
print("\n[2/3] LightGBM GOSS", flush=True)
training.cv_run(
    lgb_goss, X, y, run_name="lgbm_goss", n_splits=5,
    use_eval_set=True, fit_kwargs={"eval_metric": "auc", "callbacks": es_lgb},
    extra_params={
        "model": "LightGBM", "boosting_type": "gbdt", "data_sample_strategy": "goss",
        "n_estimators": 10000, "learning_rate": 0.02, "num_leaves": 34, "max_depth": 8,
        "early_stopping_rounds": 200,
    },
    extra_tags={"step": "3", "family": "boosting", "sampling": "goss"},
)

# --- 3/3 XGBoost --------------------------------------------------------------
xgb = XGBClassifier(
    n_estimators=10000, learning_rate=0.02, max_depth=8,
    subsample=0.87, colsample_bytree=0.95,
    reg_alpha=0.04, reg_lambda=0.07,
    min_child_weight=39, gamma=0.022,
    tree_method="hist",
    eval_metric="auc", early_stopping_rounds=200,
    n_jobs=-1, random_state=42, verbosity=0,
)
print("\n[3/3] XGBoost", flush=True)
training.cv_run(
    xgb, X, y, run_name="xgboost", n_splits=5,
    use_eval_set=True, fit_kwargs={"verbose": False},
    extra_params={
        "model": "XGBoost", "tree_method": "hist",
        "n_estimators": 10000, "learning_rate": 0.02, "max_depth": 8,
        "early_stopping_rounds": 200,
    },
    extra_tags={"step": "3", "family": "boosting"},
)

print("\nFini.", flush=True)
