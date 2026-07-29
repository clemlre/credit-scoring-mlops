r"""Étape 4 — optimisation des hyperparamètres de LightGBM avec Optuna.

Objectif optimisé : le coût métier (10*FN + FP) validé en StratifiedKFold, au
seuil choisi pour chaque fold. Median pruning pour couper les essais faibles.
Le meilleur jeu d'hyperparamètres est réentraîné sur tout le train, le seuil
optimal est balayé proprement en OOF, puis le modèle est enregistré au Model
Registry MLflow.

Lancement : .\.venv\Scripts\python.exe src\optimize_lgbm.py [n_trials]
Les artefacts (étude Optuna, courbe de seuil, courbe d'apprentissage, params)
sont écrits dans output/ pour être relus par le notebook 04.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import training

OUTPUT_DIR = training.PROJECT_ROOT / "output"
OPTUNA_DB = OUTPUT_DIR / "optuna_lgbm.db"
STUDY_NAME = "lgbm_cost"
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
REGISTERED_MODEL = "credit-default-lgbm"
# La recherche Optuna tourne sur un sous-échantillon stratifié (un fit sur 307k
# lignes prend ~5 min, intenable sur 30 essais). Le classement des hyperparamètres
# transfère bien ; le modèle final, lui, est réentraîné sur tout le train.
SEARCH_SUBSAMPLE = 50000


def best_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, int]:
    """Balaye les seuils et renvoie (seuil, coût) minimisant le coût métier."""
    best_t, best_c = 0.5, None
    for t in THRESHOLDS:
        c = training.business_cost(y_true, (proba >= t).astype(int))
        if best_c is None or c < best_c:
            best_t, best_c = float(t), c
    return best_t, int(best_c)


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    params = {
        "n_estimators": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 96),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 60.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.1),
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_costs = []
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        model = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **params)
        model.fit(
            X.iloc[tr], y.iloc[tr],
            eval_set=[(X.iloc[va], y.iloc[va])], eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        proba = model.predict_proba(X.iloc[va])[:, 1]
        _, cost = best_threshold(y.iloc[va].values, proba)
        fold_costs.append(cost)
        trial.report(float(np.mean(fold_costs)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(fold_costs))


def run_study(X: pd.DataFrame, y: pd.Series, n_trials: int) -> optuna.Study:
    OPTUNA_DB.unlink(missing_ok=True)  # repart d'une étude propre
    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage=f"sqlite:///{OPTUNA_DB.as_posix()}",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(lambda t: objective(t, X, y), n_trials=n_trials, show_progress_bar=False)
    return study


def oof_threshold_sweep(model_params: dict, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """OOF propre (3 folds) avec les params optimaux -> coût par seuil."""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        model = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **model_params)
        model.fit(
            X.iloc[tr], y.iloc[tr],
            eval_set=[(X.iloc[va], y.iloc[va])], eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
    rows = [{"threshold": float(t), "business_cost": training.business_cost(y.values, (oof >= t).astype(int))} for t in THRESHOLDS]
    sweep = pd.DataFrame(rows)
    sweep.attrs["auc_oof"] = roc_auc_score(y, oof)
    return sweep


def main(n_trials: int = 30) -> None:
    training.setup_mlflow()
    X, y = training.load_training_data()
    print(f"Données : {X.shape}, y_mean={y.mean():.4f}", flush=True)

    if SEARCH_SUBSAMPLE and SEARCH_SUBSAMPLE < len(X):
        X_search, _, y_search, _ = train_test_split(
            X, y, train_size=SEARCH_SUBSAMPLE, stratify=y, random_state=42)
    else:
        X_search, y_search = X, y
    print(f"\n[Optuna] étude sur {n_trials} essais (objectif = coût métier CV, "
          f"recherche sur {len(X_search)} lignes)", flush=True)
    study = run_study(X_search, y_search, n_trials)
    best = study.best_params
    print(f"  meilleur coût CV : {study.best_value:.0f}", flush=True)
    print(f"  params : {best}", flush=True)

    # Params finaux : on garde n_estimators raisonnable + early stopping sur un holdout.
    final_params = {**best, "n_estimators": 3000}

    # Courbe d'apprentissage (train vs valid) sur un holdout 85/15.
    n = len(X)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    cut = int(0.85 * n)
    tr_idx, va_idx = idx[:cut], idx[cut:]
    evals: dict = {}
    lc_model = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **final_params)
    lc_model.fit(
        X.iloc[tr_idx], y.iloc[tr_idx],
        eval_set=[(X.iloc[tr_idx], y.iloc[tr_idx]), (X.iloc[va_idx], y.iloc[va_idx])],
        eval_names=["train", "valid"], eval_metric=["auc", "binary_logloss"],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0), lgb.record_evaluation(evals)],
    )
    best_iter = lc_model.best_iteration_
    lc = pd.DataFrame({
        "iteration": range(1, len(evals["train"]["auc"]) + 1),
        "train_auc": evals["train"]["auc"],
        "valid_auc": evals["valid"]["auc"],
        "train_logloss": evals["train"]["binary_logloss"],
        "valid_logloss": evals["valid"]["binary_logloss"],
    })
    lc.to_csv(OUTPUT_DIR / "learning_curve.csv", index=False)
    print(f"  best_iteration (holdout) : {best_iter}", flush=True)

    # Seuil optimal en OOF propre, avec n_estimators figé au best_iter.
    fixed = {**best, "n_estimators": int(best_iter)}
    sweep = oof_threshold_sweep(fixed, X, y)
    sweep.to_csv(OUTPUT_DIR / "threshold_sweep.csv", index=False)
    opt_row = sweep.loc[sweep["business_cost"].idxmin()]
    opt_threshold = float(opt_row["threshold"])
    cost_at_opt = int(opt_row["business_cost"])
    cost_at_05 = int(sweep.loc[sweep["threshold"] == 0.5, "business_cost"].iloc[0])
    auc_oof = float(sweep.attrs["auc_oof"])
    print(f"  seuil optimal : {opt_threshold}  coût OOF : {cost_at_opt}  (vs {cost_at_05} au seuil 0.5)", flush=True)
    print(f"  AUC OOF : {auc_oof:.4f}", flush=True)

    # Modèle final réentraîné sur tout le train, n_estimators = best_iter.
    final_model = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **fixed)
    final_model.fit(X, y)

    summary = {
        "best_params": best,
        "best_iteration": int(best_iter),
        "optimal_threshold": opt_threshold,
        "business_cost_optimal_threshold": cost_at_opt,
        "business_cost_threshold_0.5": cost_at_05,
        "auc_oof": auc_oof,
        "n_trials": int(n_trials),
        "optuna_best_cv_cost": float(study.best_value),
    }
    (OUTPUT_DIR / "optuna_best_params.json").write_text(json.dumps(summary, indent=2))

    with mlflow.start_run(run_name="lgbm_tuned") as run:
        mlflow.log_params({**best, "n_estimators": int(best_iter), "threshold": opt_threshold})
        mlflow.log_metrics({
            "auc_oof": auc_oof,
            "business_cost_optimal_threshold": cost_at_opt,
            "business_cost_threshold_0.5": cost_at_05,
            "optimal_threshold": opt_threshold,
            "optuna_best_cv_cost": float(study.best_value),
        })
        mlflow.set_tags({"step": "4", "family": "boosting", "tuned": "optuna"})
        mlflow.lightgbm.log_model(final_model, name="model", registered_model_name=REGISTERED_MODEL)
        run_id = run.info.run_id
    print(f"\nModèle enregistré (run {run_id}) et versionné sous '{REGISTERED_MODEL}'.", flush=True)
    print("Fini.", flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    main(n)
