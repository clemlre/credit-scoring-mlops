"""Helpers MLFlow : setup tracking + boucle CV qui log un run par modèle.

Le backend est SQLite (mlruns.db) + artifacts locaux dans mlartifacts/.
Ça nous permettra d'utiliser le Model Registry plus tard sans migration.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH = PROJECT_ROOT / "output" / "feature_dataset.parquet"
MLRUNS_DB = PROJECT_ROOT / "mlruns.db"
MLARTIFACTS_DIR = PROJECT_ROOT / "mlartifacts"

TRACKING_URI = f"sqlite:///{MLRUNS_DB.as_posix()}"
ARTIFACT_URI = MLARTIFACTS_DIR.as_uri()
EXPERIMENT_NAME = "credit-default"


def setup_mlflow(experiment: str = EXPERIMENT_NAME) -> str:
    MLARTIFACTS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    existing = mlflow.get_experiment_by_name(experiment)
    if existing is None:
        mlflow.create_experiment(experiment, artifact_location=ARTIFACT_URI)
    mlflow.set_experiment(experiment)
    return TRACKING_URI


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PARQUET_PATH)
    mask = df["TARGET"].notna()
    y = df.loc[mask, "TARGET"].astype(int).reset_index(drop=True)
    X = df.loc[mask].drop(columns=["TARGET", "SK_ID_CURR"]).reset_index(drop=True)
    del df
    # Colonnes object résiduelles (par sécurité — Aguiar a déjà encodé)
    obj_cols = X.select_dtypes(include="object").columns
    if len(obj_cols):
        X = X.drop(columns=obj_cols)
    # Downcast float64 -> float32 pour réduire l'empreinte mémoire de moitié
    f64 = X.select_dtypes(include="float64").columns
    X[f64] = X[f64].astype("float32")
    # Les features de ratio (PAYMENT_RATE, *_PERC, var aggregations…) peuvent contenir ±inf
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y

# Definition du business cost : 10 * FN_count + 1 * FP_count
# FN => mauvais client classé bon  -> ×10
# FP => bon client classé mauvais  -> ×1
# On préfère perdre un potentiel bon client et perdre les intéret du crédit plutot qu'un mauvais client flaggué comme bon et perdre TOUT le capital prété

def business_cost(y_true: np.ndarray, y_pred: np.ndarray, fn_weight: int = 10, fp_weight: int = 1) -> int:
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return int(fn_weight * fn + fp_weight * fp)


def cv_run(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    run_name: str,
    n_splits: int = 3,
    threshold: float = 0.5,
    extra_params: dict | None = None,
    extra_tags: dict | None = None,
    use_eval_set: bool = False,
    fit_kwargs: dict | None = None,
) -> dict:
    """Tourne une StratifiedKFold, log un run MLFlow agrégé, renvoie les métriques moyennes."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, recalls, f1s, costs = [], [], [], []

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"cv_splits": n_splits, "threshold": threshold})
        if extra_params:
            mlflow.log_params(extra_params)
        if extra_tags:
            mlflow.set_tags(extra_tags)

        for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
            X_tr, y_tr = X.iloc[tr], y.iloc[tr]
            X_va, y_va = X.iloc[va], y.iloc[va]
            fit_kw = dict(fit_kwargs or {})
            if use_eval_set:
                fit_kw["eval_set"] = [(X_va, y_va)]
            model.fit(X_tr, y_tr, **fit_kw)
            proba = model.predict_proba(X_va)[:, 1]
            pred = (proba >= threshold).astype(int)

            auc = roc_auc_score(y.iloc[va], proba)
            rec = recall_score(y.iloc[va], pred)
            f1 = f1_score(y.iloc[va], pred)
            cost = business_cost(y.iloc[va].values, pred)

            aucs.append(auc); recalls.append(rec); f1s.append(f1); costs.append(cost)
            best_iter = getattr(model, "best_iteration_", None) or getattr(getattr(model, "best_iteration", None), "__int__", lambda: None)()
            log_extra = {f"fold{fold}_auc": auc, f"fold{fold}_recall_minority": rec, f"fold{fold}_f1": f1, f"fold{fold}_business_cost": cost}
            if isinstance(best_iter, int):
                log_extra[f"fold{fold}_best_iter"] = best_iter
            mlflow.log_metrics(log_extra)
            iter_str = f"  best_iter={best_iter}" if isinstance(best_iter, int) else ""
            print(f"  fold {fold}/{n_splits}  AUC={auc:.4f}  recall={rec:.3f}  cost={cost}{iter_str}", flush=True)

        metrics = {
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "recall_minority_mean": float(np.mean(recalls)),
            "f1_mean": float(np.mean(f1s)),
            "business_cost_mean": float(np.mean(costs)),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")
        run_id = mlflow.active_run().info.run_id

    metrics["run_id"] = run_id
    print(f"  -> AUC moyen {metrics['auc_mean']:.4f}, cost moyen {metrics['business_cost_mean']:.0f}", flush=True)
    return metrics
