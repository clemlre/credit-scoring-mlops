"""Comparaison des fonctions d'activation d'un MLP (scikit-learn).

Le dataset est tabulaire, donc le boosting reste meilleur — ce MLP sert à
comparer les fonctions d'activation. On garde la même architecture et on ne
change que `activation` parmi identity / logistic / tanh / relu. Régularisation :
pénalité L2 (`alpha`) + early stopping sur une fraction de validation interne.

On sous-échantillonne le train (rapide, et suffisant pour comparer) et on évalue
sur un hold-out stratifié. Chaque essai est logué dans MLflow.

Lancement : .\.venv\Scripts\python.exe src\run_mlp_activations.py [n_subsample]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import mlflow
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import training

OUTPUT_DIR = training.PROJECT_ROOT / "output"
ACTIVATIONS = ["identity", "logistic", "tanh", "relu"]


def build_mlp(activation: str) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation=activation,
            alpha=1e-3,                 # régularisation L2
            early_stopping=True,        # arrêt anticipé sur validation interne
            validation_fraction=0.1,
            n_iter_no_change=25,        # patience large : on laisse converger chaque activation
            max_iter=300,
            learning_rate_init=5e-4,
            batch_size=256,
            random_state=42,
        )),
    ])


def main(n_subsample: int = 60000) -> None:
    training.setup_mlflow()
    X, y = training.load_training_data()

    if n_subsample and n_subsample < len(X):
        X, _, y, _ = train_test_split(X, y, train_size=n_subsample, stratify=y, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Train {X_tr.shape} / test {X_te.shape}, y_mean={y.mean():.4f}", flush=True)

    results, loss_rows = [], []
    for act in ACTIVATIONS:
        print(f"\n[{act}]", flush=True)
        model = build_mlp(act)
        with mlflow.start_run(run_name=f"mlp_{act}"):
            model.fit(X_tr, y_tr)
            clf = model.named_steps["clf"]
            proba = model.predict_proba(X_te)[:, 1]
            pred = (proba >= 0.5).astype(int)
            auc = roc_auc_score(y_te, proba)
            rec = recall_score(y_te, pred)
            cost = training.business_cost(y_te.values, pred)
            n_iter = int(clf.n_iter_)
            best_val = float(max(clf.validation_scores_)) if clf.validation_scores_ else float("nan")

            mlflow.log_params({"model": "MLPClassifier", "activation": act,
                               "hidden_layer_sizes": "(64, 32)", "alpha": 1e-3,
                               "early_stopping": True, "n_subsample": int(len(X))})
            mlflow.log_metrics({"auc": auc, "recall_minority": rec, "business_cost": cost,
                                "n_iter": n_iter, "best_val_score": best_val})
            mlflow.set_tags({"step": "2", "family": "neural-net", "activation": act})

            print(f"  AUC={auc:.4f}  recall={rec:.3f}  cost={cost}  n_iter={n_iter}", flush=True)
            results.append({"activation": act, "auc": auc, "recall_minority": rec,
                            "business_cost": cost, "n_iter": n_iter, "best_val_score": best_val})
            for i, loss in enumerate(clf.loss_curve_, start=1):
                loss_rows.append({"activation": act, "epoch": i, "loss": loss})

    pd.DataFrame(results).to_csv(OUTPUT_DIR / "mlp_activation_results.csv", index=False)
    pd.DataFrame(loss_rows).to_csv(OUTPUT_DIR / "mlp_loss_curves.csv", index=False)
    print("\nRésultats -> output/mlp_activation_results.csv", flush=True)
    print("Fini.", flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    main(n)
