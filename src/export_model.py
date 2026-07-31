"""Exporte le modèle du registre MLflow de la Partie 1 vers models/.

Produit un artefact autoportant (modèle au format texte LightGBM, liste ordonnée
des features, métadonnées dont le seuil), puis vérifie qu'il prédit exactement
comme le modèle source. Format natif plutôt que pickle : pas besoin de
scikit-learn ni de MLflow dans l'image.

    uv run --group training python src/export_model.py [--p6-root CHEMIN] [--version 1]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

REGISTERED_MODEL = "credit-default-lgbm"

VERIFY_ROWS = 500
VERIFY_TOLERANCE = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p6-root",
        type=Path,
        default=os.environ.get("P6_PROJECT_ROOT"),
        help="racine du projet Partie 1 (contenant mlruns.db et mlartifacts/)",
    )
    parser.add_argument(
        "--version", type=str, default="1", help="version du modèle enregistré à exporter"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="ne pas vérifier numériquement l'artefact (déconseillé)",
    )
    args = parser.parse_args()
    if args.p6_root is None:
        parser.error("indiquer --p6-root ou définir P6_PROJECT_ROOT")
    return args


def load_from_registry(p6_root: Path, version: str):
    """Charge le modèle et son run depuis le registre MLflow de la Partie 1."""
    import mlflow
    from mlflow.tracking import MlflowClient

    db = p6_root / "mlruns.db"
    if not db.exists():
        raise FileNotFoundError(
            f"Registre MLflow introuvable : {db}\n"
            "Indique la racine du projet Partie 1 via --p6-root ou P6_PROJECT_ROOT."
        )

    mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")
    client = MlflowClient()

    model_version = client.get_model_version(REGISTERED_MODEL, version)
    run = client.get_run(model_version.run_id)
    model = mlflow.lightgbm.load_model(f"models:/{REGISTERED_MODEL}/{version}")
    return model, model_version, run


def load_verification_sample(p6_root: Path, feature_names: list[str]) -> pd.DataFrame:
    """Échantillon prétraité comme dans training.load_training_data."""
    parquet = p6_root / "output" / "feature_dataset.parquet"
    if not parquet.exists():
        raise SystemExit(
            f"Parquet absent ({parquet}) : vérification impossible. "
            "Relance avec --skip-verify si tu acceptes un export non vérifié."
        )

    df = pd.read_parquet(parquet)
    df = df[df["TARGET"].notna()].head(VERIFY_ROWS)
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    X = X.drop(columns=X.select_dtypes(include="object").columns)
    f64 = X.select_dtypes(include="float64").columns
    X[f64] = X[f64].astype("float32")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X[feature_names]


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(exist_ok=True)

    print(f"Registre Partie 1 : {args.p6_root}")
    model, model_version, run = load_from_registry(args.p6_root, args.version)
    booster = model.booster_
    feature_names = booster.feature_name()
    print(f"  modèle {REGISTERED_MODEL} v{model_version.version} (run {model_version.run_id[:8]})")
    print(f"  {len(feature_names)} features, {booster.num_trees()} arbres")

    threshold = float(run.data.params.get("threshold", run.data.metrics.get("optimal_threshold")))
    print(f"  seuil de décision : {threshold}")

    # model_to_string() : save_model() échoue sur un chemin non ASCII sous Windows.
    model_path = MODELS_DIR / "credit_default_lgbm.txt"
    model_path.write_text(booster.model_to_string(), encoding="utf-8", newline="\n")
    size_mo = model_path.stat().st_size / 1024 / 1024
    print(f"\n-> {model_path.name}  ({size_mo:.1f} Mo)")

    features_path = MODELS_DIR / "feature_names.json"
    features_path.write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    print(f"-> {features_path.name}  ({len(feature_names)} noms ordonnés)")

    metadata = {
        "model_name": REGISTERED_MODEL,
        "model_version": str(model_version.version),
        "source_run_id": model_version.run_id,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "decision_threshold": threshold,
        "threshold_rationale": (
            "Seuil minimisant le coût métier 10*FN + 1*FP, balayé en out-of-fold. "
            "Un mauvais client accepté coûte 10 fois un bon client refusé."
        ),
        "n_features": len(feature_names),
        "n_trees": booster.num_trees(),
        "metrics": {k: float(v) for k, v in run.data.metrics.items()},
        "hyperparameters": dict(run.data.params),
        "training_environment": {
            "lightgbm": lgb.__version__,
            "python": platform.python_version(),
        },
    }
    metadata_path = MODELS_DIR / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {metadata_path.name}")

    if args.skip_verify:
        print("\n! vérification numérique passée (--skip-verify)")
        return

    print(f"\nVérification sur {VERIFY_ROWS} lignes réelles...")
    sample = load_verification_sample(args.p6_root, feature_names)

    expected = model.predict_proba(sample)[:, 1]
    reloaded = lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))
    actual = reloaded.predict(sample)

    max_diff = float(np.max(np.abs(expected - actual)))
    print(f"  écart max de probabilité : {max_diff:.3e}")
    if max_diff > VERIFY_TOLERANCE:
        raise SystemExit(
            f"ECHEC : l'artefact rechargé ne reproduit pas le modèle source "
            f"(écart {max_diff:.3e} > {VERIFY_TOLERANCE:.0e})."
        )

    disagreements = int(np.sum((expected >= threshold) != (actual >= threshold)))
    print(f"  décisions divergentes au seuil {threshold} : {disagreements}/{len(sample)}")
    if disagreements:
        raise SystemExit("ECHEC : divergence de décision entre modèle source et artefact.")

    accepted = int(np.sum(actual < threshold))
    print(
        f"  répartition sur l'échantillon : {accepted} accordés / "
        f"{len(sample) - accepted} refusés"
    )
    print("\nOK — artefact déployable vérifié.")


if __name__ == "__main__":
    main()
