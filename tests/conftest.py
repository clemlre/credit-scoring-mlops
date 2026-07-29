"""Fixtures partagées par les tests.

Choix structurant : **aucune donnée client n'est versionnée dans le dépôt.** Les
dossiers de test sont générés synthétiquement, de façon déterministe, à partir du
seul contrat de features. Trois raisons :

1. la CI n'a pas accès aux données Kaggle (2,6 Go, hors dépôt) et doit pourtant
   passer ;
2. redistribuer des lignes du dataset Kaggle, même anonymisées, sortirait de ses
   conditions d'utilisation ;
3. un dossier généré est reproductible à l'octet près, donc les tests ne peuvent
   pas échouer « au hasard ».

Les tests qui exigent de vraies données existent quand même, mais sont ignorés
automatiquement si le parquet de la Partie 1 n'est pas là (voir `real_clients`).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def model():
    """Le modèle chargé une seule fois pour toute la session de test."""
    from api.model import ScoringModel

    return ScoringModel.load()


@pytest.fixture(scope="session")
def client():
    """Client HTTP de test.

    Le `with` déclenche le `lifespan` de FastAPI : sans lui, le modèle ne serait
    jamais chargé et toutes les routes répondraient 503.
    """
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


def build_features(model, names=None) -> dict[str, float]:
    """Fabrique un dossier de demande valide, sans copier la moindre donnée réelle.

    Chaque valeur respecte la plage de validité de sa famille (voir `RANGE_RULES`) :
    le dossier est donc accepté par l'API. Les valeurs n'ont aucune prétention
    réaliste — ces tests vérifient le **contrat**, pas la pertinence métier du score.
    """
    names = names if names is not None else sorted(model.application_features)
    values = {}
    for name in names:
        if name.startswith("EXT_SOURCE"):
            values[name] = 0.5
        elif name.startswith("FLAG_"):
            values[name] = 0.0
        elif name.startswith("DAYS_") and not name.endswith("_PERC"):
            values[name] = -5000.0
        elif name.startswith("AMT_"):
            values[name] = 100000.0
        elif name.startswith("CNT_"):
            values[name] = 1.0
        else:
            values[name] = 0.0
    return values


@pytest.fixture
def valid_features(model) -> dict[str, float]:
    """Un dossier complet : 100 % des features de demande renseignées."""
    return build_features(model)


@pytest.fixture
def sparse_features(model) -> dict[str, float]:
    """Un dossier volontairement trop incomplet pour être scoré."""
    names = sorted(model.application_features)[:5]
    return build_features(model, names)


@pytest.fixture(scope="session")
def real_clients():
    """Vrais dossiers issus de la Partie 1, ou `skip` si les données sont absentes.

    Sert aux tests de fidélité numérique : rien ne remplace de vraies distributions
    pour vérifier que l'API reproduit exactement le modèle.
    """
    import numpy as np
    import pandas as pd

    p6_root = Path(
        os.environ.get(
            "P6_PROJECT_ROOT",
            r"C:\Users\ClementLoire\ObsidianVault_MCP_enabled\05 - Cours\P6 - Initiez-vous au MLOps 1-2",
        )
    )
    parquet = p6_root / "output" / "feature_dataset.parquet"
    if not parquet.exists():
        pytest.skip(f"données de la Partie 1 absentes ({parquet}) — test de fidélité ignoré")

    from api.model import ScoringModel

    feature_names = ScoringModel.load().feature_names
    frame = pd.read_parquet(parquet)
    frame = frame[frame["TARGET"].notna()].head(25)
    X = frame.drop(columns=["TARGET", "SK_ID_CURR"])
    X = X.drop(columns=X.select_dtypes(include="object").columns)
    float64_cols = X.select_dtypes(include="float64").columns
    X[float64_cols] = X[float64_cols].astype("float32")
    X = X.replace([np.inf, -np.inf], np.nan)[feature_names]

    return [
        {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
        for row in X.to_dict(orient="records")
    ]
