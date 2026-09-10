"""Fixtures partagées.

Aucune donnée client n'est versionnée : les dossiers de test sont générés à partir
du contrat de features. Les tests sur données réelles sont ignorés si le parquet
de la Partie 1 est absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.smoke_test import build_application_file


@pytest.fixture(scope="session")
def model():
    from api.model import ScoringModel

    return ScoringModel.load()


@pytest.fixture(scope="session")
def client():
    """Client de test ; le `with` déclenche le lifespan (chargement du modèle)."""
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_features(model) -> dict[str, float]:
    """Un dossier complet : 100 % des features de demande renseignées."""
    return build_application_file(sorted(model.application_features))


@pytest.fixture
def sparse_features(model) -> dict[str, float]:
    """Un dossier volontairement trop incomplet pour être scoré."""
    return build_application_file(sorted(model.application_features)[:5])


@pytest.fixture(scope="session")
def real_clients(model):
    """25 vrais dossiers de la Partie 1, ou skip si les données sont absentes."""
    import pandas as pd

    from src.export_model import load_verification_sample

    p6_root = os.environ.get("P6_PROJECT_ROOT")
    if not p6_root or not (Path(p6_root) / "output" / "feature_dataset.parquet").exists():
        pytest.skip("données de la Partie 1 absentes (P6_PROJECT_ROOT) — test ignoré")
    X = load_verification_sample(Path(p6_root), model.feature_names).head(25)
    return [
        {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
        for row in X.to_dict(orient="records")
    ]
