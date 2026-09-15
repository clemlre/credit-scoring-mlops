"""Tests des requêtes utilisées par le tableau de bord."""

from __future__ import annotations

import pandas as pd

from monitoring import queries


def test_les_requetes_de_monitoring_construisent_un_resultat(monkeypatch):
    appels = []

    def lire(query, parameters=()):
        appels.append((query, parameters))
        if "DISTINCT model_version" in str(query):
            return pd.DataFrame([{"model_version": "1"}])
        return pd.DataFrame([{"nombre": 1, "probability": 0.2, "EXT_SOURCE_2": 0.4}])

    monkeypatch.setattr(queries, "_read", lire)
    periode = (None, None)

    assert queries.versions() == ["1"]
    assert queries.request_summary(periode)["nombre"] == 1
    assert queries.prediction_summary(periode, "1")["nombre"] == 1
    assert not queries.request_series(periode, "day").empty
    assert not queries.latency_series(periode, "day").empty
    assert not queries.prediction_series(periode, "1", "day").empty
    assert not queries.errors(periode).empty
    assert not queries.production_values(periode, "1", ("EXT_SOURCE_2",)).empty
    assert len(appels) == 8


def test_le_filtre_porte_la_periode_et_la_version():
    where, parameters = queries._where(("2026-01-01", "2026-02-01"), "2")

    assert parameters == ["2026-01-01", "2026-02-01", "2"]
    assert "occurred_at" in str(where)
    assert "model_version" in str(where)
