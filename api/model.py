"""Chargement du modèle LightGBM et logique de scoring, indépendante de HTTP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

from api import config


class ModelLoadError(RuntimeError):
    """Artefact de modèle absent ou illisible."""


# Bornes mesurées sur les 307 507 clients du jeu d'entraînement : aucun ne les viole.
# Les colonnes DAYS_* sont négatives par construction (jours avant la demande).
RANGE_RULES: tuple[tuple[str, float | None, float | None, str], ...] = (
    ("EXT_SOURCE_", 0.0, 1.0, "score externe normalisé, attendu entre 0 et 1"),
    ("DAYS_", None, 0.0, "nombre de jours avant la demande, attendu négatif ou nul"),
    ("AMT_", 0.0, None, "montant, attendu positif ou nul"),
    ("CNT_", 0.0, None, "effectif, attendu positif ou nul"),
    ("FLAG_", 0.0, 1.0, "indicateur binaire, attendu 0 ou 1"),
)


@dataclass(frozen=True)
class Coverage:
    provided: int
    missing: int
    application_ratio: float
    history_ratio: float


@dataclass(frozen=True)
class Prediction:
    probability: float
    decision: str
    coverage: Coverage


class ScoringModel:
    def __init__(self, booster: lgb.Booster, feature_names: list[str], metadata: dict):
        self._booster = booster
        self.feature_names = feature_names
        self.metadata = metadata
        self._index = {name: i for i, name in enumerate(feature_names)}
        self.application_features = frozenset(
            name for name in feature_names if not name.startswith(config.HISTORY_PREFIXES)
        )
        self.history_features = frozenset(feature_names) - self.application_features
        # Règle de plage de chaque feature, résolue une fois : le contrôle par requête
        # devient une recherche dans un dict au lieu de 5 startswith par valeur.
        self._bounds: dict[str, tuple[float | None, float | None, str]] = {}
        for name in feature_names:
            if name.endswith("_PERC"):
                continue
            for prefix, low, high, explanation in RANGE_RULES:
                if name.startswith(prefix):
                    self._bounds[name] = (low, high, explanation)
                    break

    @classmethod
    def load(cls, model_dir: Path | None = None) -> ScoringModel:
        model_dir = model_dir or config.MODEL_DIR
        model_file = model_dir / config.MODEL_FILE.name
        features_file = model_dir / config.FEATURES_FILE.name
        metadata_file = model_dir / config.METADATA_FILE.name

        missing = [p.name for p in (model_file, features_file, metadata_file) if not p.exists()]
        if missing:
            raise ModelLoadError(
                f"Artefact de modèle incomplet dans {model_dir} : {', '.join(missing)} "
                "introuvable(s). Lance `python src/export_model.py` pour le régénérer."
            )

        try:
            # model_str plutôt que model_file : le lecteur C++ échoue sur un chemin
            # contenant des caractères non ASCII.
            booster = lgb.Booster(model_str=model_file.read_text(encoding="utf-8"))
            feature_names = json.loads(features_file.read_text(encoding="utf-8"))
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, lgb.basic.LightGBMError) as exc:
            raise ModelLoadError(f"Artefact de modèle illisible dans {model_dir} : {exc}") from exc

        booster_features = booster.feature_name()
        if booster_features != feature_names:
            raise ModelLoadError(
                "Incohérence entre le modèle et feature_names.json "
                f"({len(booster_features)} features dans le modèle, "
                f"{len(feature_names)} dans le fichier). Régénère l'artefact."
            )

        return cls(booster, feature_names, metadata)

    @property
    def threshold(self) -> float:
        return float(self.metadata["decision_threshold"])

    @property
    def version(self) -> str:
        return self.metadata["model_version"]

    def unknown_features(self, features: dict) -> list[str]:
        return sorted(name for name in features if name not in self._index)

    def out_of_range_features(self, features: dict) -> list[str]:
        problems = []
        for name, value in features.items():
            bounds = self._bounds.get(name)
            if bounds is None or value is None:
                continue
            low, high, explanation = bounds
            if (low is not None and value < low) or (high is not None and value > high):
                problems.append((name, f"{name}={value} ({explanation})"))
        return [message for _, message in sorted(problems)]

    def coverage(self, features: dict) -> Coverage:
        provided = {name for name, value in features.items() if value is not None}
        app_hits = len(provided & self.application_features)
        hist_hits = len(provided & self.history_features)
        return Coverage(
            provided=app_hits + hist_hits,
            missing=len(self.feature_names) - app_hits - hist_hits,
            application_ratio=app_hits / len(self.application_features),
            history_ratio=hist_hits / len(self.history_features),
        )

    def _to_matrix(self, rows: list[dict]) -> np.ndarray:
        # Les features absentes restent à NaN, comme à l'entraînement : LightGBM
        # route les valeurs manquantes, alors qu'un 0 serait une valeur observée.
        matrix = np.full((len(rows), len(self.feature_names)), np.nan, dtype=np.float64)
        for row_idx, features in enumerate(rows):
            for name, value in features.items():
                if value is not None:
                    matrix[row_idx, self._index[name]] = value
        return matrix

    def predict(
        self, rows: list[dict], coverages: list[Coverage] | None = None
    ) -> list[Prediction]:
        """Score des dossiers déjà validés ; `coverages` évite de recalculer la couverture."""
        if coverages is None:
            coverages = [self.coverage(features) for features in rows]

        probabilities = self._booster.predict(self._to_matrix(rows))
        threshold = self.threshold
        return [
            Prediction(
                probability=float(proba),
                decision="rejected" if proba >= threshold else "accepted",
                coverage=coverage,
            )
            for proba, coverage in zip(probabilities, coverages, strict=True)
        ]
