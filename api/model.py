"""Chargement et inférence du modèle de scoring.

Deux principes structurent ce module :

1. **Le modèle est chargé une seule fois**, au démarrage du service (voir le
   `lifespan` de `api/main.py`). Le recharger à chaque requête coûterait ~1 s de
   lecture et 5 Mo d'allocation par appel, pour un résultat rigoureusement
   identique. `ScoringModel` est immuable après chargement.

2. **La logique métier ne dépend pas de HTTP.** Validation, couverture et
   inférence vivent ici, en Python pur, et se testent sans lever de serveur.
   `api/main.py` ne fait que traduire ces règles en codes de statut.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

from api import config


class ModelLoadError(RuntimeError):
    """L'artefact de modèle est absent ou illisible : le service ne peut pas servir."""


# --- Plages de validité métier ---
#
# Ces bornes ne sont pas inventées : elles ont été MESURÉES sur les 307 507 clients
# du jeu d'entraînement, et aucun ne les viole. Une valeur en dehors ne peut donc pas
# venir d'un dossier réel — c'est une erreur d'appel (unité, signe, échelle), et la
# laisser passer produirait un score confiant à partir d'une donnée absurde.
#
# Le piège à connaître : dans ce jeu de données, les colonnes `DAYS_*` comptent les
# jours *avant* la demande et sont donc **négatives** par construction (`DAYS_BIRTH`
# vaut −9461 pour un client de 26 ans). Une règle naïve « l'âge doit être positif »
# rejetterait 100 % des dossiers valides. `DAYS_EMPLOYED_PERC` est un ratio et non
# une durée : il est exclu de la règle.
RANGE_RULES: tuple[tuple[str, float | None, float | None, str], ...] = (
    ("EXT_SOURCE_", 0.0, 1.0, "score externe normalisé, attendu entre 0 et 1"),
    ("DAYS_", None, 0.0, "nombre de jours avant la demande, attendu négatif ou nul"),
    ("AMT_", 0.0, None, "montant, attendu positif ou nul"),
    ("CNT_", 0.0, None, "effectif, attendu positif ou nul"),
    ("FLAG_", 0.0, 1.0, "indicateur binaire, attendu 0 ou 1"),
)


@dataclass(frozen=True)
class Coverage:
    """Part des features réellement renseignées, ventilée par origine."""

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
    """Modèle de scoring prêt à servir, avec son contrat de features et son seuil."""

    def __init__(self, booster: lgb.Booster, feature_names: list[str], metadata: dict):
        self._booster = booster
        self.feature_names = feature_names
        self.metadata = metadata

        # Index nom -> position, calculé une fois : c'est le cœur du chemin chaud.
        # Une recherche linéaire dans une liste de 779 noms, répétée pour chaque
        # champ de chaque requête, dominerait le temps de réponse.
        self._index = {name: i for i, name in enumerate(feature_names)}

        self._application_features = frozenset(
            name for name in feature_names if not name.startswith(config.HISTORY_PREFIXES)
        )
        self._history_features = frozenset(feature_names) - self._application_features

    # --- Chargement ---

    @classmethod
    def load(cls, model_dir: Path | None = None) -> ScoringModel:
        """Charge l'artefact exporté par `src/export_model.py`.

        Le modèle est lu via `model_str=` et non `model_file=` : le lecteur C++ de
        LightGBM ouvre les fichiers avec l'encodage local et échoue sur un chemin
        contenant un caractère non-ASCII (un accent dans un nom de dossier suffit).
        Passer par Python rend le chargement indépendant du chemin d'installation.
        """
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
            booster = lgb.Booster(model_str=model_file.read_text(encoding="utf-8"))
            feature_names = json.loads(features_file.read_text(encoding="utf-8"))
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, lgb.basic.LightGBMError) as exc:
            raise ModelLoadError(f"Artefact de modèle illisible dans {model_dir} : {exc}") from exc

        # Le fichier de features et le modèle doivent parler du même modèle. S'ils
        # divergent, on servirait des colonnes décalées — des scores plausibles mais
        # faux, le pire des échecs possibles. On refuse de démarrer.
        booster_features = list(booster.feature_name())
        if booster_features != feature_names:
            raise ModelLoadError(
                "Incohérence entre le modèle et feature_names.json "
                f"({len(booster_features)} features dans le modèle, "
                f"{len(feature_names)} dans le fichier). Régénère l'artefact."
            )

        return cls(booster, feature_names, metadata)

    # --- Contrat ---

    @property
    def threshold(self) -> float:
        """Seuil de décision métier (0,10), issu du run d'entraînement."""
        return float(self.metadata["decision_threshold"])

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "inconnue"))

    @property
    def application_features(self) -> frozenset[str]:
        """Features issues du dossier de demande (hors agrégats d'historique)."""
        return self._application_features

    @property
    def history_features(self) -> frozenset[str]:
        return self._history_features

    def unknown_features(self, features: dict) -> list[str]:
        """Noms envoyés qui ne font pas partie du contrat, triés pour être stables.

        On les rejette au lieu de les ignorer : une faute de frappe silencieusement
        absorbée donnerait un score calculé sans la variable que l'appelant croyait
        avoir fournie.
        """
        return sorted(name for name in features if name not in self._index)

    def out_of_range_features(self, features: dict) -> list[str]:
        """Valeurs incompatibles avec la plage observée sur les données réelles.

        Renvoie des messages prêts à afficher, triés pour que deux appels identiques
        produisent le même message d'erreur.
        """
        problems = []
        for name, value in sorted(features.items()):
            if value is None or name not in self._index:
                continue
            for prefix, low, high, explanation in RANGE_RULES:
                if not name.startswith(prefix):
                    continue
                # Les ratios dérivés portent le préfixe sans en avoir la sémantique.
                if name.endswith("_PERC"):
                    continue
                if (low is not None and value < low) or (high is not None and value > high):
                    problems.append(f"{name}={value} ({explanation})")
                break
        return problems

    def coverage(self, features: dict) -> Coverage:
        """Mesure ce qui est réellement renseigné (une valeur `null` ne compte pas)."""
        provided = {name for name, value in features.items() if value is not None}
        app_hits = len(provided & self._application_features)
        hist_hits = len(provided & self._history_features)
        return Coverage(
            provided=app_hits + hist_hits,
            missing=len(self.feature_names) - app_hits - hist_hits,
            application_ratio=app_hits / len(self._application_features),
            history_ratio=hist_hits / len(self._history_features),
        )

    # --- Inférence ---

    def _to_matrix(self, rows: list[dict]) -> np.ndarray:
        """Range les dictionnaires reçus dans la matrice attendue par le modèle.

        Les cases non fournies restent à NaN : c'est exactement ce que LightGBM a vu
        à l'entraînement pour une donnée manquante, et il sait la router dans ses
        arbres. Remplacer par 0 introduirait une valeur fausse et déplacerait la
        prédiction sans prévenir.
        """
        matrix = np.full((len(rows), len(self.feature_names)), np.nan, dtype=np.float64)
        for row_idx, features in enumerate(rows):
            for name, value in features.items():
                if value is not None:
                    matrix[row_idx, self._index[name]] = value
        return matrix

    def predict(self, rows: list[dict]) -> list[Prediction]:
        """Score une ou plusieurs demandes. Les entrées ont déjà été validées."""
        if not rows:
            return []

        probabilities = self._booster.predict(self._to_matrix(rows))
        threshold = self.threshold
        return [
            Prediction(
                probability=float(proba),
                # proba = probabilité de DÉFAUT. Au-dessus du seuil, le risque est
                # jugé trop élevé : la demande est refusée.
                decision="rejected" if proba >= threshold else "accepted",
                coverage=self.coverage(features),
            )
            for proba, features in zip(probabilities, rows)
        ]
