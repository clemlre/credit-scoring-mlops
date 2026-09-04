"""Schémas d'entrée et de sortie de l'API (Pydantic).

Ce sont eux qui rendent Swagger utile : chaque champ porte sa description et un
exemple, et FastAPI en dérive la documentation interactive sur /docs.

Choix de validation : les valeurs de features sont acceptées en **types stricts**
(`int` ou `float` JSON), jamais en chaîne. Accepter `"0.5"` obligerait à décider
comment lire `"0,5"`, et une locale mal devinée sur une variable de revenu produit
un score faux sans le moindre message d'erreur. Une chaîne est donc rejetée en 422.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt


# Les exemples publiés dans Swagger doivent être **exécutables** : un « Try it out »
# sur la documentation doit renvoyer 200, pas 422. Un extrait de quelques features ne
# franchit pas le plancher de complétude (50 % des features de dossier), donc chaque
# exemple porte un dossier complet, généré depuis le contrat du modèle.
#
# Deux dossiers sont fournis, qui ne diffèrent que par vingt features : l'un est refusé,
# l'autre accepté. Ils rendent la décision tangible dans la documentation elle-même —
# le seuil de 0,10 cesse d'être une valeur abstraite dès qu'on voit les deux réponses.
#
# `tests/test_api.py` vérifie que chacun produit bien la décision annoncée. Sans ces
# tests, la documentation se périmerait en silence au premier changement de contrat.
def _charger_exemple(nom: str) -> dict[str, float]:
    return json.loads((Path(__file__).parent / nom).read_text(encoding="utf-8"))


EXEMPLE_DOSSIER_REFUSE: dict[str, float] = _charger_exemple("exemple_dossier_refuse.json")
EXEMPLE_DOSSIER_ACCEPTE: dict[str, float] = _charger_exemple("exemple_dossier_accepte.json")

# L'exemple par défaut du schéma reste le dossier refusé : il montre à la fois une
# réponse valide et le fonctionnement du seuil métier.
EXEMPLE_DOSSIER = EXEMPLE_DOSSIER_REFUSE

# Swagger affiche un sélecteur quand plusieurs exemples nommés sont fournis. Le dossier
# refusé reste en tête, donc proposé par défaut.
EXEMPLES_PREDICT: dict[str, dict] = {
    "refuse": {
        "summary": "Dossier refusé — profil à risque",
        "description": (
            "Scores externes bas et mensualité élevée au regard du revenu. "
            "Probabilité attendue autour de 0,24 : au-dessus du seuil de 0,10, "
            "la demande est refusée."
        ),
        "value": {"features": EXEMPLE_DOSSIER_REFUSE},
    },
    "accepte": {
        "summary": "Dossier accepté — profil solide",
        "description": (
            "Mêmes 245 features, dont vingt modifiées : scores externes élevés, "
            "ancienneté professionnelle, crédit plus léger. Probabilité attendue "
            "autour de 0,006, soit quarante fois moins que le dossier refusé."
        ),
        "value": {"features": EXEMPLE_DOSSIER_ACCEPTE},
    },
}

# Valeur d'une feature : un nombre, ou `null` pour « non renseignée ».
# `allow_inf_nan=False` écarte Infinity et NaN, qui traverseraient JSON sans erreur
# mais fausseraient les comparaisons de seuil dans les arbres.
FeatureValue = Annotated[StrictFloat | StrictInt, Field(allow_inf_nan=False)] | None


class PredictionRequest(BaseModel):
    """Une demande de crédit à scorer."""

    model_config = ConfigDict(
        # `model_version` dans les réponses entre en collision avec l'espace de noms
        # protégé `model_` de Pydantic ; on le libère explicitement.
        protected_namespaces=(),
        json_schema_extra={
            "example": {"features": EXEMPLE_DOSSIER}
        },
    )

    features: dict[str, FeatureValue] = Field(
        ...,
        description=(
            "Features du dossier, sous la forme nom → valeur. Les noms doivent "
            "appartenir au contrat du modèle (voir GET /features) ; un nom inconnu "
            "est rejeté. Les features non transmises sont traitées comme manquantes, "
            "ce que le modèle gère nativement — mais le dossier de demande doit être "
            "suffisamment renseigné (voir GET /model/info)."
        ),
    )


class BatchPredictionRequest(BaseModel):
    """Plusieurs demandes à scorer en un appel."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"items": [{"features": EXEMPLE_DOSSIER}]}
        }
    )

    items: list[PredictionRequest] = Field(
        ...,
        min_length=1,
        description="Liste des demandes à scorer. Le nombre maximum est exposé par GET /model/info.",
    )


class CoverageInfo(BaseModel):
    """Sur quelle quantité d'information la prédiction a été calculée.

    Exposé dans chaque réponse pour que l'appelant sache lire le score : une
    probabilité calculée sur un dossier à moitié vide n'a pas le même poids qu'une
    probabilité calculée sur un dossier complet.
    """

    features_provided: int = Field(..., description="Nombre de features renseignées.")
    features_missing: int = Field(..., description="Nombre de features laissées manquantes.")
    application_ratio: float = Field(
        ..., description="Part des features du dossier de demande qui sont renseignées (0 à 1)."
    )
    history_ratio: float = Field(
        ..., description="Part des agrégats d'historique de crédit renseignés (0 à 1)."
    )


class PredictionResponse(BaseModel):
    """Score de risque et décision d'octroi."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "probability": 0.0731,
                "decision": "accepted",
                "threshold": 0.1,
                "model_version": "1",
                "coverage": {
                    "features_provided": 583,
                    "features_missing": 196,
                    "application_ratio": 0.887,
                    "history_ratio": 0.686,
                },
            }
        },
    )

    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilité estimée que le client fasse défaut.",
    )
    decision: Literal["accepted", "rejected"] = Field(
        ...,
        description=(
            "Décision d'octroi obtenue en comparant la probabilité au seuil métier. "
            "`rejected` dès que la probabilité atteint le seuil."
        ),
    )
    threshold: float = Field(
        ...,
        description=(
            "Seuil de décision appliqué. Il vaut 0,10 et non 0,5 : le coût métier "
            "pénalise un mauvais client accepté dix fois plus qu'un bon client refusé."
        ),
    )
    model_version: str = Field(..., description="Version du modèle ayant produit le score.")
    coverage: CoverageInfo


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class PredictionLogStatus(BaseModel):
    """État du journal des prédictions, exposé pour la supervision."""

    stdout: bool = Field(
        ..., description="Le flux JSON des prédictions sur la sortie standard est actif."
    )
    database: Literal["ready", "disabled", "unavailable"] = Field(
        ...,
        description=(
            "État du stockage PostgreSQL. `disabled` : aucune base configurée "
            "(`DATABASE_URL` absent), ce qui est un mode de fonctionnement normal. "
            "`unavailable` : une base est configurée mais la dernière écriture a "
            "échoué — les prédictions restent servies et tracées sur la sortie "
            "standard, seul le monitoring est dégradé."
        ),
    )
    last_error: str | None = Field(
        None, description="Cause du dernier échec d'écriture, si le stockage est dégradé."
    )


class HealthResponse(BaseModel):
    """État du service, destiné aux sondes de disponibilité."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None
    prediction_log: PredictionLogStatus | None = Field(
        None,
        description=(
            "État de la journalisation des prédictions. Volontairement **sans effet "
            "sur le code de statut** : une base de monitoring en panne ne doit pas "
            "faire retirer l'API du trafic, son métier reste de rendre des scores."
        ),
    )


class ModelInfoResponse(BaseModel):
    """Carte d'identité du modèle servi."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    source_run_id: str
    exported_at: str
    decision_threshold: float
    threshold_rationale: str
    n_features: int
    n_trees: int
    metrics: dict[str, float]
    min_application_coverage: float = Field(
        ...,
        description="Part minimale du dossier de demande exigée pour qu'une prédiction soit rendue.",
    )
    max_batch_size: int


class FeaturesResponse(BaseModel):
    """Contrat d'entrée complet : ce que l'API sait recevoir."""

    n_features: int
    application_features: list[str] = Field(
        ..., description="Features issues du dossier de demande lui-même."
    )
    history_features: list[str] = Field(
        ...,
        description=(
            "Agrégats de l'historique de crédit (bureau, demandes précédentes, "
            "échéanciers). Légitimement absents pour un primo-emprunteur."
        ),
    )


class ErrorResponse(BaseModel):
    """Corps renvoyé pour toute erreur gérée."""

    detail: str = Field(..., description="Message expliquant la cause du refus.")
