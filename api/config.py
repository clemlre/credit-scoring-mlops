"""Configuration de l'API, lisible depuis l'environnement.

Tout ce qui peut varier entre le poste de dev, la CI et la production est ici, et
nulle part ailleurs. Rien de secret n'y figure : ce sont des chemins et des seuils.
Les vrais secrets (identifiants de déploiement) restent dans les secrets du pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Emplacement de l'artefact de modèle ---
# Surchargé par MODEL_DIR dans l'image Docker (/app/models).
MODEL_DIR = Path(os.environ.get("MODEL_DIR", PROJECT_ROOT / "models"))
MODEL_FILE = MODEL_DIR / "credit_default_lgbm.txt"
FEATURES_FILE = MODEL_DIR / "feature_names.json"
METADATA_FILE = MODEL_DIR / "model_metadata.json"

# --- Garde-fou de qualité d'entrée ---
# Le modèle consomme 779 features, dont 534 sont des agrégats de l'historique de
# crédit (bureau, crédits précédents, échéanciers…). Ces agrégats peuvent être
# légitimement absents — un primo-emprunteur n'a pas d'historique. En revanche, un
# dossier de demande quasi vide ne doit PAS produire un score : LightGBM saurait
# répondre (il gère nativement les valeurs manquantes), mais la réponse n'aurait
# aucune valeur métier tout en ayant l'apparence d'une prédiction. Mieux vaut un
# refus explicite qu'un score trompeur.
#
# Calibrage mesuré sur les 307 507 clients du jeu d'entraînement :
#   couverture des 245 features "dossier" : moyenne 88,7 %, 1er centile 78,4 %
# Un plancher à 50 % laisse donc une marge confortable au-dessus du pire dossier
# réel observé, tout en rejetant un payload de quelques champs.
MIN_APPLICATION_COVERAGE = float(os.environ.get("MIN_APPLICATION_COVERAGE", "0.5"))

# Préfixes des features issues des tables d'historique (agrégats multi-tables).
# Tout ce qui n'en porte pas vient du dossier de demande lui-même.
HISTORY_PREFIXES = (
    "BURO_",
    "ACTIVE_",
    "CLOSED_",
    "PREV_",
    "APPROVED_",
    "REFUSED_",
    "POS_",
    "INSTAL_",
    "CC_",
)

# --- Limite du mode lot ---
# Borne le coût d'une requête : sans plafond, un client peut réclamer 10 millions de
# scores et saturer le service. 1 000 couvre largement les usages de monitoring.
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "1000"))
