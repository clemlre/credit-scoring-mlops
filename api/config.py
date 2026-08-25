"""Configuration de l'API, surchargeable par variables d'environnement."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = Path(os.environ.get("MODEL_DIR", PROJECT_ROOT / "models"))
MODEL_FILE = MODEL_DIR / "credit_default_lgbm.txt"
FEATURES_FILE = MODEL_DIR / "feature_names.json"
METADATA_FILE = MODEL_DIR / "model_metadata.json"

# Part minimale des 245 features "dossier" à renseigner pour rendre un score.
# Sur le jeu d'entraînement : couverture moyenne 88,7 %, 1er centile 78,4 %.
MIN_APPLICATION_COVERAGE = float(os.environ.get("MIN_APPLICATION_COVERAGE", "0.5"))

# Préfixes des agrégats d'historique (bureau, crédits précédents, échéanciers).
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

MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "1000"))

# Contient le mot de passe PostgreSQL : jamais de valeur par défaut dans le dépôt.
# Non défini => journalisation sur stdout uniquement.
DATABASE_URL = os.environ.get("DATABASE_URL") or None

DB_POOL_MIN_SIZE = int(os.environ.get("DB_POOL_MIN_SIZE", "1"))
DB_POOL_MAX_SIZE = int(os.environ.get("DB_POOL_MAX_SIZE", "4"))
DB_CONNECT_TIMEOUT = float(os.environ.get("DB_CONNECT_TIMEOUT", "5"))
DB_WRITE_TIMEOUT = float(os.environ.get("DB_WRITE_TIMEOUT", "5"))
