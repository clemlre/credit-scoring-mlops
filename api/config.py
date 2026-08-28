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

# --- Journalisation des prédictions de production ---
#
# `DATABASE_URL` est la SEULE variable secrète de l'API : elle contient le mot de
# passe PostgreSQL. Elle vient de l'environnement (secret du pipeline, secret du
# Space, variable d'environnement du conteneur) et n'est jamais écrite dans le
# dépôt — d'où l'absence totale de valeur par défaut ici.
#
# Non définie ⇒ la journalisation en base est simplement désactivée, et l'API
# continue d'émettre ses prédictions en JSON sur la sortie standard. C'est le
# comportement voulu en test, en CI et pour un `docker run` de démonstration :
# le service ne doit pas exiger une base pour rendre un score.
DATABASE_URL = os.environ.get("DATABASE_URL") or None

# Le pool reste petit : l'écriture d'une prédiction est brève et a lieu hors du
# chemin de réponse. Un pool large immobiliserait des connexions PostgreSQL — une
# ressource comptée côté serveur — pour rien.
DB_POOL_MIN_SIZE = int(os.environ.get("DB_POOL_MIN_SIZE", "1"))
DB_POOL_MAX_SIZE = int(os.environ.get("DB_POOL_MAX_SIZE", "4"))

# Délais courts et bornés : le monitoring n'a pas le droit d'accumuler des tâches
# d'arrière-plan en attente sur une base qui ne répond plus.
DB_CONNECT_TIMEOUT = float(os.environ.get("DB_CONNECT_TIMEOUT", "5"))
DB_WRITE_TIMEOUT = float(os.environ.get("DB_WRITE_TIMEOUT", "5"))
