# syntax=docker/dockerfile:1

# Image de l'API de scoring de crédit.
#
# Construction en deux étapes : les dépendances sont résolues et installées dans
# une étape jetable, et seul l'environnement fini est recopié dans l'image livrée.
# Ni uv, ni le cache de paquets, ni les outils de compilation ne se retrouvent
# dans l'image finale.
#
# Ce qui entre dans l'image : `api/` et `models/`, rien d'autre. Les notebooks, les
# scripts d'entraînement, les tests et la documentation n'ont aucune raison d'être
# déployés — ils alourdiraient l'image et élargiraient la surface d'attaque.

# ----------------------------------------------------------------------------
# Étape 1 — installation des dépendances
# ----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

# uv est repris d'une image officielle épinglée plutôt qu'installé par script :
# la version est reproductible et rien n'est téléchargé depuis un shell.
COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Seuls les fichiers de dépendances sont copiés ici : tant qu'ils ne changent pas,
# Docker réutilise cette couche et n'a pas à réinstaller 5 paquets scientifiques
# à chaque modification du code de l'API.
COPY pyproject.toml uv.lock ./

# --frozen : le lock fait foi, aucune résolution silencieuse au build. Si le lock
#            est périmé, le build échoue au lieu de livrer d'autres versions.
# --no-dev  : pytest, ruff et httpx restent hors de l'image de production.
# Le groupe `training` (MLflow, Optuna, SHAP, Jupyter) n'est pas installé non plus :
# il n'est pas activé par défaut.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ----------------------------------------------------------------------------
# Étape 2 — image d'exécution
# ----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

# libgomp1 : LightGBM est compilé avec OpenMP et `import lightgbm` échoue sans
# cette bibliothèque, absente des images slim. C'est la cause la plus fréquente
# d'une image qui se construit parfaitement puis meurt au démarrage.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Le service ne tourne pas en root : une faille dans une dépendance ne doit pas
# donner les pleins pouvoirs sur le conteneur.
RUN useradd --create-home --uid 1000 scoring

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/app/models

COPY --from=builder --chown=scoring:scoring /app/.venv /app/.venv

# Le code servi et l'artefact de modèle. Rien de plus.
COPY --chown=scoring:scoring api/ /app/api/
COPY --chown=scoring:scoring models/credit_default_lgbm.txt /app/models/
COPY --chown=scoring:scoring models/feature_names.json /app/models/
COPY --chown=scoring:scoring models/model_metadata.json /app/models/

USER scoring

EXPOSE 8000

# La sonde interroge la vraie route de disponibilité : elle passe au vert quand le
# modèle est chargé, pas simplement quand le processus écoute.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# Forme shell pour que $PORT soit interprété : les hébergeurs (Cloud Run, Hugging
# Face Spaces, Heroku) imposent le port par cette variable.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
