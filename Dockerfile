# syntax=docker/dockerfile:1

# Image de l'API de scoring. Seuls api/ et l'artefact du modèle sont copiés.

FROM python:3.11-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock ./

# Dépendances de production uniquement (sans les groupes dev/training/monitoring).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.11-slim-bookworm AS runtime

# LightGBM a besoin d'OpenMP, absent des images slim.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 scoring

WORKDIR /app

# Réglés pour le Space Hugging Face « cpu-basic » (2 vCPU), voir docs/optimisation.md :
# - WEB_CONCURRENCY : un processus uvicorn par vCPU (lu nativement par uvicorn) ;
# - OMP_NUM_THREADS : le conteneur voit tous les cœurs de l'hôte, pas son quota, et
#   LightGBM ouvrirait autant de threads OpenMP pour scorer un seul dossier.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/app/models \
    WEB_CONCURRENCY=2 \
    OMP_NUM_THREADS=1

COPY --from=builder --chown=scoring:scoring /app/.venv /app/.venv
COPY --chown=scoring:scoring api/ /app/api/
COPY --chown=scoring:scoring models/credit_default_lgbm.txt /app/models/
COPY --chown=scoring:scoring models/feature_names.json /app/models/
COPY --chown=scoring:scoring models/model_metadata.json /app/models/

USER scoring

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# Forme shell pour lire $PORT, imposé par certains hébergeurs.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
