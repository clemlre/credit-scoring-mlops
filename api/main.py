"""API de scoring de crédit — Prêt à Dépenser.

Lancement local : uv run uvicorn api.main:app --reload
Documentation : http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Body, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from api import config
from api.model import Coverage, ModelLoadError, Prediction, ScoringModel
from api.schemas import (
    PREDICT_EXAMPLES,
    BatchPredictionRequest,
    BatchPredictionResponse,
    CoverageInfo,
    ErrorResponse,
    FeaturesResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionLogStatus,
    PredictionRequest,
    PredictionResponse,
)
from api.storage import PredictionLog, PredictionRecord
from api.tracking import RequestTrackingMiddleware

logger = logging.getLogger("api")

# Au-delà, les messages d'erreur ne listent que les premiers éléments fautifs.
MAX_REPORTED_ITEMS = 10


def _bounded_list(items: list[str], separator: str) -> str:
    shown = separator.join(items[:MAX_REPORTED_ITEMS])
    if len(items) > MAX_REPORTED_ITEMS:
        shown += f" (et {len(items) - MAX_REPORTED_ITEMS} autres)"
    return shown


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Modèle chargé une seule fois. Artefact absent => mode dégradé (/health en 503)
    # plutôt qu'un conteneur qui redémarre en boucle.
    app.state.model_error = None
    try:
        app.state.model = ScoringModel.load()
        logger.info(
            "Modèle chargé : version %s, %d features, seuil %s",
            app.state.model.version,
            len(app.state.model.feature_names),
            app.state.model.threshold,
        )
    except ModelLoadError as exc:
        app.state.model = None
        app.state.model_error = str(exc)
        logger.error("Démarrage en mode dégradé : %s", exc)

    app.state.prediction_log = PredictionLog(config.DATABASE_URL)
    app.state.prediction_log.open()

    yield

    app.state.prediction_log.close()
    app.state.model = None


app = FastAPI(
    title="API de scoring de crédit — Prêt à Dépenser",
    version="1.0.0",
    description=(
        "Estime la probabilité de défaut d'un demandeur de crédit et rend une "
        "décision d'octroi.\n\n"
        "La décision se prend au seuil **0,10**, et non 0,5 : la métrique métier "
        "pénalise un mauvais client accepté dix fois plus qu'un bon client refusé "
        "(`coût = 10 × FN + 1 × FP`).\n\n"
        "Le modèle attend 779 features agrégées sur l'historique du demandeur. "
        "Toutes ne sont pas obligatoires — voir `GET /features` pour le contrat "
        "complet et `GET /model/info` pour les règles d'acceptation."
    ),
    lifespan=lifespan,
)
app.add_middleware(RequestTrackingMiddleware)


async def get_model(request: Request) -> ScoringModel:
    state = request.app.state
    if state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Modèle indisponible : {state.model_error}",
        )
    return state.model


ModelDependency = Annotated[ScoringModel, Depends(get_model)]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # Le handler par défaut recopie la valeur rejetée : NaN/Infinity rendent alors la
    # réponse non sérialisable (500), et des données client finissent dans les logs.
    problems = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"][1:]) or "corps de la requête"
        problems.append(f"{location} : {error['msg']}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": f"Requête invalide — {_bounded_list(problems, ' ; ')}."},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Erreur non gérée sur %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Erreur interne du service."},
    )


def _validate(model: ScoringModel, features: dict) -> Coverage:
    unknown = model.unknown_features(features)
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"{len(unknown)} feature(s) inconnue(s) du modèle : "
                f"{_bounded_list(unknown, ', ')}. "
                "La liste des features acceptées est exposée par GET /features."
            ),
        )

    out_of_range = model.out_of_range_features(features)
    if out_of_range:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"{len(out_of_range)} valeur(s) hors plage : "
                f"{_bounded_list(out_of_range, '; ')}."
            ),
        )

    coverage = model.coverage(features)
    if coverage.application_ratio < config.MIN_APPLICATION_COVERAGE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Dossier de demande trop incomplet : {coverage.application_ratio:.1%} "
                f"des features du dossier sont renseignées, minimum requis "
                f"{config.MIN_APPLICATION_COVERAGE:.0%}. Un score calculé sur si peu "
                "d'information n'aurait pas de valeur métier. Les agrégats "
                "d'historique de crédit, eux, peuvent rester absents."
            ),
        )
    return coverage


def _prediction_records(
    request: Request,
    endpoint: str,
    rows: list[dict],
    predictions: list[Prediction],
    model: ScoringModel,
    latency_ms: float,
) -> list[PredictionRecord]:
    # Écrits par RequestTrackingMiddleware une fois la réponse envoyée. Pour un lot,
    # la latence est répartie sur ses dossiers.
    per_row_ms = latency_ms / len(predictions)
    now = datetime.now(UTC)
    return [
        PredictionRecord(
            request_id=request.state.request_id,
            occurred_at=now,
            endpoint=endpoint,
            model_version=model.version,
            threshold=model.threshold,
            probability=p.probability,
            decision=p.decision,
            features_provided=p.coverage.provided,
            features_missing=p.coverage.missing,
            application_ratio=p.coverage.application_ratio,
            history_ratio=p.coverage.history_ratio,
            latency_ms=per_row_ms,
            features=features,  # payload tel que reçu, pas la ligne complétée à 779 colonnes
        )
        for features, p in zip(rows, predictions, strict=True)
    ]


def _to_response(prediction: Prediction, model: ScoringModel) -> PredictionResponse:
    return PredictionResponse(
        probability=prediction.probability,
        decision=prediction.decision,
        threshold=model.threshold,
        model_version=model.version,
        coverage=CoverageInfo(
            features_provided=prediction.coverage.provided,
            features_missing=prediction.coverage.missing,
            application_ratio=round(prediction.coverage.application_ratio, 4),
            history_ratio=round(prediction.coverage.history_ratio, 4),
        ),
    )


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Service"],
    summary="État du service",
    responses={503: {"model": HealthResponse, "description": "Modèle non chargé."}},
)
def health(request: Request, response: Response) -> HealthResponse:
    """Répond 503 tant que le modèle n'est pas chargé.

    L'état du journal des prédictions est rapporté mais n'influence pas le code de
    statut : une base de monitoring en panne ne doit pas retirer l'API du trafic.
    """
    state = request.app.state
    log_status = PredictionLogStatus(**state.prediction_log.status())
    if state.model is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="degraded", model_loaded=False, prediction_log=log_status)
    return HealthResponse(
        status="ok", model_loaded=True, model_version=state.model.version, prediction_log=log_status
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Modèle"],
    summary="Carte d'identité du modèle servi",
    responses={503: {"model": ErrorResponse, "description": "Modèle non chargé."}},
)
def model_info(model: ModelDependency) -> ModelInfoResponse:
    """Version, provenance, performances et règles d'acceptation du modèle en service."""
    return ModelInfoResponse(
        **model.metadata,
        min_application_coverage=config.MIN_APPLICATION_COVERAGE,
        max_batch_size=config.MAX_BATCH_SIZE,
    )


@app.get(
    "/features",
    response_model=FeaturesResponse,
    tags=["Modèle"],
    summary="Contrat d'entrée : features acceptées",
    responses={503: {"model": ErrorResponse, "description": "Modèle non chargé."}},
)
def list_features(model: ModelDependency) -> FeaturesResponse:
    """Liste des features acceptées, séparées entre dossier de demande et historique.

    Seules les features du dossier conditionnent l'acceptation d'une requête ; les
    agrégats d'historique peuvent manquer.
    """
    return FeaturesResponse(
        n_features=len(model.feature_names),
        application_features=sorted(model.application_features),
        history_features=sorted(model.history_features),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prédiction"],
    summary="Scorer une demande de crédit",
    responses={
        422: {"model": ErrorResponse, "description": "Entrée invalide ou dossier trop incomplet."},
        503: {"model": ErrorResponse, "description": "Modèle non chargé."},
    },
)
def predict(
    payload: Annotated[PredictionRequest, Body(openapi_examples=PREDICT_EXAMPLES)],
    model: ModelDependency,
    request: Request,
) -> PredictionResponse:
    """Renvoie la probabilité de défaut et la décision d'octroi au seuil métier."""
    coverage = _validate(model, payload.features)
    start = time.perf_counter()
    prediction = model.predict([payload.features], [coverage])[0]
    latency_ms = (time.perf_counter() - start) * 1000

    request.state.predictions = _prediction_records(
        request, "/predict", [payload.features], [prediction], model, latency_ms
    )
    return _to_response(prediction, model)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prédiction"],
    summary="Scorer plusieurs demandes en un appel",
    responses={
        422: {"model": ErrorResponse, "description": "Entrée invalide, lot vide ou trop grand."},
        503: {"model": ErrorResponse, "description": "Modèle non chargé."},
    },
)
def predict_batch(
    payload: BatchPredictionRequest,
    model: ModelDependency,
    request: Request,
) -> BatchPredictionResponse:
    """Score un lot de demandes en un seul appel au modèle (taille plafonnée)."""
    if len(payload.items) > config.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Lot de {len(payload.items)} demandes, maximum autorisé "
                f"{config.MAX_BATCH_SIZE}."
            ),
        )

    rows, coverages = [], []
    for position, item in enumerate(payload.items):
        try:
            coverages.append(_validate(model, item.features))
        except HTTPException as exc:
            raise HTTPException(
                status_code=exc.status_code, detail=f"Demande n°{position} : {exc.detail}"
            ) from exc
        rows.append(item.features)

    start = time.perf_counter()
    predictions = model.predict(rows, coverages)
    latency_ms = (time.perf_counter() - start) * 1000

    request.state.predictions = _prediction_records(
        request, "/predict/batch", rows, predictions, model, latency_ms
    )
    return BatchPredictionResponse(
        predictions=[_to_response(p, model) for p in predictions]
    )
