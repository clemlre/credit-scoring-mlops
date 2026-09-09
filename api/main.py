"""API de scoring de crédit — Prêt à Dépenser.

Lancement local : uv run uvicorn api.main:app --reload
Documentation : http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated

from fastapi import BackgroundTasks, Body, Depends, FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.background import BackgroundTask

from api import config
from api.model import ModelLoadError, Prediction, ScoringModel
from api.schemas import (
    EXEMPLES_PREDICT,
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
from api.storage import PredictionLog, RequestRecord, build_record

logger = logging.getLogger("api")

MAX_REPORTED_UNKNOWN = 10
MAX_REPORTED_VALIDATION_ERRORS = 10

# Routes dont chaque appel est journalisé. /health en est exclu : la sonde du
# conteneur l'appelle toutes les 30 s et noierait le taux d'erreur.
TRACKED_PATHS = frozenset({"/predict", "/predict/batch", "/model/info", "/features"})


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Modèle chargé une seule fois. Artefact absent => mode dégradé (/health en 503)
    # plutôt qu'un conteneur qui redémarre en boucle.
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


def _journaliser_requete(journal, record: RequestRecord) -> None:
    try:
        journal.record_request(record)
    except Exception:
        logger.exception("Journalisation de la requête impossible")


@app.middleware("http")
async def tag_request(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    debut = time.perf_counter()
    suivie = request.url.path in TRACKED_PATHS

    try:
        response = await call_next(request)
    except Exception:
        # L'exception remonte jusqu'au handler 500 : on trace l'appel avant.
        if suivie:
            record = _request_record(request, 500, debut)
            await run_in_threadpool(_journaliser_requete, request.app.state.prediction_log, record)
        raise

    response.headers["X-Request-ID"] = request.state.request_id
    if suivie:
        record = _request_record(request, response.status_code, debut)
        response.background = BackgroundTask(
            _journaliser_requete, request.app.state.prediction_log, record
        )
    return response


def _request_record(request: Request, status_code: int, debut: float) -> RequestRecord:
    return RequestRecord(
        request_id=request.state.request_id,
        occurred_at=datetime.now(UTC),
        method=request.method,
        path=request.url.path,
        status_code=status_code,
        duration_ms=(time.perf_counter() - debut) * 1000,
    )


def get_model(request: Request) -> ScoringModel:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Modèle indisponible : "
                f"{getattr(request.app.state, 'model_error', 'cause inconnue')}"
            ),
        )
    return model


ModelDependency = Annotated[ScoringModel, Depends(get_model)]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # Le handler par défaut recopie la valeur rejetée : NaN/Infinity rendent alors la
    # réponse non sérialisable (500), et des données client finissent dans les logs.
    problems = []
    for error in exc.errors()[:MAX_REPORTED_VALIDATION_ERRORS]:
        emplacement = ".".join(str(part) for part in error["loc"][1:]) or "corps de la requête"
        problems.append(f"{emplacement} : {error['msg']}")

    total = len(exc.errors())
    extra = (
        f" (et {total - MAX_REPORTED_VALIDATION_ERRORS} autres)"
        if total > MAX_REPORTED_VALIDATION_ERRORS
        else ""
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": f"Requête invalide — {' ; '.join(problems)}{extra}."},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Erreur non gérée sur %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Erreur interne du service."},
    )


def _validate(model: ScoringModel, features: dict) -> None:
    unknown = model.unknown_features(features)
    if unknown:
        shown = ", ".join(unknown[:MAX_REPORTED_UNKNOWN])
        extra = (
            f" (et {len(unknown) - MAX_REPORTED_UNKNOWN} autres)"
            if len(unknown) > MAX_REPORTED_UNKNOWN
            else ""
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"{len(unknown)} feature(s) inconnue(s) du modèle : {shown}{extra}. "
                "La liste des features acceptées est exposée par GET /features."
            ),
        )

    out_of_range = model.out_of_range_features(features)
    if out_of_range:
        shown = "; ".join(out_of_range[:MAX_REPORTED_UNKNOWN])
        extra = (
            f" (et {len(out_of_range) - MAX_REPORTED_UNKNOWN} autres)"
            if len(out_of_range) > MAX_REPORTED_UNKNOWN
            else ""
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"{len(out_of_range)} valeur(s) hors plage : {shown}{extra}.",
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


def _journaliser_sans_echec(journal, records) -> None:
    try:
        journal.record(records)
    except Exception:
        logger.exception("Journalisation des prédictions impossible")


def _journaliser(
    request: Request,
    background: BackgroundTasks,
    endpoint: str,
    rows: list[dict],
    predictions: list[Prediction],
    model: ScoringModel,
    latency_ms: float,
) -> None:
    # Écriture après l'envoi de la réponse : le monitoring n'entre pas dans la latence.
    latence_unitaire = latency_ms / len(predictions) if predictions else latency_ms
    records = [
        build_record(
            request_id=getattr(request.state, "request_id", "inconnu"),
            endpoint=endpoint,
            features=features,
            probability=prediction.probability,
            decision=prediction.decision,
            coverage=prediction.coverage,
            model_version=model.version,
            threshold=model.threshold,
            latency_ms=latence_unitaire,
        )
        for features, prediction in zip(rows, predictions)
    ]
    background.add_task(_journaliser_sans_echec, request.app.state.prediction_log, records)


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
def racine() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Service"],
    summary="État du service",
    responses={503: {"model": HealthResponse, "description": "Modèle non chargé."}},
)
def health(request: Request) -> JSONResponse:
    """Répond 503 tant que le modèle n'est pas chargé.

    L'état du journal des prédictions est rapporté mais n'influence pas le code de
    statut : une base de monitoring en panne ne doit pas retirer l'API du trafic.
    """
    journal = getattr(request.app.state, "prediction_log", None)
    etat_journal = PredictionLogStatus(**journal.status()) if journal is not None else None

    model = getattr(request.app.state, "model", None)
    if model is None:
        body = HealthResponse(status="degraded", model_loaded=False, prediction_log=etat_journal)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=body.model_dump()
        )
    body = HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=model.version,
        prediction_log=etat_journal,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=body.model_dump())


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Modèle"],
    summary="Carte d'identité du modèle servi",
    responses={503: {"model": ErrorResponse, "description": "Modèle non chargé."}},
)
def model_info(model: ModelDependency) -> ModelInfoResponse:
    """Version, provenance, performances et règles d'acceptation du modèle en service."""
    meta = model.metadata
    return ModelInfoResponse(
        model_name=meta["model_name"],
        model_version=meta["model_version"],
        source_run_id=meta["source_run_id"],
        exported_at=meta["exported_at"],
        decision_threshold=meta["decision_threshold"],
        threshold_rationale=meta["threshold_rationale"],
        n_features=meta["n_features"],
        n_trees=meta["n_trees"],
        metrics=meta["metrics"],
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
def features(model: ModelDependency) -> FeaturesResponse:
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
    payload: Annotated[PredictionRequest, Body(openapi_examples=EXEMPLES_PREDICT)],
    model: ModelDependency,
    request: Request,
    background: BackgroundTasks,
) -> PredictionResponse:
    """Renvoie la probabilité de défaut et la décision d'octroi au seuil métier."""
    _validate(model, payload.features)
    debut = time.perf_counter()
    prediction = model.predict([payload.features])[0]
    latence_ms = (time.perf_counter() - debut) * 1000

    _journaliser(
        request, background, "/predict", [payload.features], [prediction], model, latence_ms
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
    background: BackgroundTasks,
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

    rows = []
    for position, item in enumerate(payload.items):
        try:
            _validate(model, item.features)
        except HTTPException as exc:
            raise HTTPException(
                status_code=exc.status_code, detail=f"Demande n°{position} : {exc.detail}"
            ) from exc
        rows.append(item.features)

    debut = time.perf_counter()
    predictions = model.predict(rows)
    latence_ms = (time.perf_counter() - debut) * 1000

    _journaliser(request, background, "/predict/batch", rows, predictions, model, latence_ms)
    return BatchPredictionResponse(
        predictions=[_to_response(p, model) for p in predictions]
    )
