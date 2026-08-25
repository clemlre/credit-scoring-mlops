"""API de scoring de crédit — « Prêt à Dépenser ».

Sert le modèle de la Partie 1 derrière quelques routes documentées. Le modèle est
chargé **une fois** au démarrage du processus (`lifespan`), jamais pendant une
requête : le recharger coûterait ~1 seconde et 5 Mo par appel, pour un résultat
identique.

Lancement local :
    uv run uvicorn api.main:app --reload
Documentation interactive : http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api import config
from api.model import ModelLoadError, Prediction, ScoringModel
from api.schemas import (
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
from api.storage import PredictionLog, build_record

logger = logging.getLogger("api")

# Nombre maximum de noms de features fautifs listés dans un message d'erreur.
# Un payload truffé de fautes de frappe ne doit pas produire une réponse d'erreur
# de plusieurs centaines de kilo-octets.
MAX_REPORTED_UNKNOWN = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage, une seule fois pour toute la vie du processus.

    Si l'artefact est absent ou corrompu, on démarre quand même, mais en mode
    dégradé : /health répond 503 et le dit. Un conteneur qui redémarre en boucle
    est bien plus difficile à diagnostiquer qu'un service qui répond « voici ce
    qui me manque » — et la sonde de disponibilité le retire du trafic dans les
    deux cas.
    """
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

    # Le journal des prédictions suit le même principe que le modèle : ouvert une
    # fois ici, fermé à l'arrêt. `open()` ne lève jamais — une base injoignable
    # laisse le service opérationnel, avec le seul canal `stdout`.
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


@app.middleware("http")
async def tag_request(request: Request, call_next):
    """Attribue un identifiant unique à chaque requête et le renvoie en en-tête.

    C'est ce qui rend le journal des prédictions exploitable : quand un conseiller
    conteste un refus, il transmet le `X-Request-ID` reçu et l'on retrouve la ligne
    exacte en base — score, seuil, version du modèle et features envoyées. Sans cet
    identifiant, il faudrait chercher par horodatage approximatif.
    """
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


def get_model(request: Request) -> ScoringModel:
    """Fournit le modèle chargé, ou refuse la requête s'il est indisponible."""
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


# Injection du modèle dans les routes. La forme `Annotated` est préférée à
# `= Depends(...)` : elle garde la signature exempte d'appel de fonction en valeur
# par défaut, ce qui la rend réutilisable et analysable par les outils statiques.
ModelDependency = Annotated[ScoringModel, Depends(get_model)]


# Nombre maximum d'erreurs de validation détaillées dans une réponse.
MAX_REPORTED_VALIDATION_ERRORS = 10


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Transforme les erreurs de validation en message lisible, sans réémettre la valeur.

    Le comportement par défaut de FastAPI recopie la valeur rejetée dans le corps de
    la réponse. C'est un problème pour deux raisons :

    1. **Il fait tomber le service.** `Infinity` et `NaN` franchissent le parseur JSON
       de Python mais ne sont pas sérialisables en JSON valide : la réponse 422 lève
       alors une `ValueError` et le client reçoit une erreur 500. Un simple
       `{"AMT_CREDIT": Infinity}` suffisait à provoquer ça.
    2. **Il recopie de la donnée client** dans les réponses d'erreur, donc dans les
       journaux de tout ce qui est en aval. Un revenu ou un identifiant n'ont rien à
       y faire.

    On ne renvoie donc que l'emplacement et la raison — ce dont l'appelant a besoin
    pour corriger sa requête, et rien de plus.
    """
    problems = []
    for error in exc.errors()[:MAX_REPORTED_VALIDATION_ERRORS]:
        # `loc` commence par "body" : on le retire, il n'apprend rien à l'appelant.
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
    """Filet de sécurité : une erreur imprévue ne doit pas fuiter de trace interne.

    On journalise la pile côté serveur (pour le diagnostic) et on ne renvoie au
    client qu'un message neutre — une trace exposerait des chemins et des versions
    de bibliothèques.
    """
    logger.exception("Erreur non gérée sur %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Erreur interne du service."},
    )


def _validate(model: ScoringModel, features: dict) -> None:
    """Applique le contrat d'entrée. Lève une HTTPException 422 si l'entrée le viole."""
    unknown = model.unknown_features(features)
    if unknown:
        shown = ", ".join(unknown[:MAX_REPORTED_UNKNOWN])
        extra = f" (et {len(unknown) - MAX_REPORTED_UNKNOWN} autres)" if len(unknown) > MAX_REPORTED_UNKNOWN else ""
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
    """Ultime filet : aucune exception ne sort d'une tâche de journalisation.

    `PredictionLog.record` absorbe déjà ses propres erreurs, mais l'invariant « le
    monitoring ne casse jamais une prédiction » ne doit pas dépendre de la
    discipline d'une seule classe. Une exception qui remonterait d'ici serait levée
    *après* l'envoi de la réponse : le client aurait son score, mais le serveur
    afficherait une trace et pourrait couper la connexion. On la piège donc ici, une
    fois pour toutes, quel que soit le journal branché.
    """
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
    """Programme l'écriture des prédictions au journal, **après** la réponse.

    `BackgroundTasks` garantit que l'appelant reçoit son score sans attendre
    l'écriture en base : le monitoring n'entre pas dans le temps de réponse. Pour
    un lot, tout part en un seul appel — donc un seul `executemany`.

    Passé une certaine charge, cette approche montrerait ses limites (une tâche par
    requête, sans file d'attente bornée) ; la suite serait une file interne avec un
    consommateur unique, ou un envoi vers un collecteur externe.
    """
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


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Service"],
    summary="État du service",
    responses={503: {"model": HealthResponse, "description": "Modèle non chargé."}},
)
def health(request: Request) -> JSONResponse:
    """Sonde de disponibilité : répond 503 tant que le modèle n'est pas servable.

    L'état du journal des prédictions est rapporté, mais **n'influence pas le code
    de statut** : si PostgreSQL tombe, l'API sait toujours scorer, et la retirer du
    trafic transformerait une panne de monitoring en panne de production.
    """
    journal = getattr(request.app.state, "prediction_log", None)
    etat_journal = PredictionLogStatus(**journal.status()) if journal is not None else None

    model = getattr(request.app.state, "model", None)
    if model is None:
        body = HealthResponse(status="degraded", model_loaded=False, prediction_log=etat_journal)
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=body.model_dump())
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
    """Liste exhaustive des features acceptées, séparées par origine.

    Les features « dossier » viennent de la demande elle-même et conditionnent
    l'acceptation d'une requête ; les features « historique » sont des agrégats
    calculés sur les crédits passés et peuvent légitimement manquer.
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
    payload: PredictionRequest,
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
    """Score un lot de demandes.

    Un seul appel au modèle est fait pour tout le lot : LightGBM vectorise sur la
    matrice entière, ce qui est nettement plus rapide que N appels unitaires.
    Le lot est plafonné pour borner le coût d'une requête.
    """
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
            # Sans la position, l'appelant d'un lot de 500 ne sait pas quoi corriger.
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
