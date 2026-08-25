"""Journalisation des prédictions rendues en production.

Sans cette brique, une fois l'API déployée, plus personne ne sait ce que le modèle
a réellement répondu : ni pour enquêter sur une réclamation client, ni pour mesurer
la dérive des données. C'est la matière première du monitoring.

Deux canaux, délibérément distincts
-----------------------------------

1. **Sortie standard, au format JSON, une ligne par prédiction.** Toujours active,
   sans dépendance ni configuration. Dans un conteneur, `stdout` *est* le transport
   de journaux : Docker, Kubernetes et Hugging Face Spaces le collectent d'office.
   Ce canal ne contient **aucune valeur de feature** — seulement le score, la
   décision et des métadonnées. Les journaux applicatifs finissent souvent chez un
   tiers (Datadog, CloudWatch…) : les revenus et l'âge d'un demandeur de crédit
   n'ont rien à y faire.

2. **PostgreSQL.** Reçoit en plus les features de la requête, en `JSONB`. C'est la
   base interrogeable qui alimente l'analyse de data drift et le tableau de bord de
   monitoring. Elle reste sous le contrôle de « Prêt à Dépenser ».

Trois invariants
----------------

- **Le monitoring ne casse jamais une prédiction.** L'écriture en base a lieu
  *après* l'envoi de la réponse (`BackgroundTasks`) et toute exception y est
  absorbée. Une base indisponible dégrade l'observabilité, elle n'interrompt pas
  le service de scoring.
- **Le pool de connexions est ouvert une seule fois** au démarrage, comme le
  modèle. Ouvrir une connexion PostgreSQL par requête coûterait plus cher que
  l'inférence elle-même.
- **Une base absente est un mode de fonctionnement normal**, pas une panne :
  sans `DATABASE_URL`, seul le canal `stdout` fonctionne. C'est le cas en test,
  en CI et pour un simple `docker run` de démonstration.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from api import config

logger = logging.getLogger("api")

# Nom de table volontairement constant, et non paramétrable par l'environnement :
# il est interpolé dans du SQL, et une valeur venue de l'extérieur y ouvrirait une
# injection pour ne rendre service à personne.
TABLE = "predictions"

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id                BIGSERIAL   PRIMARY KEY,
    request_id        UUID        NOT NULL,
    occurred_at       TIMESTAMPTZ NOT NULL,
    endpoint          TEXT        NOT NULL,
    model_version     TEXT        NOT NULL,
    threshold         DOUBLE PRECISION NOT NULL,
    probability       DOUBLE PRECISION NOT NULL,
    decision          TEXT        NOT NULL,
    features_provided INTEGER     NOT NULL,
    features_missing  INTEGER     NOT NULL,
    application_ratio DOUBLE PRECISION NOT NULL,
    history_ratio     DOUBLE PRECISION NOT NULL,
    latency_ms        DOUBLE PRECISION NOT NULL,
    features          JSONB       NOT NULL
);

-- Le monitoring interroge presque toujours « les N derniers jours », et souvent
-- pour une version de modèle donnée (comparer avant/après un déploiement).
CREATE INDEX IF NOT EXISTS {TABLE}_occurred_at_idx
    ON {TABLE} (occurred_at DESC);
CREATE INDEX IF NOT EXISTS {TABLE}_model_version_idx
    ON {TABLE} (model_version, occurred_at DESC);
"""

INSERT_SQL = f"""
INSERT INTO {TABLE} (
    request_id, occurred_at, endpoint, model_version, threshold,
    probability, decision, features_provided, features_missing,
    application_ratio, history_ratio, latency_ms, features
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


@dataclass(frozen=True)
class PredictionRecord:
    """Une prédiction rendue, telle qu'on veut pouvoir la relire des mois plus tard.

    On stocke la **couverture** et la **latence** en plus du score : sans elles, un
    écart constaté a posteriori est indiagnosticable (le modèle a-t-il dérivé, ou
    bien les appelants ont-ils commencé à envoyer des dossiers plus incomplets ?).
    """

    request_id: str
    occurred_at: datetime
    endpoint: str
    model_version: str
    threshold: float
    probability: float
    decision: str
    features_provided: int
    features_missing: int
    application_ratio: float
    history_ratio: float
    latency_ms: float
    features: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """Vue sans aucune valeur de feature, destinée au canal `stdout`."""
        return {
            "event": "prediction",
            "request_id": self.request_id,
            "occurred_at": self.occurred_at.isoformat(),
            "endpoint": self.endpoint,
            "model_version": self.model_version,
            "threshold": self.threshold,
            "probability": round(self.probability, 6),
            "decision": self.decision,
            "features_provided": self.features_provided,
            "features_missing": self.features_missing,
            "application_ratio": round(self.application_ratio, 4),
            "history_ratio": round(self.history_ratio, 4),
            "latency_ms": round(self.latency_ms, 3),
        }

    def row(self) -> tuple:
        """Les paramètres de l'insertion, dans l'ordre de `INSERT_SQL`."""
        from psycopg.types.json import Jsonb

        return (
            self.request_id,
            self.occurred_at,
            self.endpoint,
            self.model_version,
            self.threshold,
            self.probability,
            self.decision,
            self.features_provided,
            self.features_missing,
            self.application_ratio,
            self.history_ratio,
            self.latency_ms,
            Jsonb(self.features),
        )


class _StdoutHandler(logging.StreamHandler):
    """Handler qui résout `sys.stdout` **à l'émission**, pas à la construction.

    `logging.StreamHandler(sys.stdout)` capture le flux une fois pour toutes. Le
    handler continue alors d'écrire dans l'objet d'origine même si `sys.stdout` a
    été remplacé depuis — ce qui arrive dès qu'on redirige la sortie : capture de
    tests, `contextlib.redirect_stdout`, ou un superviseur qui réouvre le flux.
    Résoudre à l'émission rend le canal fidèle à ce qui se passe réellement sur la
    sortie standard du processus.
    """

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, _value):
        # `StreamHandler.__init__` affecte `self.stream` : on ignore l'affectation,
        # la propriété fait autorité.
        pass


def _stdout_channel() -> logging.Logger:
    """Le flux de prédictions, séparé du journal applicatif.

    `propagate = False` l'isole du logger d'uvicorn : les lignes sortent en JSON
    brut, sans préfixe de niveau ni horodatage ajouté, pour qu'un collecteur puisse
    les parser sans expression régulière.
    """
    channel = logging.getLogger("api.predictions")
    if not channel.handlers:
        handler = _StdoutHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        channel.addHandler(handler)
        channel.setLevel(logging.INFO)
        channel.propagate = False
    return channel


class PredictionLog:
    """Journal des prédictions : `stdout` toujours, PostgreSQL si configuré."""

    def __init__(self, dsn: str | None = None):
        self._dsn = dsn
        self._pool = None
        self._schema_ready = False
        self._channel = _stdout_channel()
        self.last_error: str | None = None

    # --- Cycle de vie ---

    def open(self) -> None:
        """Ouvre le pool de connexions. Ne lève jamais : au pire, on reste en `stdout`."""
        if not self._dsn:
            logger.info(
                "Journal des prédictions : sortie standard uniquement "
                "(DATABASE_URL non défini)."
            )
            return

        try:
            from psycopg_pool import ConnectionPool

            # `open=False` puis `open(wait=False)` : le démarrage de l'API ne doit pas
            # dépendre de la disponibilité de la base. Si PostgreSQL démarre plus
            # lentement que l'API — le cas normal avec docker compose — le pool se
            # connectera tout seul, sans que le service ait échoué entre-temps.
            self._pool = ConnectionPool(
                self._dsn,
                min_size=config.DB_POOL_MIN_SIZE,
                max_size=config.DB_POOL_MAX_SIZE,
                kwargs={"connect_timeout": config.DB_CONNECT_TIMEOUT},
                open=False,
            )
            self._pool.open(wait=False)
        except Exception as exc:  # noqa: BLE001 - aucune panne d'infra ne doit empêcher l'API de démarrer
            self._pool = None
            self.last_error = _describe(exc)
            logger.error("Journal des prédictions : pool inutilisable — %s", self.last_error)
            return

        # Tentative immédiate de création du schéma, pour que /health dise la vérité
        # dès le démarrage. Un échec ici n'est pas fatal : il sera retenté à la
        # première écriture.
        try:
            self._prepare_schema()
        except Exception as exc:  # noqa: BLE001 - base pas encore prête : ce n'est pas une erreur fatale
            self.last_error = _describe(exc)
            logger.warning(
                "Journal des prédictions : base pas encore joignable, "
                "nouvelle tentative à la première prédiction — %s",
                self.last_error,
            )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None
        self._schema_ready = False

    # --- État, pour /health ---

    @property
    def database_enabled(self) -> bool:
        return self._pool is not None

    def status(self) -> dict[str, Any]:
        """État connu du journal, **sans aller interroger la base**.

        Une sonde de disponibilité est appelée toutes les 30 secondes : y glisser un
        aller-retour SQL la rendrait lente et sensible à un pic de charge de la base.
        On rapporte donc le dernier état constaté lors d'une écriture réelle.
        """
        if not self.database_enabled:
            etat = "disabled"
        elif self._schema_ready and self.last_error is None:
            etat = "ready"
        else:
            etat = "unavailable"
        return {"stdout": True, "database": etat, "last_error": self.last_error}

    # --- Écriture ---

    def record(self, records: Sequence[PredictionRecord]) -> None:
        """Journalise un lot de prédictions. **Ne lève jamais.**

        Appelée depuis une tâche d'arrière-plan, donc après que la réponse HTTP est
        partie. Une exception qui remonterait d'ici serait un incident de monitoring
        transformé en incident de production : c'est précisément ce qu'on refuse.
        """
        if not records:
            return

        for record in records:
            self._channel.info(json.dumps(record.summary(), ensure_ascii=False))

        if self._pool is None:
            return

        try:
            self._write(records)
            self.last_error = None
        except Exception as exc:  # noqa: BLE001 - voir le commentaire ci-dessous
            # `except Exception` large et assumé : aucune défaillance de la couche de
            # stockage — réseau, schéma, disque plein, mot de passe changé — ne doit
            # se propager. La perte est signalée dans les journaux applicatifs, et la
            # ligne JSON de `stdout` reste, elle, écrite.
            self.last_error = _describe(exc)
            logger.warning(
                "Journal des prédictions : %d prédiction(s) non stockée(s) en base — %s",
                len(records),
                self.last_error,
            )

    def _write(self, records: Sequence[PredictionRecord]) -> None:
        with self._pool.connection(timeout=config.DB_WRITE_TIMEOUT) as conn:
            if not self._schema_ready:
                conn.execute(SCHEMA_SQL)
                self._schema_ready = True
            with conn.cursor() as cur:
                # `executemany` pour un lot : un aller-retour par ligne annulerait
                # l'intérêt d'avoir vectorisé l'inférence sur tout le lot.
                cur.executemany(INSERT_SQL, [r.row() for r in records])

    def _prepare_schema(self) -> None:
        with self._pool.connection(timeout=config.DB_WRITE_TIMEOUT) as conn:
            conn.execute(SCHEMA_SQL)
        self._schema_ready = True
        self.last_error = None


def _describe(exc: BaseException) -> str:
    """Message d'erreur court, sans jamais recopier la chaîne de connexion.

    `DATABASE_URL` contient un mot de passe : il ne doit apparaître ni dans les
    journaux, ni dans une réponse HTTP.
    """
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
    return f"{exc.__class__.__name__}: {message}"[:300]


def build_record(
    *,
    request_id: str,
    endpoint: str,
    features: dict[str, Any],
    probability: float,
    decision: str,
    coverage,
    model_version: str,
    threshold: float,
    latency_ms: float,
) -> PredictionRecord:
    """Assemble un enregistrement à partir d'une prédiction rendue."""
    return PredictionRecord(
        request_id=request_id,
        occurred_at=datetime.now(UTC),
        endpoint=endpoint,
        model_version=model_version,
        threshold=threshold,
        probability=probability,
        decision=decision,
        features_provided=coverage.provided,
        features_missing=coverage.missing,
        application_ratio=coverage.application_ratio,
        history_ratio=coverage.history_ratio,
        latency_ms=latency_ms,
        # On stocke le payload **tel que reçu**, pas la ligne complétée à 779
        # colonnes : c'est ce que l'appelant a réellement envoyé, et c'est la seule
        # version qui reste interprétable si le contrat de features change.
        features=dict(features),
    )
