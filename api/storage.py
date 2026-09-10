"""Journalisation de production : prédictions et requêtes HTTP.

Deux canaux :
- stdout, une ligne JSON par événement, sans aucune valeur de feature ;
- PostgreSQL (si DATABASE_URL est défini) : table `predictions`, avec les features
  en JSONB pour l'analyse de dérive, et table `requests` (statut et durée de chaque
  appel, erreurs comprises).

Une panne de la base ne doit jamais faire échouer une prédiction : les écritures
ont lieu après la réponse et toutes les erreurs sont absorbées.
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

TABLE = "predictions"
REQUESTS_TABLE = "requests"

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
CREATE INDEX IF NOT EXISTS {TABLE}_occurred_at_idx
    ON {TABLE} (occurred_at DESC);
CREATE INDEX IF NOT EXISTS {TABLE}_model_version_idx
    ON {TABLE} (model_version, occurred_at DESC);

CREATE TABLE IF NOT EXISTS {REQUESTS_TABLE} (
    id          BIGSERIAL   PRIMARY KEY,
    request_id  UUID        NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL,
    method      TEXT        NOT NULL,
    path        TEXT        NOT NULL,
    status_code INTEGER     NOT NULL,
    duration_ms DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS {REQUESTS_TABLE}_occurred_at_idx
    ON {REQUESTS_TABLE} (occurred_at DESC);
"""

INSERT_SQL = f"""
INSERT INTO {TABLE} (
    request_id, occurred_at, endpoint, model_version, threshold,
    probability, decision, features_provided, features_missing,
    application_ratio, history_ratio, latency_ms, features
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

INSERT_REQUEST_SQL = f"""
INSERT INTO {REQUESTS_TABLE} (
    request_id, occurred_at, method, path, status_code, duration_ms
) VALUES (%s, %s, %s, %s, %s, %s)
"""


@dataclass(frozen=True)
class PredictionRecord:
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
        """Version sans features, pour stdout."""
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


@dataclass(frozen=True)
class RequestRecord:
    request_id: str
    occurred_at: datetime
    method: str
    path: str
    status_code: int
    duration_ms: float

    def summary(self) -> dict[str, Any]:
        return {
            "event": "request",
            "request_id": self.request_id,
            "occurred_at": self.occurred_at.isoformat(),
            "method": self.method,
            "path": self.path,
            "status_code": self.status_code,
            "duration_ms": round(self.duration_ms, 3),
        }

    def row(self) -> tuple:
        return (
            self.request_id,
            self.occurred_at,
            self.method,
            self.path,
            self.status_code,
            self.duration_ms,
        )


class _StdoutHandler(logging.StreamHandler):
    """Résout sys.stdout à chaque émission (compatible avec la capture de pytest)."""

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, _value):
        pass


def _stdout_channel() -> logging.Logger:
    channel = logging.getLogger("api.predictions")
    if not channel.handlers:
        handler = _StdoutHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        channel.addHandler(handler)
        channel.setLevel(logging.INFO)
        channel.propagate = False
    return channel


class PredictionLog:
    def __init__(self, dsn: str | None = None):
        self._dsn = dsn
        self._pool = None
        self._schema_ready = False
        self._channel = _stdout_channel()
        self.last_error: str | None = None

    def open(self) -> None:
        """Ouvre le pool de connexions. Ne lève jamais."""
        if not self._dsn:
            logger.info("Journal des prédictions : stdout uniquement (DATABASE_URL non défini).")
            return

        try:
            from psycopg_pool import ConnectionPool

            # open(wait=False) : l'API démarre même si PostgreSQL n'est pas encore prêt.
            self._pool = ConnectionPool(
                self._dsn,
                min_size=config.DB_POOL_MIN_SIZE,
                max_size=config.DB_POOL_MAX_SIZE,
                kwargs={"connect_timeout": config.DB_CONNECT_TIMEOUT},
                open=False,
            )
            self._pool.open(wait=False)
        except Exception as exc:  # noqa: BLE001
            self._pool = None
            self.last_error = _describe(exc)
            logger.error("Journal des prédictions : pool inutilisable — %s", self.last_error)
            return

        try:
            self._prepare_schema()
        except Exception as exc:  # noqa: BLE001
            self.last_error = _describe(exc)
            logger.warning(
                "Journal des prédictions : base injoignable, nouvel essai à la première "
                "écriture — %s",
                self.last_error,
            )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None
        self._schema_ready = False

    @property
    def database_enabled(self) -> bool:
        return self._pool is not None

    def status(self) -> dict[str, Any]:
        """Dernier état connu, sans requête SQL (appelé par /health)."""
        if not self.database_enabled:
            etat = "disabled"
        elif self._schema_ready and self.last_error is None:
            etat = "ready"
        else:
            etat = "unavailable"
        return {"stdout": True, "database": etat, "last_error": self.last_error}

    def record(self, records: Sequence[PredictionRecord]) -> None:
        """Journalise un lot de prédictions. Ne lève jamais."""
        self._record([(INSERT_SQL, records)])

    def record_request(
        self, request: RequestRecord, predictions: Sequence[PredictionRecord] = ()
    ) -> None:
        """Journalise une requête HTTP et les prédictions qu'elle a produites, en une
        seule transaction. Ne lève jamais."""
        self._record([(INSERT_SQL, predictions), (INSERT_REQUEST_SQL, [request])])

    def _record(self, batches: list[tuple[str, Sequence]]) -> None:
        batches = [(sql, records) for sql, records in batches if records]
        if not batches:
            return

        for _, records in batches:
            for record in records:
                self._channel.info(json.dumps(record.summary(), ensure_ascii=False))

        if self._pool is None:
            return

        try:
            self._write(batches)
            self.last_error = None
        except Exception as exc:  # noqa: BLE001
            self.last_error = _describe(exc)
            logger.warning(
                "Journal de production : %d ligne(s) non stockée(s) — %s",
                sum(len(records) for _, records in batches),
                self.last_error,
            )

    def _write(self, batches: list[tuple[str, Sequence]]) -> None:
        with self._pool.connection(timeout=config.DB_WRITE_TIMEOUT) as conn:
            if not self._schema_ready:
                conn.execute(SCHEMA_SQL)
                self._schema_ready = True
            with conn.cursor() as cur:
                for sql, records in batches:
                    cur.executemany(sql, [r.row() for r in records])

    def _prepare_schema(self) -> None:
        with self._pool.connection(timeout=config.DB_WRITE_TIMEOUT) as conn:
            conn.execute(SCHEMA_SQL)
        self._schema_ready = True
        self.last_error = None


def _describe(exc: BaseException) -> str:
    """Message court et borné, pour ne jamais recopier la chaîne de connexion."""
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
        # Payload tel que reçu, pas la ligne complétée à 779 colonnes.
        features=dict(features),
    )
