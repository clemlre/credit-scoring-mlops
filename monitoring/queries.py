"""Requêtes PostgreSQL utilisées par le tableau de bord."""

from __future__ import annotations

import os

import pandas as pd
import psycopg
from psycopg import sql

DSN = os.environ.get(
    "DATABASE_URL", "postgresql://scoring:scoring_dev@127.0.0.1:5432/monitoring"
)


def _read(query, parameters=()) -> pd.DataFrame:
    with psycopg.connect(DSN, connect_timeout=5) as connection:
        cursor = connection.execute(query, parameters)
        columns = [column.name for column in cursor.description]
        return pd.DataFrame(cursor.fetchall(), columns=columns)


def _one(query, parameters) -> dict:
    row = _read(query, parameters).iloc[0]
    return {key: (None if pd.isna(value) else value) for key, value in row.items()}


def _where(period: tuple, version: str | None = None) -> tuple:
    start, end = period
    conditions, parameters = [sql.SQL("TRUE")], []
    if start is not None:
        conditions.append(sql.SQL("occurred_at >= %s"))
        parameters.append(start)
    if end is not None:
        conditions.append(sql.SQL("occurred_at < %s"))
        parameters.append(end)
    if version is not None:
        conditions.append(sql.SQL("model_version = %s"))
        parameters.append(version)
    return sql.SQL(" AND ").join(conditions), parameters


def versions() -> list[str]:
    query = "SELECT DISTINCT model_version FROM predictions ORDER BY 1"
    return _read(query)["model_version"].tolist()


def request_summary(period: tuple) -> dict:
    where, parameters = _where(period)
    return _one(
        sql.SQL("""
            SELECT count(*) AS appels,
                   count(*) FILTER (WHERE status_code BETWEEN 400 AND 499) AS erreurs_4xx,
                   count(*) FILTER (WHERE status_code >= 500) AS erreurs_5xx,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms)
                       FILTER (WHERE path = '/predict') AS latence_p50_ms,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms)
                       FILTER (WHERE path = '/predict') AS latence_p95_ms
            FROM requests WHERE {where}
        """).format(where=where),
        parameters,
    )


def prediction_summary(period: tuple, version: str | None) -> dict:
    where, parameters = _where(period, version)
    return _one(
        sql.SQL("""
            SELECT count(*) AS nombre,
                   avg((decision = 'rejected')::int)::float AS taux_refus,
                   avg(probability) AS proba_moyenne,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) AS inference_p50_ms,
                   avg(application_ratio) AS couverture_dossier,
                   avg(history_ratio) AS couverture_historique
            FROM predictions WHERE {where}
        """).format(where=where),
        parameters,
    )


def request_series(period: tuple, step: str) -> pd.DataFrame:
    where, parameters = _where(period)
    return _read(
        sql.SQL("""
            SELECT date_trunc({step}, occurred_at) AS instant,
                   CASE WHEN status_code >= 500 THEN '5xx'
                        WHEN status_code >= 400 THEN '4xx'
                        ELSE '2xx' END AS classe,
                   count(*) AS appels
            FROM requests WHERE {where}
            GROUP BY 1, 2 ORDER BY 1
        """).format(step=sql.Literal(step), where=where),
        parameters,
    )


def latency_series(period: tuple, step: str) -> pd.DataFrame:
    where, parameters = _where(period)
    return _read(
        sql.SQL("""
            SELECT date_trunc({step}, occurred_at) AS instant,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms) AS p50,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95
            FROM requests
            WHERE {where} AND path = '/predict' AND status_code < 400
            GROUP BY 1 ORDER BY 1
        """).format(step=sql.Literal(step), where=where),
        parameters,
    )


def prediction_series(period: tuple, version: str | None, step: str) -> pd.DataFrame:
    where, parameters = _where(period, version)
    return _read(
        sql.SQL("""
            SELECT date_trunc({step}, occurred_at) AS instant,
                   count(*) AS predictions,
                   avg((decision = 'rejected')::int)::float AS taux_refus,
                   avg(application_ratio) AS couverture_dossier,
                   avg(history_ratio) AS couverture_historique
            FROM predictions WHERE {where}
            GROUP BY 1 ORDER BY 1
        """).format(step=sql.Literal(step), where=where),
        parameters,
    )


def errors(period: tuple) -> pd.DataFrame:
    where, parameters = _where(period)
    return _read(
        sql.SQL("""
            SELECT path AS route, status_code AS statut, count(*) AS appels
            FROM requests WHERE {where} AND status_code >= 400
            GROUP BY 1, 2 ORDER BY 3 DESC
        """).format(where=where),
        parameters,
    )


def production_values(period: tuple, version: str | None, names: tuple[str, ...]) -> pd.DataFrame:
    where, parameters = _where(period, version)
    columns = sql.SQL(", ").join(
        sql.SQL("(features->>{})::float AS {}").format(
            sql.Literal(name), sql.Identifier(name)
        )
        for name in names
    )
    query = sql.SQL("SELECT probability, {columns} FROM predictions WHERE {where}").format(
        columns=columns, where=where
    )
    return _read(query, parameters).astype("float64")
