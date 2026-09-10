"""Middleware ASGI : identifiant de requête, chronométrage et journalisation.

Middleware ASGI pur plutôt que `@app.middleware("http")` : ce dernier
(BaseHTTPMiddleware) ouvre un groupe de tâches et des flux mémoire à chaque
requête, un surcoût visible au profilage pour un service de quelques ms.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime

from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from api.storage import RequestRecord

logger = logging.getLogger("api")

# /health est exclue : la sonde du conteneur l'appelle toutes les 30 s.
TRACKED_PATHS = frozenset({"/predict", "/predict/batch", "/model/info", "/features"})


class RequestTrackingMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        start = time.perf_counter()
        # Reste à 500 si l'application lève avant d'avoir commencé sa réponse.
        status_code = 500

        async def send_with_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = [*message.get("headers", []), (b"x-request-id", request_id.encode())]
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            if scope["path"] in TRACKED_PATHS:
                record = RequestRecord(
                    request_id=request_id,
                    occurred_at=datetime.now(UTC),
                    method=scope["method"],
                    path=scope["path"],
                    status_code=status_code,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
                # Sauf erreur interne (le 500 est émis plus haut, après ce bloc), la
                # réponse est déjà partie : l'écriture n'entre pas dans la latence.
                await run_in_threadpool(
                    _write_log,
                    scope["app"].state.prediction_log,
                    record,
                    state.get("predictions", ()),
                )


def _write_log(log, record: RequestRecord, predictions) -> None:
    try:
        log.record_request(record, predictions)
    except Exception:
        logger.exception("Journalisation de la requête impossible")
