"""Reintentos con backoff para fallas transitorias (5xx, 429, timeouts).

Regla de la guía operativa (p.8): "5xx o timeout → falla transitoria →
reintentar con backoff y la misma llave". No reintenta 4xx (esos son
errores reales del payload/autenticación, no transitorios).
"""

from __future__ import annotations

import time
from typing import Any

import httpx

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def request_with_retry(
    method: str,
    url: str,
    *,
    max_attempts: int = 4,
    backoff_seconds: float = 2.0,
    **kwargs: Any,
) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = httpx.request(method, url, **kwargs)
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as exc:
            last_exc = exc
        else:
            if response.status_code not in RETRYABLE_STATUS:
                return response
            last_exc = httpx.HTTPStatusError(
                f"{response.status_code} en {url}", request=response.request, response=response
            )
        if attempt < max_attempts:
            time.sleep(backoff_seconds * attempt)
    assert last_exc is not None
    raise last_exc
