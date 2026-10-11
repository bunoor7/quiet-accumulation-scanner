"""Small shared HTTP helper: retries, timeouts, one place to see them.

Every exchange adapter calls ``get_json`` instead of ``requests.get``
directly, so the retry/backoff policy lives in exactly one place and
matches the project-wide retry philosophy (a transient failure is
retried a bounded number of times with a short backoff; a persistent
failure is raised so the caller can report "data unavailable" honestly
instead of silently returning stale or empty data).
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests


class ExchangeHTTPError(RuntimeError):
    """Raised after all retries are exhausted for a public API call."""


def get_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout_s: float = 10.0,
    retries: int = 3,
    retry_wait_s: float = 2.0,
    headers: Optional[Dict[str, str]] = None,
) -> Any:
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout_s, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - we deliberately want to retry anything
            last_exc = exc
            if attempt < retries:
                time.sleep(retry_wait_s)
    raise ExchangeHTTPError(
        f"GET {url} failed after {retries} attempts: {last_exc}"
    ) from last_exc
