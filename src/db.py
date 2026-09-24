"""Acceso idempotente a la memoria operacional en Supabase.

Reglas del taller que este módulo respeta:
- upsert por llave primaria (nunca insert simple) para que repetir una
  ejecución no duplique filas;
- el cursor solo avanza después de que el upsert fue confirmado;
- cada corrida deja evidencia en `ingestion_runs`, incluso sin novedades.
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from functools import lru_cache
from typing import Any, Iterable

import pandas as pd
from supabase import Client, create_client


class SupabaseNotConfigured(RuntimeError):
    """SUPABASE_URL / SUPABASE_KEY ausentes o vacíos en el entorno."""


@lru_cache(maxsize=1)
def get_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise SupabaseNotConfigured("Configura SUPABASE_URL y SUPABASE_KEY en .env (ver README).")
    return create_client(url, key)


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {key: _json_safe(value) for key, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def upsert_in_chunks(
    table: str,
    records: Iterable[dict[str, Any]],
    on_conflict: str,
    chunk_size: int = 2000,
) -> int:
    records = list(records)
    client = get_client()
    total = 0
    for start in range(0, len(records), chunk_size):
        chunk = records[start : start + chunk_size]
        if not chunk:
            continue
        client.table(table).upsert(chunk, on_conflict=on_conflict).execute()
        total += len(chunk)
    return total


def upsert_stations(stations: pd.DataFrame) -> int:
    if stations.empty:
        return 0
    records = []
    for row in stations.to_dict(orient="records"):
        station_id = row.pop("station_id")
        records.append({"station_id": station_id, "attributes": {k: _json_safe(v) for k, v in row.items()}})
    return upsert_in_chunks("stations", records, on_conflict="station_id")


def upsert_observations(observations: pd.DataFrame) -> int:
    if observations.empty:
        return 0
    records = _records(observations[["station_id", "observed_at", "demand"]])
    return upsert_in_chunks("observations", records, on_conflict="station_id,observed_at")


def upsert_context(context: pd.DataFrame) -> int:
    if context.empty:
        return 0
    records = []
    for row in context.to_dict(orient="records"):
        observed_at = row.pop("observed_at")
        records.append(
            {
                "observed_at": _json_safe(observed_at),
                "payload": {k: _json_safe(v) for k, v in row.items()},
            }
        )
    return upsert_in_chunks("context", records, on_conflict="observed_at")


def get_cursor(stream: str) -> dict[str, Any] | None:
    client = get_client()
    result = client.table("collector_cursor").select("*").eq("stream", stream).execute()
    rows = result.data or []
    return rows[0] if rows else None


def set_cursor(stream: str, last_cursor: str | None, last_observed_at: Any) -> None:
    client = get_client()
    client.table("collector_cursor").upsert(
        {
            "stream": stream,
            "last_cursor": last_cursor,
            "last_observed_at": _json_safe(last_observed_at),
            "updated_at": datetime.utcnow().isoformat(),
        },
        on_conflict="stream",
    ).execute()


def log_ingestion_run(
    stream: str,
    rows_fetched: int,
    status: str,
    cursor_before: str | None = None,
    cursor_after: str | None = None,
    error_message: str | None = None,
) -> None:
    client = get_client()
    client.table("ingestion_runs").insert(
        {
            "stream": stream,
            "finished_at": datetime.utcnow().isoformat(),
            "rows_fetched": rows_fetched,
            "cursor_before": cursor_before,
            "cursor_after": cursor_after,
            "status": status,
            "error_message": error_message,
        }
    ).execute()


def register_model_version(
    version: str,
    data_cutoff: Any,
    artifact_path: str,
    commit_sha: str | None,
    features: list[str],
    validation_metric: float,
    status: str = "candidate",
) -> None:
    client = get_client()
    client.table("model_versions").upsert(
        {
            "version": version,
            "data_cutoff": _json_safe(data_cutoff),
            "commit_sha": commit_sha,
            "features": features,
            "validation_metric": validation_metric,
            "artifact_path": artifact_path,
            "status": status,
        },
        on_conflict="version",
    ).execute()
