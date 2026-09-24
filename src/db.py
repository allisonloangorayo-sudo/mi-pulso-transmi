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
from pathlib import Path
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


MODEL_BUCKET = "models"


def ensure_model_bucket() -> None:
    client = get_client()
    try:
        client.storage.create_bucket(MODEL_BUCKET, options={"public": False})
    except Exception as exc:  # noqa: BLE001 - "ya existe" no es un error real
        if "already exists" not in str(exc).lower() and "duplicate" not in str(exc).lower():
            raise


def upload_model_artifact(local_path: str | Path, remote_name: str) -> str:
    """Sube el .joblib a Supabase Storage. Necesario porque cada corrida de
    GitHub Actions empieza en un runner limpio: sin esto, train.yml guarda el
    modelo pero infer.yml (en otra ejecución) nunca podría encontrarlo.
    """
    ensure_model_bucket()
    client = get_client()
    client.storage.from_(MODEL_BUCKET).upload(
        remote_name,
        str(local_path),
        file_options={"upsert": "true", "content-type": "application/octet-stream"},
    )
    return f"{MODEL_BUCKET}/{remote_name}"


def download_model_artifact(remote_name: str, local_path: str | Path) -> Path:
    client = get_client()
    data = client.storage.from_(MODEL_BUCKET).download(remote_name)
    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def get_state(key: str) -> Any | None:
    client = get_client()
    result = client.table("pipeline_state").select("value").eq("key", key).execute()
    rows = result.data or []
    return rows[0]["value"] if rows else None


def set_state(key: str, value: Any) -> None:
    client = get_client()
    client.table("pipeline_state").upsert(
        {"key": key, "value": _json_safe(value), "updated_at": datetime.utcnow().isoformat()},
        on_conflict="key",
    ).execute()


def get_champion() -> dict[str, Any] | None:
    client = get_client()
    result = (
        client.table("model_versions")
        .select("*")
        .eq("status", "champion")
        .order("trained_at", desc=True)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    return rows[0] if rows else None


def promote_if_better(version: str, validation_metric: float) -> bool:
    """Promueve `version` a champion solo si supera al champion vigente.

    Regla de la guía: "una nueva versión reemplaza al champion únicamente si
    supera los criterios de validación... la novedad por sí sola no es una
    mejora". Si no hay champion todavía, el primer candidato válido lo es.
    """
    client = get_client()
    champion = get_champion()
    if champion is not None and validation_metric <= champion["validation_metric"]:
        return False
    if champion is not None:
        client.table("model_versions").update({"status": "historical"}).eq("version", champion["version"]).execute()
    client.table("model_versions").update({"status": "champion"}).eq("version", version).execute()
    return True


def insert_evaluations(records: list[dict[str, Any]]) -> None:
    if not records:
        return
    client = get_client()
    client.table("evaluations").insert([{k: _json_safe(v) for k, v in r.items()} for r in records]).execute()


def insert_drift_metric(record: dict[str, Any]) -> None:
    client = get_client()
    client.table("drift_metrics").insert({k: _json_safe(v) for k, v in record.items()}).execute()


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
