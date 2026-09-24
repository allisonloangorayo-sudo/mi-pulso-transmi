"""Collector: descarga el histórico + el stream incremental y los persiste
en Supabase de forma idempotente.

Confirmado en producción (2026-09-24, clock state=running): el histórico
estático `/v1/observations` se congela en `history_end` del dataset inicial.
Los datos de la ventana competitiva (después de ese corte) solo se liberan
por `/v1/stream/observations`, paginado con cursor — por eso este collector
sincroniza ambas fuentes. El cursor solo avanza después de confirmar el
upsert (regla de la guía operativa p.5).
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from dotenv import load_dotenv
from pulso_transmi import PulsoTransmiClient

from src import db
from src.http_utils import request_with_retry

load_dotenv()

PULSO_API_URL = os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io")
PULSO_API_KEY = os.getenv("PULSO_API_KEY")


def validate_observations(observations: pd.DataFrame) -> dict:
    """Continuidad, duplicados, tipos y cobertura por estación (Fase 1, paso 2)."""
    duplicates = int(observations.duplicated(subset=["station_id", "observed_at"]).sum())
    null_demand = int(observations["demand"].isna().sum())
    negative_demand = int((observations["demand"] < 0).sum())

    counts = observations.groupby("station_id")["observed_at"].count()
    expected = observations["observed_at"].nunique()
    stations_with_gaps = counts[counts < expected]

    report = {
        "rows": len(observations),
        "stations": observations["station_id"].nunique(),
        "expected_periods_per_station": expected,
        "duplicates": duplicates,
        "null_demand": null_demand,
        "negative_demand": negative_demand,
        "stations_with_gaps": stations_with_gaps.to_dict(),
    }
    return report


def sync_stream() -> int:
    """Consume /v1/stream/observations desde el último cursor confirmado."""
    if not PULSO_API_KEY:
        print("PULSO_API_KEY no configurado: se omite el stream incremental (solo histórico estático).")
        return 0

    headers = {"Authorization": f"Bearer {PULSO_API_KEY}"}
    cursor_row = db.get_cursor("stream_observations")
    cursor = cursor_row["last_cursor"] if cursor_row else None
    cursor_before = cursor

    frames = []
    while True:
        params = {"limit": 5000}
        if cursor:
            params["cursor"] = cursor
        response = request_with_retry(
            "GET", f"{PULSO_API_URL}/v1/stream/observations", params=params, headers=headers, timeout=45
        )
        response.raise_for_status()
        page = response.json()
        if page["data"]:
            frames.append(pd.DataFrame(page["data"]))
        cursor = page.get("next_cursor")
        if cursor is None:
            break

    if not frames:
        db.log_ingestion_run(stream="stream_observations", rows_fetched=0, status="no_new_data")
        return 0

    stream_df = pd.concat(frames, ignore_index=True)
    stream_df["observed_at"] = pd.to_datetime(stream_df["observed_at"], utc=True)
    stream_df["station_id"] = stream_df["station_id"].astype("string")
    n = db.upsert_observations(stream_df[["station_id", "observed_at", "demand"]])

    db.log_ingestion_run(
        stream="stream_observations", rows_fetched=n, status="ok", cursor_before=cursor_before, cursor_after=cursor
    )
    # El cursor solo avanza aquí, después de confirmar el upsert arriba.
    db.set_cursor("stream_observations", last_cursor=cursor, last_observed_at=stream_df["observed_at"].max())
    return n


def run() -> None:
    with PulsoTransmiClient() as client:
        stations = client.stations()
        observations = client.observations_dataframe()
        context = client.context_dataframe()

    report = validate_observations(observations)
    print("Reporte de validación (histórico estático):")
    for key, value in report.items():
        print(f"  {key}: {value}")

    if report["duplicates"] or report["null_demand"] or report["negative_demand"]:
        print("ADVERTENCIA: hay problemas de calidad; revisa antes de entrenar.", file=sys.stderr)

    try:
        n_stations = db.upsert_stations(stations)
        n_obs = db.upsert_observations(observations)
        n_ctx = db.upsert_context(context)
        n_stream = sync_stream()
    except Exception as exc:  # noqa: BLE001 - se registra y se relanza
        db.log_ingestion_run(stream="observations", rows_fetched=0, status="error", error_message=str(exc))
        raise

    db.log_ingestion_run(stream="stations", rows_fetched=n_stations, status="ok")
    db.log_ingestion_run(
        stream="observations",
        rows_fetched=n_obs,
        status="ok" if n_obs else "no_new_data",
    )
    db.log_ingestion_run(
        stream="context",
        rows_fetched=n_ctx,
        status="ok" if n_ctx else "no_new_data",
    )

    if n_obs:
        db.set_cursor("observations", last_cursor=None, last_observed_at=observations["observed_at"].max())
    if n_ctx:
        db.set_cursor("context", last_cursor=None, last_observed_at=context["observed_at"].max())

    print(
        f"Sincronizado: {n_stations} estaciones, {n_obs} observaciones estáticas, "
        f"{n_stream} observaciones del stream, {n_ctx} filas de contexto."
    )


if __name__ == "__main__":
    run()
