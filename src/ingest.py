"""Collector: descarga el histórico/incremental y lo persiste en Supabase.

Fase actual (API 0.2.0, solo lectura): no existe todavía un cursor de
`/v1/observations` en el sentido incremental del contrato operativo — el
histórico es fijo (45 días). Por eso esta primera versión descarga todo el
corte disponible y lo sincroniza por upsert (station_id, observed_at), lo
que ya es idempotente: correrlo dos veces no duplica filas.

Cuando la API libere observaciones nuevas (ventana competitiva), el mismo
patrón de upsert seguirá siendo válido; solo cambia qué tanto se descarga
en cada corrida (usar `start=` con el último `observed_at` confirmado).
"""

from __future__ import annotations

import sys

import pandas as pd
from dotenv import load_dotenv
from pulso_transmi import PulsoTransmiClient

from src import db

load_dotenv()


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


def run() -> None:
    with PulsoTransmiClient() as client:
        stations = client.stations()
        observations = client.observations_dataframe()
        context = client.context_dataframe()

    report = validate_observations(observations)
    print("Reporte de validación:")
    for key, value in report.items():
        print(f"  {key}: {value}")

    if report["duplicates"] or report["null_demand"] or report["negative_demand"]:
        print("ADVERTENCIA: hay problemas de calidad; revisa antes de entrenar.", file=sys.stderr)

    try:
        n_stations = db.upsert_stations(stations)
        n_obs = db.upsert_observations(observations)
        n_ctx = db.upsert_context(context)
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

    print(f"Sincronizado: {n_stations} estaciones, {n_obs} observaciones, {n_ctx} filas de contexto.")


if __name__ == "__main__":
    run()
