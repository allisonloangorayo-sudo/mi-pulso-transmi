"""Monitoreo de desempeño y señal de drift (Fase 5 del taller).

Dos modos:
- offline_backtest_report(): usa el histórico estático de hoy para validar
  que el pipeline de métricas funciona, sin depender de submissions reales.
- accuracy_from_evaluations(): lee la tabla `evaluations` en Supabase, que
  solo tendrá filas una vez existan predicciones reales evaluadas contra la
  demanda observada (ventana competitiva).
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
from dotenv import load_dotenv
from pulso_transmi import PulsoTransmiClient

from src.train import temporal_split, wape_accuracy_by_station

load_dotenv()

DRIFT_DROP_THRESHOLD = 10.0  # puntos de accuracy: caída que amerita investigar


def offline_backtest_report() -> pd.Series:
    with PulsoTransmiClient() as client:
        observations = client.observations_dataframe()

    _, validation = temporal_split(observations)
    naive = validation.copy()
    naive["prediction"] = naive.groupby("station_id")["demand"].shift(1)
    naive = naive.dropna(subset=["prediction"])
    scores = wape_accuracy_by_station(naive, "prediction")
    print("Accuracy por estación (baseline naive, últimos 7 días):")
    print(scores.sort_values().round(2))
    print(f"Accuracy promedio: {scores.mean():.2f}")
    return scores


def accuracy_from_evaluations() -> dict:
    from src import db

    client = db.get_client()
    result = client.table("evaluations").select("*").execute()
    rows = result.data or []
    if not rows:
        print("Sin evaluaciones todavía (no hay submissions reales evaluadas).")
        return {}

    df = pd.DataFrame(rows)
    df["evaluated_at"] = pd.to_datetime(df["evaluated_at"], utc=True)

    overall = 100 * (1 - df["abs_error"].sum() / df["real"].sum())
    recent_cutoff = df["evaluated_at"].max() - timedelta(hours=24)
    recent = df[df["evaluated_at"] > recent_cutoff]
    rolling_24h = 100 * (1 - recent["abs_error"].sum() / recent["real"].sum()) if not recent.empty else None

    print(f"Accuracy acumulada: {overall:.2f}")
    if rolling_24h is not None:
        print(f"Accuracy rolling 24h: {rolling_24h:.2f}")
        if overall - rolling_24h > DRIFT_DROP_THRESHOLD:
            print(
                f"SEÑAL DE DRIFT: rolling 24h cayó {overall - rolling_24h:.1f} pts "
                "respecto a la acumulada. Investigar antes de reentrenar (guía operativa p.9)."
            )

    per_station = df.groupby("station_id").apply(
        lambda g: 100 * (1 - g["abs_error"].sum() / g["real"].sum())
    )
    print("\nAccuracy por estación:")
    print(per_station.sort_values().round(2))

    return {"overall": overall, "rolling_24h": rolling_24h, "per_station": per_station.to_dict()}


def run() -> None:
    print("=== Reporte offline (histórico estático) ===")
    offline_backtest_report()
    print("\n=== Reporte desde Supabase (evaluaciones reales) ===")
    from src.db import SupabaseNotConfigured

    try:
        accuracy_from_evaluations()
    except SupabaseNotConfigured:
        print("SUPABASE_URL/SUPABASE_KEY no configurados: se omite el reporte de evaluaciones.")


if __name__ == "__main__":
    run()
