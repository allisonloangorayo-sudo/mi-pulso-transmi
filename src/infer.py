"""Inferencia horaria: 12 estaciones × 4 horizontes = 48 predicciones.

Dos modos, elegidos automáticamente:

1. Ciclo real: si `/v1/forecast-cycles/current` responde (competencia activa),
   delega en `src.predict`, que sigue el contrato oficial al pie de la letra.
2. Simulación: mientras ese endpoint no exista (fase actual), avanza un
   "reloj simulado" una hora por corrida sobre el histórico YA conocido.
   Como el dato real de esos instantes ya está en nuestra base (no es
   futuro de verdad, es futuro simulado), cada corrida puede evaluarse de
   inmediato contra `evaluations`. Esto es lo que le da datos reales a
   `src/drift.py` sin esperar a la ventana competitiva.

Al llegar al final de los 7 días reservados como validación, el reloj
simulado reinicia (loop) para seguir generando evidencia indefinidamente.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pulso_transmi import PulsoTransmiClient

from src import db
from src.features import FEATURE_COLUMNS, build_features
from src.predict import NoOpenCycle, double_check_predictions, get_current_cycle, load_champion

load_dotenv()

SIM_STEP = timedelta(hours=1)
HORIZONS_MIN = (15, 30, 45, 60)
VALIDATION_DAYS = 7  # debe coincidir con src.train.VALIDATION_DAYS


def _load_observations() -> pd.DataFrame:
    with PulsoTransmiClient() as client:
        return client.observations_dataframe()


def _initial_sim_cutoff(observations: pd.DataFrame) -> pd.Timestamp:
    return observations["observed_at"].max() - timedelta(days=VALIDATION_DAYS)


def _next_cutoff(observations: pd.DataFrame) -> pd.Timestamp:
    max_ts = observations["observed_at"].max()
    stored = db.get_state("sim_cutoff")
    cutoff = pd.Timestamp(stored) if stored else _initial_sim_cutoff(observations)
    next_cutoff = cutoff + SIM_STEP
    if next_cutoff + timedelta(minutes=max(HORIZONS_MIN)) > max_ts:
        next_cutoff = _initial_sim_cutoff(observations) + SIM_STEP
    return next_cutoff


def _build_batch(observations: pd.DataFrame, next_cutoff: pd.Timestamp) -> pd.DataFrame:
    known = observations[observations["observed_at"] <= next_cutoff]
    features_df = build_features(known)
    latest = features_df.sort_values("observed_at").groupby("station_id").tail(1).set_index("station_id")

    static_cols = [c for c in FEATURE_COLUMNS if c not in ("hour", "minute", "dayofweek", "is_weekend")]
    rows = []
    for station_id in observations["station_id"].unique():
        if station_id not in latest.index:
            continue
        base = latest.loc[station_id]
        for minutes in HORIZONS_MIN:
            target_at = next_cutoff + timedelta(minutes=minutes)
            row = {col: base[col] for col in static_cols}
            row.update(
                station_id=station_id,
                target_at=target_at,
                hour=target_at.hour,
                minute=target_at.minute,
                dayofweek=target_at.dayofweek,
                is_weekend=int(target_at.dayofweek in (5, 6)),
            )
            rows.append(row)
    return pd.DataFrame(rows)


def run_simulated_cycle() -> None:
    observations = _load_observations()
    next_cutoff = _next_cutoff(observations)
    batch = _build_batch(observations, next_cutoff)
    if batch.empty:
        print("Simulación sin datos suficientes todavía (histórico muy corto).")
        return

    model, champion = load_champion()
    batch["value"] = np.clip(double_check_predictions(model, batch), 0, None)

    cycle_id = f"sim-{next_cutoff.isoformat()}"
    pred_records = [
        {
            "cycle_id": cycle_id,
            "station_id": row["station_id"],
            "target_at": row["target_at"].isoformat(),
            "model_version": champion["version"],
            "value": float(row["value"]),
        }
        for row in batch.to_dict(orient="records")
    ]
    db.upsert_in_chunks("predictions", pred_records, on_conflict="cycle_id,station_id,target_at,model_version")

    real_lookup = observations.set_index(["station_id", "observed_at"])["demand"]
    eval_records = []
    for row in batch.to_dict(orient="records"):
        key = (row["station_id"], row["target_at"])
        if key not in real_lookup.index:
            continue
        eval_records.append(
            {
                "station_id": row["station_id"],
                "target_at": row["target_at"].isoformat(),
                "model_version": champion["version"],
                "predicted": float(row["value"]),
                "real": float(real_lookup.loc[key]),
            }
        )
    db.insert_evaluations(eval_records)
    db.set_state("sim_cutoff", next_cutoff.isoformat())

    print(
        f"[simulación] {cycle_id}: {len(pred_records)} predicciones "
        f"({len(eval_records)} evaluadas de inmediato) con {champion['version']}."
    )


def run() -> None:
    try:
        cycle = get_current_cycle()
    except NoOpenCycle:
        cycle = None

    if cycle is not None:
        print(f"Ciclo real detectado ({cycle.get('cycle_id')}); delego en src.predict.")
        from src.predict import run as run_real

        run_real()
        return

    run_simulated_cycle()


if __name__ == "__main__":
    run()
