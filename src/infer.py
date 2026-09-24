"""Inferencia horaria: 12 estaciones × 4 horizontes = 48 predicciones.

Dos modos, elegidos automáticamente:

1. Ciclo real: si `/v1/forecast-cycles/current` responde, delega en
   `src.predict`, que sigue el contrato oficial.
2. Simulación: entre ciclos, avanza un "reloj simulado" una hora por corrida
   sobre el histórico ya conocido. Como el dato real de esos instantes ya
   está en nuestra base, cada corrida se evalúa de inmediato y alimenta a
   `src/drift.py` sin esperar a que abra el siguiente ciclo oficial.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
from dotenv import load_dotenv

from src import db
from src.data import fetch_all_observations
from src.features import FEATURE_COLUMNS, HORIZONS, build_serving_design
from src.predict import (
    NoOpenCycle,
    double_check_predictions,
    get_current_cycle,
    load_champion,
)

load_dotenv()

SIM_STEP = timedelta(hours=1)
VALIDATION_DAYS = 5  # debe coincidir con src.train.VALIDATION_DAYS


def _initial_sim_cutoff(observations: pd.DataFrame) -> pd.Timestamp:
    return observations["observed_at"].max() - timedelta(days=VALIDATION_DAYS)


def _next_cutoff(observations: pd.DataFrame) -> pd.Timestamp:
    max_ts = observations["observed_at"].max()
    guardado = db.get_state("sim_cutoff")
    cutoff = pd.Timestamp(guardado) if guardado else _initial_sim_cutoff(observations)
    siguiente = cutoff + SIM_STEP
    if siguiente + timedelta(minutes=15 * max(HORIZONS)) > max_ts:
        siguiente = _initial_sim_cutoff(observations) + SIM_STEP
    return siguiente


def _build_targets(observations: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    filas = [
        {"station_id": station_id, "target_at": cutoff + timedelta(minutes=15 * h), "horizon": h}
        for station_id in observations["station_id"].unique()
        for h in HORIZONS
    ]
    return pd.DataFrame(filas)


def run_simulated_cycle() -> None:
    observations = fetch_all_observations()
    cutoff = _next_cutoff(observations)
    targets = _build_targets(observations, cutoff)

    conocido = observations[observations["observed_at"] <= cutoff]
    design = build_serving_design(conocido, targets).reset_index(drop=True)
    design = design.dropna(subset=FEATURE_COLUMNS)
    if design.empty:
        print("Simulación sin historial suficiente todavía.")
        return

    bundle, champion = load_champion()
    design["value"] = double_check_predictions(bundle, design)

    cycle_id = f"sim-{cutoff.isoformat()}"
    pred_records = [
        {
            "cycle_id": cycle_id,
            "station_id": fila["station_id"],
            "target_at": pd.Timestamp(fila["observed_at"]).isoformat(),
            "model_version": champion["version"],
            "value": float(fila["value"]),
        }
        for fila in design.to_dict(orient="records")
    ]
    db.upsert_in_chunks(
        "predictions", pred_records, on_conflict="cycle_id,station_id,target_at,model_version"
    )

    real = observations.set_index(["station_id", "observed_at"])["demand"]
    evaluaciones = []
    for fila in design.to_dict(orient="records"):
        clave = (fila["station_id"], pd.Timestamp(fila["observed_at"]))
        if clave not in real.index:
            continue
        evaluaciones.append({
            "station_id": fila["station_id"],
            "target_at": clave[1].isoformat(),
            "model_version": champion["version"],
            "predicted": float(fila["value"]),
            "real": float(real.loc[clave]),
        })
    db.insert_evaluations(evaluaciones)
    db.set_state("sim_cutoff", cutoff.isoformat())

    print(
        f"[simulación] {cycle_id}: {len(pred_records)} predicciones "
        f"({len(evaluaciones)} evaluadas) con {champion['version']}."
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
