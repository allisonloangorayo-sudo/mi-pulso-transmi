"""Experimento 07: periodo adaptativo + ensamble online de expertos.

Contexto: el 2026-10-04 la accuracy de los últimos 6 ciclos cayó a 0. Causas:
(1) el stream cambió de esquema (`measurement.value`) y todo quedó NaN — ver
src/data.py — y (2) el generador cambió la forma de la serie: ciclo diario ->
ciclo de 4 h (18-sep) -> ciclo de ~8 h (20-sep 12:00).

Backtest walk-forward con el código de producción (src.adaptive): reentrena
cada RETRAIN_H horas y predice los 4 horizontes de cada instante del bloque
siguiente, como los ciclos reales. Compara contra la receta anterior
(src.model.train_global) sobre exactamente las mismas filas.

Uso:
    python experiments/07_regimen_adaptativo.py [observaciones.pkl]
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src import adaptive, model  # noqa: E402
from src.features import FEATURE_COLUMNS, LEVEL_COLUMNS, build_design as build_old  # noqa: E402

def _opcion(nombre: str, defecto: str) -> str:
    for a in sys.argv[1:]:
        if a.startswith(f"--{nombre}="):
            return a.split("=", 1)[1]
    return defecto


START = pd.Timestamp(_opcion("desde", "2026-09-14 01:30"), tz="UTC")
RETRAIN_H = int(_opcion("reentreno-h", "12"))
COMPARE_OLD = "--sin-anterior" not in sys.argv
CONTEXT_DAYS = 16  # el experto "nivel" necesita rezagos de 2 semanas


def accuracy(frame: pd.DataFrame, columna: str) -> float:
    error = (frame["demand"] - frame[columna]).abs().groupby(frame["station_id"]).sum()
    wape = error / frame["demand"].groupby(frame["station_id"]).sum()
    return float((100 * (1 - wape)).clip(lower=0).mean())


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        observations = pd.read_pickle(args[0])
    else:
        from src.data import fetch_all_observations

        observations = fetch_all_observations()
    observations = observations[["station_id", "observed_at", "demand", "imputed"]]

    nuevo = adaptive.build_design(observations)
    if COMPARE_OLD:
        viejo = pd.concat([build_old(observations, h) for h in adaptive.HORIZONS], ignore_index=True)
        viejo = viejo.dropna(subset=FEATURE_COLUMNS + LEVEL_COLUMNS + ["demand"])
        viejo["imputed"] = viejo["imputed"].astype(bool)

    fin = observations["observed_at"].max()
    cortes = list(pd.date_range(START, fin, freq=f"{RETRAIN_H}h"))
    if fin - cortes[-1] < pd.Timedelta(hours=1):
        cortes = cortes[:-1]  # un bloque final sin datos no se puede evaluar
    resultados = []
    for a, b in zip(cortes, cortes[1:] + [fin]):
        t0 = time.time()
        bundle = adaptive.train(nuevo[nuevo["observed_at"] <= a])
        # El ensamble necesita la historia previa para ponderar expertos:
        # se predice sobre todo el diseño conocido y se toma el bloque.
        ventana = nuevo[(nuevo["observed_at"] > a - pd.Timedelta(days=CONTEXT_DAYS))
                        & (nuevo["observed_at"] <= b)]
        ventana = ventana.reset_index(drop=True).copy()
        ventana["nuevo"] = adaptive.predict(bundle, ventana)
        expertos = adaptive.expert_forecasts(bundle, ventana)
        for e in expertos.columns:
            ventana[f"x_{e}"] = expertos[e].to_numpy()
        bloque = ventana[(ventana["observed_at"] > a) & ~ventana["imputed"].astype(bool)].copy()
        bloque = bloque.dropna(subset=["demand"])

        if COMPARE_OLD:
            anterior = model.train_global(viejo[(viejo["observed_at"] <= a) & ~viejo["imputed"]])
            vb = viejo[(viejo["observed_at"] > a) & (viejo["observed_at"] <= b) & ~viejo["imputed"]]
            vb = vb.reset_index(drop=True)
            vb["anterior"] = model.predict(anterior, vb)
            bloque = bloque.merge(vb[["station_id", "observed_at", "horizon", "anterior"]],
                                  on=["station_id", "observed_at", "horizon"], how="left")
        resultados.append(bloque)
        resumen = {"nuevo": accuracy(bloque, "nuevo")}
        if COMPARE_OLD:
            resumen["anterior"] = accuracy(bloque.dropna(subset=["anterior"]), "anterior")
        print(a, f"{time.time() - t0:.0f}s", {k: round(v, 2) for k, v in resumen.items()}, flush=True)

    todo = pd.concat(resultados, ignore_index=True)
    todo["dia"] = todo["observed_at"].dt.floor("D")
    columnas = ["nuevo"] + (["anterior"] if COMPARE_OLD else []) + [c for c in todo if c.startswith("x_")]
    comun = todo.dropna(subset=columnas)
    print("\nPor día:")
    print(comun.groupby("dia").apply(lambda g: pd.Series({c: accuracy(g, c) for c in columnas})).round(2))
    print("\nPor horizonte:")
    print(comun.groupby("horizon").apply(lambda g: pd.Series({c: accuracy(g, c) for c in columnas})).round(2))
    print("\nTOTAL:", {c: round(accuracy(comun, c), 2) for c in columnas})
    ultimas = comun[comun["observed_at"] > fin - pd.Timedelta(hours=6)]
    print("Últimas 6 h:", {c: round(accuracy(ultimas, c), 2) for c in columnas})


if __name__ == "__main__":
    main()
