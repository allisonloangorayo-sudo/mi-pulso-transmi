"""Experimento 06: modelo adaptativo al nivel vs. modelo por estación.

Backtest que replica los ciclos reales: corte a la hora en punto y targets a
:15, :30, :45 y :00, que corresponden a los horizontes 1, 2, 3 y 4. Como
`build_design` solo usa datos con rezago >= horizonte, la fila del diseño de
un target es exactamente lo que se habría servido en ese ciclo.

Ventana: 11-sep 17:00 → 15-sep 04:00 (UTC), la misma de las evaluaciones
reales. Reentreno diario a las 05:00 (cada modelo solo ve datos anteriores a
su corte).

Uso (necesita PULSO_API_KEY para leer el stream):
    python experiments/06_nivel_adaptativo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import model  # noqa: E402
from src.data import fetch_all_observations  # noqa: E402
from src.features import build_global_training_design  # noqa: E402

INICIO = pd.Timestamp("2026-09-11 17:00", tz="UTC")
REENTRENOS = ["2026-09-12 05:00", "2026-09-13 05:00", "2026-09-14 05:00"]
MINUTO_A_HORIZONTE = {15: 1, 30: 2, 45: 3, 0: 4}


def filas_de_ciclo(design: pd.DataFrame) -> pd.DataFrame:
    return design[design["horizon"] == design["observed_at"].dt.minute.map(MINUTO_A_HORIZONTE)]


def accuracy_por_estacion(frame: pd.DataFrame) -> pd.Series:
    error = (frame["demand"] - frame["prediction"]).abs().groupby(frame["station_id"]).sum()
    return (100 * (1 - error / frame["demand"].groupby(frame["station_id"]).sum())).clip(lower=0)


def backtest(design: pd.DataFrame, entrenar) -> pd.Series:
    cortes = [INICIO] + [pd.Timestamp(c, tz="UTC") for c in REENTRENOS]
    partes = []
    for i, corte in enumerate(cortes):
        fin = cortes[i + 1] if i + 1 < len(cortes) else design["observed_at"].max()
        ventana = filas_de_ciclo(
            design[(design["observed_at"] > corte) & (design["observed_at"] <= fin)]
        ).reset_index(drop=True)
        if ventana.empty:
            continue
        bundle = entrenar(design[design["observed_at"] <= corte])
        partes.append(ventana.assign(prediction=model.predict(bundle, ventana)))
    return accuracy_por_estacion(pd.concat(partes))


def main() -> None:
    design = build_global_training_design(fetch_all_observations())
    for nombre, entrenar in {
        "por estación (anterior)": model.train_per_station,
        "global normalizado, ensamble": model.train_global,
    }.items():
        acc = backtest(design, entrenar)
        print(f"{nombre:32s} {acc.mean():6.2f}  | " + " ".join(f"{k}:{v:.1f}" for k, v in acc.items()))


if __name__ == "__main__":
    np.set_printoptions(precision=2)
    main()
