"""Experimento 04 — ¿se puede subir de 86.7%?

Prueba dos palancas que no se habían explorado:
  1. Ponderar por recencia (los datos viejos pesan menos).
  2. Afinar hiperparámetros del boosting.
  3. Calibración multiplicativa por estación (corregir sesgo sistemático).

Protocolo idéntico a los experimentos anteriores: métrica oficial
(WAPE -> accuracy por estación, promedio), últimos 5 días como validación,
los 4 horizontes, mismo conjunto en todas las variantes.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "/Users/angie/Desktop/mi-pulso-transmi")
load_dotenv("/Users/angie/Desktop/mi-pulso-transmi/.env")

from src.data import fetch_all_observations
from src.features import FEATURE_COLUMNS, build_training_design

VALIDATION_DAYS = 5


def accuracy(frame: pd.DataFrame) -> float:
    per = frame.groupby("station_id").apply(
        lambda g: 100 * max(0.0, 1 - (g["demand"] - g["pred"]).abs().sum() / g["demand"].sum()),
        include_groups=False)
    return float(per.mean())


def run(design, *, half_life_days=None, params=None, calibrate=False):
    val_start = design["observed_at"].max() - pd.Timedelta(days=VALIDATION_DAYS)
    tr, va = design[design["observed_at"] < val_start], design[design["observed_at"] >= val_start]
    base_params = dict(loss="absolute_error", max_iter=600, learning_rate=0.05, random_state=42)
    base_params.update(params or {})

    preds = []
    for sid, g_tr in tr.groupby("station_id", observed=True):
        g_va = va[va["station_id"] == sid]
        if g_va.empty:
            continue
        modelo = HistGradientBoostingRegressor(**base_params)

        peso = None
        if half_life_days:
            edad = (g_tr["observed_at"].max() - g_tr["observed_at"]).dt.total_seconds() / 86400
            peso = 0.5 ** (edad / half_life_days)

        modelo.fit(g_tr[FEATURE_COLUMNS], g_tr["demand"], sample_weight=peso)
        p = np.clip(modelo.predict(g_va[FEATURE_COLUMNS]), 0, None)

        if calibrate:
            # factor que corrige sesgo sistemático, estimado en el tramo final
            # del entrenamiento (nunca en validación)
            corte = g_tr["observed_at"].max() - pd.Timedelta(days=2)
            reciente = g_tr[g_tr["observed_at"] >= corte]
            if len(reciente) > 50:
                p_ref = np.clip(modelo.predict(reciente[FEATURE_COLUMNS]), 0, None)
                if p_ref.sum() > 0:
                    factor = reciente["demand"].sum() / p_ref.sum()
                    p = p * np.clip(factor, 0.8, 1.25)

        preds.append(g_va.assign(pred=p)[["station_id", "observed_at", "demand", "pred", "horizon"]])

    out = pd.concat(preds, ignore_index=True)
    por_h = out.groupby("horizon").apply(
        lambda g: accuracy(g), include_groups=False).round(2).to_dict()
    return accuracy(out), por_h


def main():
    t0 = time.time()
    observations = fetch_all_observations()
    design = build_training_design(observations)
    print(f"datos: {len(observations):,} obs hasta {observations['observed_at'].max()}")
    print(f"diseño: {len(design):,} filas\n")

    pruebas = [
        ("Q. actual (referencia)", dict()),
        ("R. + recencia media vida 7d", dict(half_life_days=7)),
        ("S. + recencia media vida 14d", dict(half_life_days=14)),
        ("T. + recencia media vida 21d", dict(half_life_days=21)),
        ("U. + más iteraciones, lr menor", dict(params=dict(max_iter=1200, learning_rate=0.03))),
        ("V. + hojas y regularización", dict(params=dict(max_iter=900, learning_rate=0.04,
                                                         max_leaf_nodes=63, l2_regularization=1.0))),
        ("W. + calibración por estación", dict(calibrate=True)),
    ]
    mejor = ("", 0.0)
    for nombre, kw in pruebas:
        acc, por_h = run(design, **kw)
        marca = ""
        if acc > mejor[1]:
            mejor = (nombre, acc)
            marca = "  <-- mejor"
        print(f"{nombre:34s} {acc:6.2f}%  {por_h}{marca}")
    print(f"\nmejor: {mejor[0]} con {mejor[1]:.2f}%")
    print(f"tiempo: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
