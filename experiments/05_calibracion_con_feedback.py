"""Experimento 05 — corregir el sesgo usando el feedback real de competencia.

Hallazgo que lo motiva: sobre 2.400 evaluaciones reales, el modelo predice
solo el 94.7% de la demanda observada (y en 09122 apenas el 82.5%). Es un
sesgo sistemático, no ruido: la demanda creció respecto al periodo de
entrenamiento.

La calibración que se probó en el experimento 04 no sirvió porque se estimaba
sobre histórico, donde no hay ese desfase. Aquí se estima con las
evaluaciones REALES que va devolviendo la competencia.

Backtest sin fuga: los ciclos se recorren en orden cronológico y el factor
de cada ciclo se calcula SOLO con evaluaciones anteriores a él.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, "/Users/angie/Desktop/mi-pulso-transmi")
load_dotenv("/Users/angie/Desktop/mi-pulso-transmi/.env")

from src import db

MIN_MUESTRAS = 20          # por estación, antes de confiar en un factor
LIMITES = (0.85, 1.25)     # tope al factor, para no amplificar un error puntual


def accuracy(frame: pd.DataFrame, col: str = "pred") -> float:
    per = frame.groupby("station_id").apply(
        lambda g: 100 * max(0.0, 1 - (g["real"] - g[col]).abs().sum() / g["real"].sum()),
        include_groups=False)
    return float(per.mean())


def cargar() -> pd.DataFrame:
    df = pd.DataFrame(db.fetch_all_rows(
        "evaluations", "station_id,target_at,predicted,real,source,cycle_id"))
    r = df[df["source"] == "real"].copy()
    r["target_at"] = pd.to_datetime(r["target_at"], utc=True)
    r["origen"] = pd.to_datetime(
        r["cycle_id"].str.extract(r"_(\d{8}T\d{6}Z)$")[0], format="%Y%m%dT%H%M%SZ", utc=True)
    return r.sort_values(["origen", "station_id", "target_at"]).reset_index(drop=True)


def backtest(df: pd.DataFrame, ventana_ciclos: int | None, por_estacion: bool) -> float:
    """Recorre los ciclos en orden; calibra con lo anterior, aplica al actual."""
    ciclos = list(dict.fromkeys(df["origen"]))
    salida = []
    for i, ciclo in enumerate(ciclos):
        actual = df[df["origen"] == ciclo].copy()
        previos = df[df["origen"] < ciclo]
        if ventana_ciclos:
            recientes = ciclos[max(0, i - ventana_ciclos):i]
            previos = previos[previos["origen"].isin(recientes)]

        actual["pred"] = actual["predicted"]
        if len(previos) > 0:
            if por_estacion:
                for sid, g in previos.groupby("station_id"):
                    if len(g) < MIN_MUESTRAS or g["predicted"].sum() <= 0:
                        continue
                    factor = np.clip(g["real"].sum() / g["predicted"].sum(), *LIMITES)
                    mask = actual["station_id"] == sid
                    actual.loc[mask, "pred"] = actual.loc[mask, "predicted"] * factor
            elif len(previos) >= MIN_MUESTRAS and previos["predicted"].sum() > 0:
                factor = np.clip(previos["real"].sum() / previos["predicted"].sum(), *LIMITES)
                actual["pred"] = actual["predicted"] * factor
        salida.append(actual)
    return accuracy(pd.concat(salida, ignore_index=True))


def main():
    df = cargar()
    ciclos = df["origen"].nunique()
    print(f"evaluaciones reales: {len(df):,} en {ciclos} ciclos")
    print(f"sesgo global predicho/real: {df['predicted'].sum()/df['real'].sum():.4f}\n")

    base = accuracy(df.assign(pred=df["predicted"]))
    print(f"{'sin calibrar (hoy)':44s} {base:6.2f}%")

    pruebas = [
        ("calibración global, todo el historial", None, False),
        ("calibración global, últimos 6 ciclos", 6, False),
        ("calibración global, últimos 12 ciclos", 12, False),
        ("calibración por estación, todo el historial", None, True),
        ("calibración por estación, últimos 6 ciclos", 6, True),
        ("calibración por estación, últimos 12 ciclos", 12, True),
        ("calibración por estación, últimos 24 ciclos", 24, True),
    ]
    mejor = ("sin calibrar", base)
    for nombre, ventana, por_est in pruebas:
        acc = backtest(df, ventana, por_est)
        marca = ""
        if acc > mejor[1]:
            mejor = (nombre, acc)
            marca = "  <-- mejor"
        print(f"{nombre:44s} {acc:6.2f}%  ({acc-base:+.2f}){marca}")

    print(f"\nmejor: {mejor[0]} -> {mejor[1]:.2f}% ({mejor[1]-base:+.2f} pts)")


if __name__ == "__main__":
    main()
