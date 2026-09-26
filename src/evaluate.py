"""Cierra el ciclo: predicción real -> valor observado -> evaluación.

Este paso faltaba. `predict.py` guardaba las predicciones de cada ciclo
oficial, pero nadie volvía después a compararlas con la demanda real cuando
la API la liberaba. Resultado: 2.400 predicciones reales y 0 evaluadas, así
que el drift solo veía lotes simulados sobre historia que el modelo ya
conocía (89.5%) en vez del desempeño real en competencia (~84%).

Corre después del collector, que es justo cuando llegan los datos nuevos que
resuelven los targets pendientes.
"""

from __future__ import annotations

import pandas as pd
from dotenv import load_dotenv

from src import db
from src.data import fetch_all_observations

load_dotenv()


def pending_real_predictions() -> pd.DataFrame:
    """Predicciones de ciclos oficiales que todavía no tienen evaluación."""
    predicciones = pd.DataFrame(
        db.fetch_all_rows("predictions", "cycle_id,station_id,target_at,model_version,value")
    )
    if predicciones.empty:
        return predicciones

    reales = predicciones[~predicciones["cycle_id"].str.startswith("sim-")].copy()
    if reales.empty:
        return reales

    evaluadas = pd.DataFrame(
        db.fetch_all_rows("evaluations", "station_id,target_at,model_version,source")
    )
    if not evaluadas.empty:
        ya = evaluadas[evaluadas["source"] == "real"]
        claves = set(zip(ya["station_id"], ya["target_at"], ya["model_version"]))
        reales = reales[
            [
                (s, t, m) not in claves
                for s, t, m in zip(reales["station_id"], reales["target_at"], reales["model_version"])
            ]
        ]
    return reales


def run() -> None:
    pendientes = pending_real_predictions()
    if pendientes.empty:
        print("No hay predicciones reales pendientes de evaluar.")
        return

    observaciones = fetch_all_observations()
    real = observaciones.set_index(["station_id", "observed_at"])["demand"]

    registros = []
    for fila in pendientes.to_dict(orient="records"):
        clave = (fila["station_id"], pd.Timestamp(fila["target_at"]))
        if clave not in real.index:
            continue  # la API todavía no libera ese instante
        registros.append({
            "station_id": fila["station_id"],
            "target_at": pd.Timestamp(fila["target_at"]).isoformat(),
            "model_version": fila["model_version"],
            "predicted": float(fila["value"]),
            "real": float(real.loc[clave]),
            "source": "real",
            "cycle_id": fila["cycle_id"],
        })

    if not registros:
        print(f"{len(pendientes)} predicciones pendientes, ninguna resuelta todavía por la API.")
        return

    db.upsert_in_chunks(
        "evaluations", registros, on_conflict="station_id,target_at,model_version,source"
    )

    marco = pd.DataFrame(registros)
    marco["abs_error"] = (marco["predicted"] - marco["real"]).abs()
    por_estacion = marco.groupby("station_id").apply(
        lambda g: 100 * max(0.0, 1 - g["abs_error"].sum() / g["real"].sum()), include_groups=False
    )
    print(f"Evaluadas {len(registros)} predicciones reales "
          f"({len(pendientes) - len(registros)} siguen pendientes).")
    print(f"Accuracy de este lote real: {por_estacion.mean():.2f}")


if __name__ == "__main__":
    run()
