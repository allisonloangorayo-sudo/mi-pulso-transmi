"""Modelos: entrenamiento y predicción de cada tipo de bundle.

Hay dos tipos de bundle y ambos deben poder predecir, porque el duelo de
promoción enfrenta al candidato contra el champion vigente sobre la misma
ventana, y el champion puede ser de la generación anterior:

- "per_station" (anterior): un HistGradientBoosting por estación sobre
  demanda cruda. Sigue a `sday`/`sweek` y no se adapta a cambios de nivel.
- "global_norm" (actual): un solo modelo para las 12 estaciones con la
  estación como feature categórica, entrenado sobre demanda *normalizada*
  por una escala de nivel reciente. Es un ensamble de 3 escalas distintas
  porque cada una gana en estaciones distintas (medido en
  experiments/06_nivel_adaptativo.py).

Normalizar el objetivo cambia la pérdida: MAE sobre y/s pondera cada error
por 1/s. Para que siga siendo WAPE (error absoluto en unidades reales) se
entrena con sample_weight = s.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.features import EPS, FEATURE_COLUMNS, GLOBAL_FEATURES, LEVEL_COLUMNS

GLOBAL_SCALES = ("roll96", "roll16", "exp_w")


def _per_station_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=600, learning_rate=0.05, random_state=42
    )


def _global_model(station_feature: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        categorical_features=[station_feature], random_state=42,
    )


def _normalized(design: pd.DataFrame, stations: list[str], scale: str):
    X = design.reindex(columns=GLOBAL_FEATURES).copy()
    codigos = {s: i for i, s in enumerate(stations)}
    X["station_code"] = design["station_id"].astype(str).map(codigos).fillna(-1).astype(int)
    s = np.maximum(design[scale].to_numpy(dtype=float), EPS)
    for columna in LEVEL_COLUMNS:
        X[columna] = X[columna].to_numpy(dtype=float) / s
    return X, s


def train_global(design: pd.DataFrame) -> dict:
    stations = sorted(design["station_id"].astype(str).unique())
    modelos = {}
    for scale in GLOBAL_SCALES:
        X, s = _normalized(design, stations, scale)
        modelo = _global_model(GLOBAL_FEATURES.index("station_code"))
        modelo.fit(X, design["demand"].to_numpy(dtype=float) / s, sample_weight=s)
        modelos[scale] = modelo
    return {"kind": "global_norm", "stations": stations, "models": modelos, "features": GLOBAL_FEATURES}


def train_per_station(design: pd.DataFrame) -> dict:
    modelos = {}
    for station_id, grupo in design.groupby("station_id", observed=True):
        modelo = _per_station_model()
        modelo.fit(grupo[FEATURE_COLUMNS], grupo["demand"])
        modelos[str(station_id)] = modelo
    return {"kind": "per_station", "models": modelos, "features": FEATURE_COLUMNS}


def predict(bundle: dict, design: pd.DataFrame) -> np.ndarray:
    design = design.reset_index(drop=True)
    if bundle.get("kind") == "global_norm":
        salidas = []
        for scale, modelo in bundle["models"].items():
            X, s = _normalized(design, bundle["stations"], scale)
            salidas.append(modelo.predict(X) * s)
        return np.clip(np.mean(salidas, axis=0), 0, None)

    # Bundles sin "kind" son los por estación de la generación anterior.
    salida = np.zeros(len(design))
    for station_id, grupo in design.groupby("station_id", observed=True):
        modelo = bundle["models"].get(str(station_id))
        if modelo is None:
            raise RuntimeError(f"El bundle no tiene modelo para la estación {station_id}.")
        salida[grupo.index.to_numpy()] = modelo.predict(grupo[FEATURE_COLUMNS])
    return np.clip(salida, 0, None)


def required_features(bundle: dict) -> list[str]:
    if bundle.get("kind") == "global_norm":
        return [c for c in GLOBAL_FEATURES if c != "station_code"]
    return FEATURE_COLUMNS
