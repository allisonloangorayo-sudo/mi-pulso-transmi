"""Construcción de features a partir de observaciones (+ contexto opcional).

Granularidad: 15 minutos → 96 pasos = 24 horas, 672 pasos = 7 días.
"""

from __future__ import annotations

import pandas as pd

STEPS_PER_HOUR = 4
STEPS_PER_DAY = 24 * STEPS_PER_HOUR
STEPS_PER_WEEK = 7 * STEPS_PER_DAY

LAG_STEPS = [1, 4, STEPS_PER_DAY, STEPS_PER_WEEK]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["observed_at"].dt.hour
    df["minute"] = df["observed_at"].dt.minute
    df["dayofweek"] = df["observed_at"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["station_id", "observed_at"]).copy()
    grouped = df.groupby("station_id")["demand"]
    for lag in LAG_STEPS:
        df[f"lag_{lag}"] = grouped.shift(lag)
    df["rolling_mean_4"] = grouped.shift(1).rolling(4).mean().reset_index(level=0, drop=True)
    return df


def build_features(observations: pd.DataFrame) -> pd.DataFrame:
    """Devuelve observations + features de calendario y rezagos.

    Las primeras STEPS_PER_WEEK filas de cada estación quedan con NaN en los
    lags largos; se descartan antes de entrenar (ver train.py).
    """
    df = add_calendar_features(observations)
    df = add_lag_features(df)
    return df


FEATURE_COLUMNS = [
    "hour",
    "minute",
    "dayofweek",
    "is_weekend",
    *[f"lag_{lag}" for lag in LAG_STEPS],
    "rolling_mean_4",
]
