"""Features alineadas con el servicio real (sin train/serve skew).

El problema que esto resuelve: la versión anterior entrenaba con "el lag más
reciente = 15 min antes del target", pero al servir el horizonte +60 ese lag
tenía 60 minutos de antigüedad. Medido con la métrica oficial, ese desajuste
costaba ~5 puntos de accuracy y empeoraba con la distancia:

    +15min 85.9% | +30min 82.8% | +45min 77.3% | +60min 70.9%

La solución es nombrar las features por su posición relativa al target
("slots") en vez de por su rezago absoluto. `s0` siempre significa "la
observación más reciente disponible al momento de predecir", tanto al
entrenar como al servir, sea cual sea el horizonte.
"""

from __future__ import annotations

import pandas as pd

STEPS_PER_HOUR = 4
STEPS_PER_DAY = 24 * STEPS_PER_HOUR      # 96
STEPS_PER_WEEK = 7 * STEPS_PER_DAY       # 672

# Horizontes del contrato: +15, +30, +45, +60 minutos = 1..4 pasos.
HORIZONS = (1, 2, 3, 4)

# Desplazamientos de cada slot, relativos al horizonte h.
SLOT_OFFSETS = {
    "s0": 0, "s1": 1, "s2": 2, "s3": 3, "s4": 4, "s8": 8,
    "sday": STEPS_PER_DAY, "sday2": 2 * STEPS_PER_DAY,
    "sday_prev": STEPS_PER_DAY - 1, "sday_next": STEPS_PER_DAY + 1,
    "sweek": STEPS_PER_WEEK, "sweek2": 2 * STEPS_PER_WEEK,
}
ROLLING = ["roll4", "roll16", "roll96"]
DERIVED = ["std16", "trend", "day_delta"]
CALENDAR = ["hour", "minute", "dow", "is_weekend", "tod"]

FEATURE_COLUMNS = list(SLOT_OFFSETS) + ROLLING + DERIVED + CALENDAR + ["horizon"]

# --- Features adaptativas al nivel (modelo global normalizado) ---------------
#
# El modelo por estación seguía a `sday`/`sweek` y a niveles absolutos. Cuando
# la estación 05100 pasó a operar al ~40% de su demanda normal (13-sep, 14:00
# virtual) siguió prediciendo el doble durante más de un día: 7.4% de accuracy
# en esa estación y −6.5 puntos en el promedio. Un árbol no extrapola a
# niveles que nunca vio.
#
# La corrección: comparar lo reciente contra la misma ventana ayer y hace una
# semana (ratios `lvl*`), y expresar todo relativo a una escala de nivel
# reciente. Así un corte o un desvío de demanda se ve en cuanto aparece.
EPS = 1.0
TARGET_LAGS = {"tday": STEPS_PER_DAY, "tweek": STEPS_PER_WEEK, "tweek2": 2 * STEPS_PER_WEEK}
LEVEL_COLUMNS = (
    list(SLOT_OFFSETS) + list(TARGET_LAGS) + ROLLING
    + ["roll4_d", "roll16_d", "roll4_w", "roll16_w", "roll96_w", "exp_w", "exp_d"]
)
RATIO_COLUMNS = ["lvl4_d", "lvl4_w", "lvl16_d", "lvl16_w", "lvl96_w"]
GLOBAL_FEATURES = LEVEL_COLUMNS + RATIO_COLUMNS + CALENDAR + ["horizon", "station_code"]


def build_design(observations: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Matriz de diseño para un horizonte dado.

    Solo usa información disponible `horizon` pasos antes de cada target, así
    que la misma función sirve para entrenar y para inferir.
    """
    df = observations.sort_values(["station_id", "observed_at"]).copy()
    grouped = df.groupby("station_id")["demand"]

    for slot, offset in SLOT_OFFSETS.items():
        df[slot] = grouped.shift(horizon + offset)

    base = grouped.shift(horizon)
    df["roll4"] = base.rolling(4).mean()
    df["roll16"] = base.rolling(16).mean()
    df["roll96"] = base.rolling(STEPS_PER_DAY).mean()
    df["std16"] = base.rolling(16).std()
    df["trend"] = df["s0"] - df["s4"]
    df["day_delta"] = df["s0"] - df["sday"]

    df["hour"] = df["observed_at"].dt.hour
    df["minute"] = df["observed_at"].dt.minute
    df["dow"] = df["observed_at"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["tod"] = df["hour"] * STEPS_PER_HOUR + df["minute"] // 15
    df["horizon"] = horizon
    _add_level_features(df, grouped, horizon)
    return df


def _rolling_by_station(serie: pd.Series, station: pd.Series, window: int) -> pd.Series:
    return serie.groupby(station).transform(lambda s: s.rolling(window).mean())


def _add_level_features(df: pd.DataFrame, grouped, horizon: int) -> None:
    # Demanda del mismo instante del target ayer / hace 1 y 2 semanas. Son
    # rezagos >= 96 pasos, así que están disponibles para cualquier horizonte.
    for columna, offset in TARGET_LAGS.items():
        df[columna] = grouped.shift(offset)

    for sufijo, offset in (("d", STEPS_PER_DAY), ("w", STEPS_PER_WEEK)):
        pasado = grouped.shift(horizon + offset)
        df[f"roll4_{sufijo}"] = _rolling_by_station(pasado, df["station_id"], 4)
        df[f"roll16_{sufijo}"] = _rolling_by_station(pasado, df["station_id"], 16)
    df["roll96_w"] = _rolling_by_station(
        grouped.shift(horizon + STEPS_PER_WEEK), df["station_id"], STEPS_PER_DAY
    )

    df["lvl4_d"] = (df["roll4"] + EPS) / (df["roll4_d"] + EPS)
    df["lvl4_w"] = (df["roll4"] + EPS) / (df["roll4_w"] + EPS)
    df["lvl16_d"] = (df["roll16"] + EPS) / (df["roll16_d"] + EPS)
    df["lvl16_w"] = (df["roll16"] + EPS) / (df["roll16_w"] + EPS)
    df["lvl96_w"] = (df["roll96"] + EPS) / (df["roll96_w"] + EPS)

    # Expectativa adaptativa: lo de ayer / la semana pasada a esa hora,
    # escalado por cuánto se desvía hoy el nivel reciente.
    df["exp_w"] = df["tweek"] * df["lvl16_w"]
    df["exp_d"] = df["tday"] * df["lvl16_d"]


def build_training_design(observations: pd.DataFrame) -> pd.DataFrame:
    """Los 4 horizontes apilados, con `horizon` como feature.

    Permite un solo modelo por estación en vez de uno por (estación, horizonte):
    misma accuracy (86.69% vs 86.64%) con 4x menos artefactos.
    """
    partes = [build_design(observations, h) for h in HORIZONS]
    stacked = pd.concat(partes, ignore_index=True)
    return stacked.dropna(subset=FEATURE_COLUMNS + ["demand"])


def build_global_training_design(observations: pd.DataFrame) -> pd.DataFrame:
    """Como build_training_design, pero exige también las features de nivel
    (necesitan 2 semanas de historia por los rezagos semanales)."""
    partes = [build_design(observations, h) for h in HORIZONS]
    stacked = pd.concat(partes, ignore_index=True)
    return stacked.dropna(subset=FEATURE_COLUMNS + LEVEL_COLUMNS + ["demand"]).reset_index(drop=True)


def build_serving_design(observations: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Features para instantes futuros que todavía no están en `observations`.

    `targets` necesita columnas station_id, target_at y horizon. Se añaden los
    targets como filas con demanda desconocida para que los rezagos se calculen
    desde el historial real, y luego se extraen esas filas.
    """
    placeholders = targets.rename(columns={"target_at": "observed_at"})[
        ["station_id", "observed_at"]
    ].copy()
    placeholders["demand"] = pd.NA

    combinado = pd.concat(
        [observations[["station_id", "observed_at", "demand"]], placeholders],
        ignore_index=True,
    ).drop_duplicates(subset=["station_id", "observed_at"], keep="first")
    combinado["demand"] = pd.to_numeric(combinado["demand"], errors="coerce")

    filas = []
    for horizon in sorted(targets["horizon"].unique()):
        design = build_design(combinado, int(horizon))
        pedidos = targets[targets["horizon"] == horizon]
        clave = set(zip(pedidos["station_id"], pedidos["target_at"]))
        seleccion = design[
            [(s, t) in clave for s, t in zip(design["station_id"], design["observed_at"])]
        ]
        filas.append(seleccion)

    return pd.concat(filas, ignore_index=True)
