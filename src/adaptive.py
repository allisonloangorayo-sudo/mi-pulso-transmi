"""Modelo adaptativo al régimen: periodo detectado + ensamble de expertos.

Por qué existe (análisis del 2026-10-04 sobre los datos de la competencia):
el generador no solo cambia niveles, cambia la *forma* de la serie.

    hasta 18-sep 02:00   estacionalidad diaria (96 pasos)
    18-sep -> 20-sep     ciclo de 16 pasos (4 h), ACF ≈ 0.8 en las 12 estaciones
    desde 20-sep 12:00   ciclo de ~32 pasos (8 h), fase distinta por estación

Un modelo basado en "ayer / la semana pasada a esta hora" se hundió a 44-58%
en esos cambios. Esto lo resuelve en dos capas:

1. Features con periodo adaptativo: para cada estación y cada corte se elige
   el periodo P (8..100 pasos) cuyo "naive estacional" tuvo menor error en
   las últimas W observaciones (W=16 reacciona rápido, W=48 es estable), y
   se usan los valores en la misma fase uno y dos periodos atrás.
2. Ensamble online de expertos: un GBM global normalizado y varios
   pronosticadores simples. En cada corte, el peso de cada experto depende de
   su error reciente en esa estación y horizonte (solo con targets ya
   conocidos al corte). Cuando el régimen cambia, el peso migra en horas, sin
   esperar a un reentrenamiento.

Backtest walk-forward (reentreno cada 12 h, ver experiments/07):
ver README, sección "Modelo actual".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HORIZONS = (1, 2, 3, 4)
CANDIDATE_PERIODS = np.arange(8, 101)
WINDOWS = (16, 48)
EXPERT_WINDOW = 6          # targets recientes con los que se pondera a cada experto
EXPERT_TEMPERATURE = 20.0  # qué tan fuerte se castiga el error relativo
RECENCY_DAYS = 7.0         # vida media (e-fold) del peso de cada fila al entrenar
RECENT_DAYS = 3.0          # el experto "gbm_reciente" solo ve estos últimos días
SCALE = "r16"

LEVEL_FEATURES = (
    ["a0", "a1", "a2", "a3", "a4", "a5", "r4", "r16", "r96", "s96", "slope"]
    + [f"{p}{w}" for w in WINDOWS for p in ("sp", "base", "exp", "lin")]
    + [f"sp{w}m" for w in WINDOWS] + [f"sp{w}p" for w in WINDOWS] + [f"sp2_{w}" for w in WINDOWS]
)
OTHER_FEATURES = (
    [f"P{w}" for w in WINDOWS] + [f"R{w}" for w in WINDOWS] + [f"lv{w}" for w in WINDOWS] + ["horizon"]
)
FEATURES = LEVEL_FEATURES + OTHER_FEATURES
SIMPLE_EXPERTS = ["sp16", "exp16", "lin16", "exp48", "lin48", "a0"]
REQUIRED = ["a5", "r96", "sp16", "sp48", "sp2_48"]


def period_track(y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """P[t] y su error relativo, usando solo datos hasta t (inclusive)."""
    n = len(y)
    errores = np.zeros((len(CANDIDATE_PERIODS), n))
    validos = np.zeros((len(CANDIDATE_PERIODS), n), dtype=bool)
    for i, p in enumerate(CANDIDATE_PERIODS):
        if p < n:
            errores[i, p:] = np.abs(y[p:] - y[:-p])
            validos[i, p:] = True
    cero = np.zeros((len(CANDIDATE_PERIODS), 1))
    acum = np.concatenate([cero, np.cumsum(errores, axis=1)], axis=1)
    cuenta = np.concatenate([cero, np.cumsum(validos, axis=1)], axis=1)
    nivel = pd.Series(y).rolling(window, min_periods=1).mean().to_numpy() + 1.0

    periodo = np.full(n, 96)
    relativo = np.ones(n)
    for t in range(window, n):
        completo = (cuenta[:, t + 1] - cuenta[:, t + 1 - window]) == window
        media = np.where(completo, (acum[:, t + 1] - acum[:, t + 1 - window]) / window, np.inf)
        j = int(np.argmin(media))
        if np.isfinite(media[j]):
            periodo[t] = CANDIDATE_PERIODS[j]
            relativo[t] = media[j] / nivel[t]
    return periodo, relativo


def _station_features(y: np.ndarray, horizon: int, tracks: dict) -> dict[str, np.ndarray]:
    """Una fila por instante objetivo T; el corte es T - horizon."""
    n = len(y)
    corte = np.arange(n) - horizon
    ok = corte >= 0
    tt = np.clip(corte, 0, n - 1)
    T = np.arange(n)

    def at(idx: np.ndarray) -> np.ndarray:
        out = np.full(n, np.nan)
        m = ok & (idx >= 0) & (idx < n)
        out[m] = y[idx[m]]
        return out

    serie = pd.Series(y)
    r4 = serie.rolling(4).mean().to_numpy()
    f: dict[str, np.ndarray] = {f"a{k}": at(tt - k) for k in range(6)}
    f["r4"] = np.where(ok, r4[tt], np.nan)
    f["r16"] = np.where(ok, serie.rolling(16).mean().to_numpy()[tt], np.nan)
    f["r96"] = np.where(ok, serie.rolling(96).mean().to_numpy()[tt], np.nan)

    for w, (periodos, relativos) in tracks.items():
        p = np.where(ok, periodos[tt], 96)
        f[f"P{w}"] = p.astype(float)
        f[f"R{w}"] = np.where(ok, relativos[tt], np.nan)
        # Misma fase un periodo atrás; si eso todavía es futuro, dos periodos.
        fase = np.where(T - p <= tt, T - p, T - 2 * p)
        f[f"sp{w}"] = at(fase)
        f[f"sp{w}m"] = at(fase - 1)
        f[f"sp{w}p"] = np.where(fase + 1 <= tt, at(fase + 1), np.nan)
        f[f"sp2_{w}"] = at(fase - p)
        f[f"base{w}"] = at(tt - p)
        previo = np.full(n, np.nan)
        m = ok & (tt - p - 3 >= 0)
        previo[m] = r4[(tt - p)[m]]
        f[f"lv{w}"] = (f["r4"] + 1) / (previo + 1)
        f[f"exp{w}"] = f[f"sp{w}"] * np.clip(f[f"lv{w}"], 0.2, 5)
        f[f"lin{w}"] = np.clip(f[f"sp{w}"] + (f["a0"] - f[f"base{w}"]), 0, None)

    f["s96"] = at(T - 96)
    f["slope"] = f["a0"] - f["a2"]
    f["horizon"] = np.full(n, float(horizon))
    return f


def build_design(observations: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Diseño para todos los instantes de `observations` (grilla completa).

    Sirve igual para entrenar y para inferir: los instantes futuros se pasan
    como filas con demanda NaN y sus features solo miran hasta el corte.
    """
    partes = []
    ordenadas = observations.sort_values(["station_id", "observed_at"])
    for station_id, grupo in ordenadas.groupby("station_id", observed=True):
        y = pd.to_numeric(grupo["demand"], errors="coerce").to_numpy(dtype=float)
        conocida = np.where(np.isnan(y), 0.0, y)
        tracks = {w: period_track(conocida, w) for w in WINDOWS}
        for h in horizons:
            parte = pd.DataFrame(_station_features(y, int(h), tracks))
            parte["station_id"] = station_id
            parte["observed_at"] = grupo["observed_at"].to_numpy()
            parte["demand"] = y
            parte["imputed"] = (
                grupo["imputed"].to_numpy() if "imputed" in grupo else np.zeros(len(y), dtype=bool)
            )
            partes.append(parte)
    design = pd.concat(partes, ignore_index=True)
    design["station_id"] = design["station_id"].astype("string")
    return design


def _matrix(design: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    escala = np.maximum(design[SCALE].to_numpy(dtype=float), 1.0)
    X = design[LEVEL_FEATURES].div(escala, axis=0)
    for columna in OTHER_FEATURES:
        X[columna] = design[columna].to_numpy(dtype=float)
    return X, escala


def _observations_from(design: pd.DataFrame) -> pd.DataFrame:
    """La serie original (una fila por estación e instante) a partir del diseño."""
    base = design[design["horizon"] == design["horizon"].min()]
    return base[["station_id", "observed_at", "demand"]].reset_index(drop=True)


def _level_model_design(design: pd.DataFrame) -> pd.DataFrame:
    """Features del modelo por nivel (src.features) para las mismas filas."""
    from src.features import build_design as build_level_design

    observaciones = _observations_from(design)
    partes = [build_level_design(observaciones, int(h)) for h in sorted(design["horizon"].unique())]
    nivel = pd.concat(partes, ignore_index=True)
    nivel["horizon"] = nivel["horizon"].astype(float)
    llaves = design[["station_id", "observed_at", "horizon"]].reset_index()
    nivel = nivel.drop(columns=["demand"]).merge(llaves, on=["station_id", "observed_at", "horizon"])
    return nivel.set_index("index").reindex(design.index)


def train(design: pd.DataFrame) -> dict:
    """GBM global sobre demanda normalizada; pesa más lo reciente.

    sample_weight = escala × recencia: la escala mantiene la pérdida en
    unidades reales (WAPE) y la recencia hace que un régimen nuevo pese
    aunque sea minoría en el histórico.

    Además entrena el modelo por nivel de la generación anterior
    (src.model.train_global) como un experto más: en régimen diario sigue
    siendo el mejor (~85-87%), y el ensamble lo elige solo cuando acierta.
    """
    filas = design.dropna(subset=REQUIRED + ["demand"])
    filas = filas[~filas["imputed"].astype(bool)]
    X, escala = _matrix(filas)
    edad = (filas["observed_at"].max() - filas["observed_at"]).dt.total_seconds() / 86400
    peso = escala * np.exp(-edad.to_numpy() / RECENCY_DAYS)
    gbm = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=300, learning_rate=0.06, max_leaf_nodes=31, random_state=42
    )
    gbm.fit(X, filas["demand"].to_numpy(dtype=float) / escala, sample_weight=peso)

    # Experto "gbm_reciente": solo los últimos días. Aprende el régimen nuevo
    # sin que la historia vieja lo diluya (walk-forward desde 18-sep, reentreno
    # cada 3 h: ensamble 83.24 -> 83.58, mejor en los 4 días).
    recientes = filas[filas["observed_at"] > filas["observed_at"].max() - pd.Timedelta(days=RECENT_DAYS)]
    Xr, escala_r = _matrix(recientes)
    gbm_reciente = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=200, learning_rate=0.06, max_leaf_nodes=15, random_state=42
    )
    gbm_reciente.fit(Xr, recientes["demand"].to_numpy(dtype=float) / escala_r, sample_weight=escala_r)

    from src import model
    from src.features import FEATURE_COLUMNS, LEVEL_COLUMNS

    nivel = _level_model_design(design)
    nivel = nivel.assign(demand=design["demand"], imputed=design["imputed"])
    nivel = nivel.dropna(subset=FEATURE_COLUMNS + LEVEL_COLUMNS + ["demand"])
    nivel = nivel[~nivel["imputed"].astype(bool)]
    return {
        "kind": "adaptive_ensemble",
        "gbm": gbm,
        "gbm_recent": gbm_reciente,
        "level_model": model.train_global(nivel.reset_index(drop=True)),
        "features": FEATURES,
        "experts": ["gbm", "gbm_reciente", "nivel"] + SIMPLE_EXPERTS,
        "expert_window": EXPERT_WINDOW,
        "temperature": EXPERT_TEMPERATURE,
    }


def expert_forecasts(bundle: dict, design: pd.DataFrame) -> pd.DataFrame:
    salida = pd.DataFrame(index=design.index)
    validas = design[REQUIRED].notna().all(axis=1).to_numpy()
    gbm = np.full(len(design), np.nan)
    reciente = np.full(len(design), np.nan)
    if validas.any():
        X, escala = _matrix(design[validas])
        gbm[validas] = bundle["gbm"].predict(X) * escala
        if bundle.get("gbm_recent") is not None:
            reciente[validas] = bundle["gbm_recent"].predict(X) * escala
    salida["gbm"] = gbm
    if bundle.get("gbm_recent") is not None:
        salida["gbm_reciente"] = reciente
    if bundle.get("level_model") is not None:
        from src import model

        nivel = _level_model_design(design)
        requeridas = model.required_features(bundle["level_model"])
        listas = nivel[requeridas].notna().all(axis=1).to_numpy()
        valores = np.full(len(design), np.nan)
        if listas.any():
            valores[listas] = model.predict(bundle["level_model"], nivel[listas].reset_index(drop=True))
        salida["nivel"] = valores
    for experto in SIMPLE_EXPERTS:
        salida[experto] = design[experto].to_numpy(dtype=float)
    # Un experto sin valor (historia corta) cae al último dato conocido.
    respaldo = design["a0"].to_numpy(dtype=float)
    for experto in salida.columns:
        salida[experto] = np.where(np.isfinite(salida[experto]), salida[experto], respaldo)
    return np.clip(salida, 0, None)


def predict(bundle: dict, design: pd.DataFrame) -> np.ndarray:
    """Combina los expertos con pesos según su error reciente.

    `design` debe traer, por estación y horizonte, la historia contigua antes
    de las filas a predecir (build_design ya lo hace): el error de un experto
    para el target T solo se conoce cuando T <= corte, por eso la media móvil
    se desplaza `horizon` filas.
    """
    design = design.reset_index(drop=True)
    expertos = expert_forecasts(bundle, design)
    nombres = list(expertos.columns)
    k = int(bundle.get("expert_window", EXPERT_WINDOW))
    temperatura = float(bundle.get("temperature", EXPERT_TEMPERATURE))
    resultado = np.full(len(design), np.nan)

    for (_, horizon), grupo in design.groupby(["station_id", "horizon"], observed=True, sort=False):
        orden = grupo.sort_values("observed_at").index.to_numpy()
        h = int(horizon)
        F = expertos.loc[orden, nombres].to_numpy(dtype=float)
        real = design.loc[orden, "demand"].to_numpy(dtype=float)
        if "imputed" in design:
            real = np.where(design.loc[orden, "imputed"].astype(bool).to_numpy(), np.nan, real)
        errores = pd.DataFrame(np.abs(F - real[:, None]))
        media = errores.rolling(k, min_periods=1).mean().shift(h).to_numpy()
        nivel = pd.Series(real).rolling(k, min_periods=1).mean().shift(h).to_numpy() + 1.0
        relativo = media / nivel[:, None]
        sin_dato = ~np.isfinite(relativo).any(axis=1)
        relativo = np.where(np.isfinite(relativo), relativo, np.inf)
        relativo[sin_dato] = 0.0  # sin evidencia: pesos iguales
        relativo = relativo - relativo.min(axis=1, keepdims=True)
        pesos = np.exp(-temperatura * relativo)
        pesos = pesos / pesos.sum(axis=1, keepdims=True)
        resultado[orden] = (pesos * F).sum(axis=1)

    return np.clip(resultado, 0, None)


def forecast_targets(bundle: dict, observations: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Predicción para `targets` (station_id, target_at, horizon) desde el historial."""
    placeholders = targets.rename(columns={"target_at": "observed_at"})[["station_id", "observed_at"]]
    placeholders = placeholders.assign(
        observed_at=pd.to_datetime(placeholders["observed_at"], utc=True).astype("datetime64[ns, UTC]"),
        demand=np.nan, imputed=False,
    ).drop_duplicates(
        subset=["station_id", "observed_at"]
    )
    historia = observations[["station_id", "observed_at", "demand"]].assign(
        imputed=observations["imputed"] if "imputed" in observations else False,
        observed_at=pd.to_datetime(observations["observed_at"], utc=True).astype("datetime64[ns, UTC]"),
    )
    # Basta con las últimas 3 semanas: el periodo máximo es 100 pasos y la
    # ponderación de expertos mira unas pocas horas.
    desde = historia["observed_at"].max() - pd.Timedelta(days=21)
    historia = historia[historia["observed_at"] >= desde]
    combinado = pd.concat([historia, placeholders], ignore_index=True).drop_duplicates(
        subset=["station_id", "observed_at"], keep="first"
    )
    combinado["station_id"] = combinado["station_id"].astype("string")

    horizontes = sorted(int(h) for h in targets["horizon"].unique())
    design = build_design(combinado, horizontes)
    design["value"] = predict(bundle, design)

    pedidos = targets.assign(
        station_id=targets["station_id"].astype("string"),
        target_at=pd.to_datetime(targets["target_at"], utc=True).astype("datetime64[ns, UTC]"),
        horizon=targets["horizon"].astype(int),
    )
    salida = pedidos.merge(
        design[["station_id", "observed_at", "horizon", "value", "a0"]].rename(
            columns={"observed_at": "target_at"}
        ).assign(
            horizon=lambda d: d["horizon"].astype(int),
            target_at=lambda d: pd.to_datetime(d["target_at"], utc=True).astype("datetime64[ns, UTC]"),
        ),
        on=["station_id", "target_at", "horizon"],
        how="left",
    )
    return salida
