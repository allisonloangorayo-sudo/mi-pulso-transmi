"""Carga de observaciones: histórico estático + stream incremental.

`/v1/observations` se congela en `history_end` del dataset inicial; todo lo
que libera la ventana competitiva llega solo por `/v1/stream/observations`.
Entrenamiento e inferencia deben ver exactamente las mismas fuentes, por eso
esta función vive aquí y no duplicada en cada módulo.

Cambio de esquema del stream (detectado 2026-10-04, virtual 2026-09-20 12:15):
desde ese instante las filas llegan como `schema_version: 2`:

    {"station_id": ..., "observed_at": ..., "schema_version": 2,
     "measurement": {"value": "546.00", "unit": "passengers", "quality": "observed"}}

en vez de `{"demand": 546}`. `value` es texto y puede venir `null` con
`quality: "missing"`. El código leía solo `demand`: todo lo posterior quedó
NaN, los rolling se volvieron NaN, las predicciones salieron NaN, la doble
verificación (`allclose(NaN, NaN)` es False) abortó cada envío y la accuracy
de los últimos ciclos cayó a 0. `parse_stream_records` acepta ambos esquemas
y `fill_gaps` imputa los huecos para que una medición faltante nunca vuelva a
tumbar una entrega.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pulso_transmi import PulsoTransmiClient
from pulso_transmi.client import PulsoTransmiError

from src.http_utils import request_with_retry

load_dotenv()

PULSO_API_URL = os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io")
PULSO_API_KEY = os.getenv("PULSO_API_KEY")

FREQ = "15min"
# Factor para llevar cada unidad a pasajeros. Una unidad desconocida no se
# adivina: esa medición se trata como faltante y se imputa (y se avisa).
UNIT_FACTORS = {
    "passengers": 1.0,
    "passenger": 1.0,
    "pax": 1.0,
    "hundreds_of_passengers": 100.0,
    "thousands_of_passengers": 1000.0,
    "kpassengers": 1000.0,
}
STREAM_COLUMNS = ["station_id", "observed_at", "demand", "quality"]


def _parse_record(record: dict[str, Any]) -> dict[str, Any]:
    medicion = record.get("measurement")
    if medicion is None:
        # Esquema 1: {"demand": 123}
        demand = record.get("demand")
        quality = "observed" if demand is not None else "missing"
    else:
        # Esquema 2: {"measurement": {"value": "123.00", "unit": ..., "quality": ...}}
        quality = str(medicion.get("quality") or "observed").lower()
        unidad = str(medicion.get("unit") or "passengers").lower()
        valor = medicion.get("value")
        factor = UNIT_FACTORS.get(unidad)
        if factor is None:
            print(f"AVISO: unidad desconocida {unidad!r}; se trata como faltante.", file=sys.stderr)
            demand, quality = None, "unknown_unit"
        elif valor is None or quality == "missing":
            demand = None
        else:
            demand = float(valor) * factor
    try:
        demand = None if demand is None else float(demand)
    except (TypeError, ValueError):
        demand = None
    if demand is not None and (not np.isfinite(demand) or demand < 0):
        demand = None
    if demand is None and quality == "observed":
        quality = "missing"
    return {
        "station_id": str(record["station_id"]),
        "observed_at": record["observed_at"],
        "demand": demand,
        "quality": quality,
    }


def parse_stream_records(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Filas crudas del stream (esquema 1 o 2) -> station_id, observed_at, demand, quality."""
    filas = [_parse_record(r) for r in records]
    if not filas:
        return pd.DataFrame(columns=STREAM_COLUMNS)
    frame = pd.DataFrame(filas)
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["station_id"] = frame["station_id"].astype("string")
    frame["demand"] = pd.to_numeric(frame["demand"], errors="coerce")
    return frame[STREAM_COLUMNS]


def fill_gaps(observations: pd.DataFrame) -> pd.DataFrame:
    """Completa la grilla de 15 min por estación e imputa la demanda faltante.

    - huecos internos: interpolación lineal en el tiempo;
    - huecos al final (lo más reciente aún no medido): último valor conocido.

    Agrega `imputed` (bool) para que el entrenamiento nunca use un valor
    imputado como objetivo ni como "real" al evaluar.
    """
    if observations.empty:
        return observations.assign(imputed=pd.Series(dtype=bool))

    partes = []
    for station_id, grupo in observations.groupby("station_id", observed=True):
        serie = grupo.set_index("observed_at")["demand"].astype(float).sort_index()
        serie = serie[~serie.index.duplicated(keep="last")]
        grilla = pd.date_range(serie.index.min(), serie.index.max(), freq=FREQ)
        serie = serie.reindex(grilla)
        faltante = serie.isna()
        rellena = serie.interpolate(method="time", limit_area="inside").ffill().bfill()
        partes.append(pd.DataFrame({
            "station_id": station_id,
            "observed_at": grilla,
            "demand": rellena.to_numpy(),
            "imputed": faltante.to_numpy(),
        }))
    salida = pd.concat(partes, ignore_index=True)
    salida["station_id"] = salida["station_id"].astype("string")
    return salida.sort_values(["station_id", "observed_at"]).reset_index(drop=True)


def fetch_stream_records() -> list[dict[str, Any]]:
    if not PULSO_API_KEY:
        # Silenciar esto costó caro una vez: train.yml corría sin la key y
        # entrenaba solo con el histórico estático sin que nadie lo notara.
        print(
            "AVISO: PULSO_API_KEY ausente. Se omite /v1/stream/observations, "
            "así que NO se verán los datos de la ventana competitiva.",
            file=sys.stderr,
        )
        return []

    headers = {"Authorization": f"Bearer {PULSO_API_KEY}"}
    cursor, filas = None, []
    while True:
        params = {"limit": 5000}
        if cursor:
            params["cursor"] = cursor
        response = request_with_retry(
            "GET", f"{PULSO_API_URL}/v1/stream/observations",
            params=params, headers=headers, timeout=45,
        )
        response.raise_for_status()
        page = response.json()
        filas.extend(page["data"] or [])
        cursor = page.get("next_cursor")
        if cursor is None:
            break
    return filas


def fetch_stream_observations() -> pd.DataFrame:
    return parse_stream_records(fetch_stream_records())


def _static_observations(intentos: int = 4) -> pd.DataFrame:
    """Histórico estático vía SDK, con reintentos: el SDK no reintenta y un
    ConnectTimeout aislado tumbó una corrida de infer.yml (2026-10-04)."""
    for intento in range(1, intentos + 1):
        try:
            with PulsoTransmiClient(timeout=45.0) as client:
                return client.observations_dataframe()
        except PulsoTransmiError as exc:
            if intento == intentos:
                raise
            print(f"Histórico estático: intento {intento} falló ({exc}); reintento.", file=sys.stderr)
            time.sleep(2.0 * intento)
    raise AssertionError("inalcanzable")


def fetch_all_observations() -> pd.DataFrame:
    """Histórico + stream, en grilla completa de 15 min, con huecos imputados.

    Columnas: station_id, observed_at, demand (nunca NaN), imputed (bool).
    """
    static = _static_observations()
    static = static[["station_id", "observed_at", "demand"]].assign(quality="observed")

    stream = fetch_stream_observations()
    combinado = static if stream.empty else pd.concat([static, stream], ignore_index=True)
    # Una medición real gana sobre una faltante del mismo instante.
    combinado = (
        combinado.assign(_tiene=combinado["demand"].notna())
        .sort_values(["station_id", "observed_at", "_tiene"])
        .drop_duplicates(subset=["station_id", "observed_at"], keep="last")
        .drop(columns="_tiene")
    )

    faltantes = int(combinado["demand"].isna().sum())
    if faltantes:
        print(f"Observaciones faltantes imputadas: {faltantes}.", file=sys.stderr)
    return fill_gaps(combinado[["station_id", "observed_at", "demand"]])
