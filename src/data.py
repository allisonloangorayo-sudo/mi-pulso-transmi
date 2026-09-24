"""Carga de observaciones: histórico estático + stream incremental.

`/v1/observations` se congela en `history_end` del dataset inicial; todo lo
que libera la ventana competitiva llega solo por `/v1/stream/observations`.
Entrenamiento e inferencia deben ver exactamente las mismas fuentes, por eso
esta función vive aquí y no duplicada en cada módulo.
"""

from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv
from pulso_transmi import PulsoTransmiClient

from src.http_utils import request_with_retry

load_dotenv()

PULSO_API_URL = os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io")
PULSO_API_KEY = os.getenv("PULSO_API_KEY")


def fetch_stream_observations() -> pd.DataFrame:
    if not PULSO_API_KEY:
        return pd.DataFrame(columns=["station_id", "observed_at", "demand"])

    headers = {"Authorization": f"Bearer {PULSO_API_KEY}"}
    cursor, frames = None, []
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
        if page["data"]:
            frames.append(pd.DataFrame(page["data"]))
        cursor = page.get("next_cursor")
        if cursor is None:
            break

    if not frames:
        return pd.DataFrame(columns=["station_id", "observed_at", "demand"])

    stream = pd.concat(frames, ignore_index=True)
    stream["observed_at"] = pd.to_datetime(stream["observed_at"], utc=True)
    stream["station_id"] = stream["station_id"].astype("string")
    return stream[["station_id", "observed_at", "demand"]]


def fetch_all_observations() -> pd.DataFrame:
    with PulsoTransmiClient() as client:
        static = client.observations_dataframe()

    stream = fetch_stream_observations()
    if stream.empty:
        combinado = static
    else:
        combinado = pd.concat([static, stream], ignore_index=True)

    return (
        combinado.drop_duplicates(subset=["station_id", "observed_at"])
        .sort_values(["station_id", "observed_at"])
        .reset_index(drop=True)
    )
