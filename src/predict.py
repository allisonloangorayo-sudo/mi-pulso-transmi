"""Inferencia y submission contra la API real de Pulso TransMi.

Ventana competitiva activa desde 2026-09-24 (clock state=running, API
0.7.1). Esquema de /v1/submissions confirmado en producción (201 accepted
con una entrega real). Ver docs/api.md del SDK del profesor para filtros;
el contrato de submissions vive en el openapi.json del servidor.

Flujo (no reintenta a ciegas; reutiliza idempotency-key; 404 = salida en verde):
    sync -> get_current_cycle -> receipt_exists? -> load_champion ->
    predict -> validate -> submit -> save_receipt
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import httpx
import joblib
import pandas as pd
from dotenv import load_dotenv
from pulso_transmi import PulsoTransmiClient

from src.features import FEATURE_COLUMNS, add_calendar_features, build_features

load_dotenv()

PULSO_API_URL = os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io")
PULSO_API_KEY = os.getenv("PULSO_API_KEY")


class NoOpenCycle(Exception):
    """Señal de que no hay ciclo vigente (404 no_open_cycle): salida normal."""


def get_current_cycle() -> dict:
    headers = {"Authorization": f"Bearer {PULSO_API_KEY}"} if PULSO_API_KEY else {}
    response = httpx.get(f"{PULSO_API_URL}/v1/forecast-cycles/current", headers=headers, timeout=30)
    if response.status_code == 404:
        raise NoOpenCycle
    response.raise_for_status()
    return response.json()


def load_champion():
    """Carga el modelo champion: metadata desde Supabase, artefacto desde
    Supabase Storage (descargado solo si no está ya en este runner). Los
    runners de Actions empiezan en limpio en cada corrida (guía metodológica
    p.13), así que nunca asumimos que el .joblib ya existe localmente.
    """
    from src import db

    champion = db.get_champion()
    if champion is None:
        raise RuntimeError("No hay modelo champion registrado. Corre train.py y promuévelo primero.")

    local_path = Path("artifacts") / f"{champion['version']}.joblib"
    if not local_path.exists():
        db.download_model_artifact(f"{champion['version']}.joblib", local_path)
    model = joblib.load(local_path)
    return model, champion


def receipt_exists(cycle_id: str, model_version: str) -> bool:
    from src import db

    client = db.get_client()
    result = (
        client.table("predictions")
        .select("id")
        .eq("cycle_id", cycle_id)
        .eq("model_version", model_version)
        .limit(1)
        .execute()
    )
    return bool(result.data)


def fetch_all_observations() -> pd.DataFrame:
    """Histórico estático (/v1/observations) + incremental liberado
    (/v1/stream/observations). El primero se congela en el corte inicial del
    dataset; los datos de la ventana competitiva solo llegan por el stream
    (confirmado en producción: /v1/observations no incluye nada después de
    history_end aunque el ciclo pida un data_cutoff mucho más reciente).
    """
    with PulsoTransmiClient() as client:
        static_obs = client.observations_dataframe()

    headers = {"Authorization": f"Bearer {PULSO_API_KEY}"} if PULSO_API_KEY else {}
    cursor = None
    stream_frames = []
    while True:
        params = {"limit": 5000}
        if cursor:
            params["cursor"] = cursor
        response = httpx.get(f"{PULSO_API_URL}/v1/stream/observations", params=params, headers=headers, timeout=45)
        response.raise_for_status()
        page = response.json()
        if page["data"]:
            stream_frames.append(pd.DataFrame(page["data"]))
        cursor = page.get("next_cursor")
        if cursor is None:
            break

    if stream_frames:
        stream_df = pd.concat(stream_frames, ignore_index=True)
        stream_df["observed_at"] = pd.to_datetime(stream_df["observed_at"], utc=True)
        stream_df["station_id"] = stream_df["station_id"].astype("string")
        stream_df = stream_df[["station_id", "observed_at", "demand"]]
        return pd.concat([static_obs, stream_df], ignore_index=True).drop_duplicates(
            subset=["station_id", "observed_at"]
        )
    return static_obs


def build_features_as_of(data_cutoff: str, targets: list[dict]) -> pd.DataFrame:
    """Última fila de features conocida por estación (lags fijos al cutoff,
    calendario tomado del target_at futuro). Ver README para su limitación
    conocida (mismo bloque de lags para los 4 horizontes) y cómo mejorarla.
    """
    cutoff_ts = pd.Timestamp(data_cutoff)
    observations = fetch_all_observations()
    observations = observations[observations["observed_at"] <= cutoff_ts]

    features_df = build_features(observations)
    latest = features_df.sort_values("observed_at").groupby("station_id").tail(1).set_index("station_id")

    rows = []
    for target in targets:
        station_id, target_at = target["station_id"], target["target_at"]
        base = latest.loc[station_id]
        row = {col: base[col] for col in FEATURE_COLUMNS if col not in ("hour", "minute", "dayofweek", "is_weekend")}
        target_ts = pd.Timestamp(target_at)
        row.update(
            {
                "station_id": station_id,
                "target_at": target_at,
                "hour": target_ts.hour,
                "minute": target_ts.minute,
                "dayofweek": target_ts.dayofweek,
                "is_weekend": int(target_ts.dayofweek in (5, 6)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def validate_exact_targets(predictions: pd.DataFrame, targets: list[dict]) -> None:
    expected = {(t["station_id"], t["target_at"]) for t in targets}
    got = set(zip(predictions["station_id"], predictions["target_at"].astype(str)))
    if got != expected:
        raise ValueError(f"El batch no coincide con los targets solicitados. faltan={expected - got} extra={got - expected}")
    if predictions["value"].isna().any() or (predictions["value"] < 0).any():
        raise ValueError("Hay predicciones nulas o negativas.")


def stable_key(cycle_id: str, model_version: str, predictions: pd.DataFrame) -> str:
    payload = json.dumps(
        {
            "cycle_id": cycle_id,
            "model_version": model_version,
            "predictions": predictions.sort_values(["station_id", "target_at"]).to_dict(orient="records"),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def submit_predictions(
    cycle_id: str, data_cutoff: str, predictions: pd.DataFrame, idempotency_key: str, champion: dict
) -> dict:
    """POST /v1/submissions (esquema confirmado en producción: 201 accepted).

    client_run_id (body) e Idempotency-Key (header) usan la misma llave: para
    el mismo ciclo y contenido, un reintento de red reutiliza la llave en vez
    de crear una entrega nueva (regla de la guía operativa p.5).
    """
    payload = {
        "schema_version": "1.0",
        "cycle_id": cycle_id,
        "client_run_id": idempotency_key,
        "data_cutoff": data_cutoff,
        "model": {
            "version": champion["version"],
            "trained_at": champion.get("trained_at"),
            "training_data_end": champion.get("data_cutoff"),
            "git_commit": champion.get("commit_sha"),
        },
        "predictions": [
            {"station_id": row["station_id"], "target_at": row["target_at"], "value": float(row["value"])}
            for row in predictions.to_dict(orient="records")
        ],
    }
    headers = {"Authorization": f"Bearer {PULSO_API_KEY}", "Idempotency-Key": idempotency_key}
    response = httpx.post(f"{PULSO_API_URL}/v1/submissions", json=payload, headers=headers, timeout=45)
    response.raise_for_status()
    return response.json()


def save_receipt(cycle_id: str, model_version: str, predictions: pd.DataFrame, submission_id: str, idempotency_key: str) -> None:
    from src import db

    client = db.get_client()
    records = [
        {
            "cycle_id": cycle_id,
            "station_id": row["station_id"],
            "target_at": row["target_at"],
            "model_version": model_version,
            "value": row["value"],
            "submission_id": submission_id,
            "idempotency_key": idempotency_key,
        }
        for row in predictions.to_dict(orient="records")
    ]
    client.table("predictions").upsert(records, on_conflict="cycle_id,station_id,target_at,model_version").execute()


def run() -> None:
    try:
        cycle = get_current_cycle()
    except NoOpenCycle:
        print("No hay ciclo abierto (404 no_open_cycle). Fin en verde.")
        return

    model, champion = load_champion()
    if receipt_exists(cycle["cycle_id"], champion["version"]):
        print(f"Ciclo {cycle['cycle_id']} ya tiene recibo con {champion['version']}. Fin.")
        return

    features_df = build_features_as_of(cycle["data_cutoff"], cycle["targets"])
    features_df["value"] = model.predict(features_df[FEATURE_COLUMNS])
    validate_exact_targets(features_df, cycle["targets"])

    key = stable_key(cycle["cycle_id"], champion["version"], features_df)
    receipt = submit_predictions(cycle["cycle_id"], cycle["data_cutoff"], features_df, key, champion)
    save_receipt(cycle["cycle_id"], champion["version"], features_df, receipt["submission_id"], key)
    print(
        f"Entregado ciclo {cycle['cycle_id']} con {champion['version']}: "
        f"{receipt['predictions_received']}/{receipt['expected_predictions']} predicciones, "
        f"status={receipt['status']}."
    )


if __name__ == "__main__":
    run()
