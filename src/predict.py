"""Inferencia y submission, siguiendo el contrato de la guía operativa v2.0.

IMPORTANTE — estado actual: la API pública (SDK 0.2.0) todavía NO expone
`/v1/forecast-cycles/current` ni el endpoint de submissions; por eso este
módulo es un scaffold fiel al flujo documentado (páginas 5-7 de la guía
operativa), listo para activarse cuando el profesor habilite la ventana
competitiva. Las partes bloqueadas por falta de endpoint público están
marcadas con TODO y no deben ejecutarse en producción todavía.

Flujo (no reintenta a ciegas; reutiliza idempotency-key; 404 = salida en verde):
    sync -> get_current_cycle -> receipt_exists? -> load_champion ->
    predict -> validate -> submit -> save_receipt
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import timedelta

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
    """Carga el modelo champion desde Supabase (model_versions.status='champion').

    TODO: cuando se implemente el paso 4 (promoción), el artefacto debe
    descargarse desde Supabase Storage o GitHub Release en vez de asumir que
    ya está presente en el filesystem del runner (los runners de Actions
    empiezan en limpio, ver guía metodológica p.13).
    """
    from src import db

    client = db.get_client()
    result = (
        client.table("model_versions")
        .select("*")
        .eq("status", "champion")
        .order("trained_at", desc=True)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise RuntimeError("No hay modelo champion registrado. Corre train.py y promuévelo primero.")
    champion = rows[0]
    model = joblib.load(champion["artifact_path"])
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


def build_features_as_of(data_cutoff: str, targets: list[dict]) -> pd.DataFrame:
    """Última fila de features conocida por estación (lags fijos al cutoff,
    calendario tomado del target_at futuro). Ver README para su limitación
    conocida (mismo bloque de lags para los 4 horizontes) y cómo mejorarla.
    """
    cutoff_ts = pd.Timestamp(data_cutoff)
    with PulsoTransmiClient() as client:
        observations = client.observations_dataframe(end=cutoff_ts.isoformat())

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


def submit_predictions(cycle_id: str, predictions: pd.DataFrame, idempotency_key: str, model_metadata: dict) -> dict:
    """TODO: ruta y payload exactos se publican en el contrato técnico oficial
    (repositorio uexternadojz/pulso-transmi) cuando se habiliten submissions.
    Placeholder de la forma documentada en la guía operativa (p.7-8).
    """
    raise NotImplementedError(
        "El endpoint de submissions aún no está publicado. Actualiza esta función "
        "con la ruta y el payload del contrato técnico cuando se habilite."
    )


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
    receipt = submit_predictions(cycle["cycle_id"], features_df, key, champion)
    save_receipt(cycle["cycle_id"], champion["version"], features_df, receipt["submission_id"], key)
    print(f"Entregado ciclo {cycle['cycle_id']} con {champion['version']}.")


if __name__ == "__main__":
    run()
