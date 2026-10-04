"""Inferencia y submission contra la API real de Pulso TransMi.

Ventana competitiva activa desde 2026-09-24 (clock state=running, API
0.7.1). Esquema de /v1/submissions confirmado en producción.

Flujo (no reintenta a ciegas; reutiliza idempotency-key; 404 = salida en verde):
    get_current_cycle -> receipt_exists? -> load_champion ->
    predict -> doble verificación -> validate -> submit -> save_receipt

El modelo es un bundle (ver src.model: global normalizado o, en la
generación anterior, uno por estación). Las features se construyen con la
misma función que se usó al entrenar (src.features), de modo que no puede
reaparecer el train/serve skew.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src import model
from src.data import fetch_all_observations
from src.features import build_serving_design
from src.http_utils import request_with_retry

load_dotenv()

PULSO_API_URL = os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io")
PULSO_API_KEY = os.getenv("PULSO_API_KEY")


class NoOpenCycle(Exception):
    """Señal de que no hay ciclo vigente (404 no_open_cycle): salida normal."""


def get_current_cycle() -> dict:
    headers = {"Authorization": f"Bearer {PULSO_API_KEY}"} if PULSO_API_KEY else {}
    response = request_with_retry(
        "GET", f"{PULSO_API_URL}/v1/forecast-cycles/current", headers=headers, timeout=30
    )
    if response.status_code == 404:
        raise NoOpenCycle
    response.raise_for_status()
    return response.json()


def load_champion() -> tuple[dict, dict]:
    """Bundle del champion: metadata desde Supabase, artefacto desde Storage.

    Los runners de Actions empiezan en limpio, así que nunca se asume que el
    .joblib ya está en disco (ver el paso de caché en infer.yml).
    """
    from src import db

    champion = db.get_champion()
    if champion is None:
        raise RuntimeError("No hay modelo champion registrado. Corre train.py primero.")

    local_path = Path("artifacts") / f"{champion['version']}.joblib"
    if not local_path.exists():
        db.download_model_artifact(f"{champion['version']}.joblib", local_path)
    return joblib.load(local_path), champion


def receipt_exists(cycle_id: str, model_version: str) -> bool:
    from src import db

    result = (
        db.get_client()
        .table("predictions")
        .select("id")
        .eq("cycle_id", cycle_id)
        .eq("model_version", model_version)
        .limit(1)
        .execute()
    )
    return bool(result.data)


def targets_to_frame(targets: list[dict]) -> pd.DataFrame:
    """Targets del ciclo -> DataFrame con el horizonte en pasos de 15 min."""
    frame = pd.DataFrame(targets)
    frame["station_id"] = frame["station_id"].astype("string")
    frame["target_at"] = pd.to_datetime(frame["target_at"], utc=True)
    frame["horizon"] = (frame["horizon_minutes"] // 15).astype(int)
    return frame[["station_id", "target_at", "horizon"]]


def predict_bundle(bundle: dict, design: pd.DataFrame) -> np.ndarray:
    return model.predict(bundle, design)


def double_check_predictions(bundle: dict, design: pd.DataFrame) -> np.ndarray:
    """Dos inferencias sobre el mismo batch antes de enviar nada.

    `equal_nan=True` porque un NaN no es no-determinismo: así fue como el
    cambio de esquema del stream se disfrazó de "doble verificación falló" y
    abortó todos los envíos. Los NaN se resuelven después con un fallback.
    """
    primera = predict_bundle(bundle, design)
    segunda = predict_bundle(bundle, design)
    if not np.allclose(primera, segunda, equal_nan=True):
        raise RuntimeError(
            "Doble verificación falló: dos inferencias sobre el mismo batch "
            "dieron resultados distintos. Se aborta el envío."
        )
    return primera


def forecast_with_check(bundle: dict, observations: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Dos inferencias sobre el mismo batch y, después, cero huecos.

    `equal_nan=True` porque un NaN no es no-determinismo: así fue como el
    cambio de esquema del stream se disfrazó de "doble verificación falló" y
    abortó todos los envíos. Un target sin valor puntúa 0; uno aproximado no,
    así que los NaN se reemplazan por el último dato conocido de la estación.
    """
    primera = model.forecast_targets(bundle, observations, targets)
    segunda = model.forecast_targets(bundle, observations, targets)
    if not np.allclose(primera["value"], segunda["value"], equal_nan=True):
        raise RuntimeError(
            "Doble verificación falló: dos inferencias sobre el mismo batch "
            "dieron resultados distintos. Se aborta el envío."
        )
    valores = primera["value"].to_numpy(dtype=float)
    malos = ~np.isfinite(valores)
    if malos.any():
        print(f"AVISO: {int(malos.sum())} predicciones no finitas; se usa el último dato conocido.")
        valores = np.where(malos, primera["fallback"].to_numpy(dtype=float), valores)
        malos = ~np.isfinite(valores)
        if malos.any():
            valores[malos] = float(np.nanmedian(valores)) if np.isfinite(valores).any() else 0.0
    return primera.assign(value=np.clip(valores, 0, None))


def build_batch(cycle: dict, bundle: dict) -> pd.DataFrame:
    targets = targets_to_frame(cycle["targets"])
    observations = fetch_all_observations()
    observations = observations[observations["observed_at"] <= pd.Timestamp(cycle["data_cutoff"])]
    batch = forecast_with_check(bundle, observations, targets)
    return batch[["station_id", "target_at", "horizon", "value"]]


def validate_exact_targets(predictions: pd.DataFrame, targets: list[dict]) -> None:
    esperados = {(t["station_id"], pd.Timestamp(t["target_at"])) for t in targets}
    obtenidos = set(zip(predictions["station_id"], predictions["target_at"]))
    if obtenidos != esperados:
        raise ValueError(
            f"El batch no coincide con los targets. faltan={esperados - obtenidos} "
            f"extra={obtenidos - esperados}"
        )
    if predictions["value"].isna().any() or (predictions["value"] < 0).any():
        raise ValueError("Hay predicciones nulas o negativas.")


def stable_key(cycle_id: str, model_version: str, predictions: pd.DataFrame) -> str:
    payload = json.dumps(
        {
            "cycle_id": cycle_id,
            "model_version": model_version,
            "predictions": predictions.sort_values(["station_id", "target_at"])
            .to_dict(orient="records"),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def submit_predictions(
    cycle_id: str, data_cutoff: str, predictions: pd.DataFrame, idempotency_key: str, champion: dict
) -> dict:
    """POST /v1/submissions. client_run_id e Idempotency-Key comparten llave:
    un reintento tras falla transitoria reutiliza la entrega en vez de crear otra.
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
            {
                "station_id": fila["station_id"],
                "target_at": pd.Timestamp(fila["target_at"]).isoformat().replace("+00:00", "Z"),
                "value": float(fila["value"]),
            }
            for fila in predictions.to_dict(orient="records")
        ],
    }
    headers = {"Authorization": f"Bearer {PULSO_API_KEY}", "Idempotency-Key": idempotency_key}
    response = request_with_retry(
        "POST", f"{PULSO_API_URL}/v1/submissions", json=payload, headers=headers, timeout=45
    )
    response.raise_for_status()
    return response.json()


def save_receipt(
    cycle_id: str, model_version: str, predictions: pd.DataFrame,
    submission_id: str, idempotency_key: str,
) -> None:
    from src import db

    records = [
        {
            "cycle_id": cycle_id,
            "station_id": fila["station_id"],
            "target_at": pd.Timestamp(fila["target_at"]).isoformat(),
            "model_version": model_version,
            "value": float(fila["value"]),
            "submission_id": submission_id,
            "idempotency_key": idempotency_key,
        }
        for fila in predictions.to_dict(orient="records")
    ]
    db.upsert_in_chunks(
        "predictions", records, on_conflict="cycle_id,station_id,target_at,model_version"
    )


def run() -> None:
    try:
        cycle = get_current_cycle()
    except NoOpenCycle:
        print("No hay ciclo abierto (404 no_open_cycle). Fin en verde.")
        return {"mode": "sin_ciclo", "status": "ok"}

    bundle, champion = load_champion()
    if receipt_exists(cycle["cycle_id"], champion["version"]):
        print(f"Ciclo {cycle['cycle_id']} ya tiene recibo con {champion['version']}. Fin.")
        return {
            "mode": "real", "status": "skipped", "cycle_id": cycle["cycle_id"],
            "data_cutoff": cycle["data_cutoff"], "model_version": champion["version"],
        }

    batch = build_batch(cycle, bundle)
    validate_exact_targets(batch, cycle["targets"])

    key = stable_key(cycle["cycle_id"], champion["version"], batch)
    recibo = submit_predictions(cycle["cycle_id"], cycle["data_cutoff"], batch, key, champion)
    save_receipt(cycle["cycle_id"], champion["version"], batch, recibo["submission_id"], key)
    print(
        f"Entregado ciclo {cycle['cycle_id']} con {champion['version']}: "
        f"{recibo['predictions_received']}/{recibo['expected_predictions']} predicciones, "
        f"status={recibo['status']}."
    )
    return {
        "mode": "real", "status": "ok", "cycle_id": cycle["cycle_id"],
        "data_cutoff": cycle["data_cutoff"], "model_version": champion["version"],
        "predictions_count": int(recibo["predictions_received"]),
        "submission_id": recibo["submission_id"],
    }


if __name__ == "__main__":
    run()
