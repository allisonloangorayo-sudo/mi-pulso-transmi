"""Detección de drift: accuracy acumulada vs. rolling 24h.

Si la caída (acumulada − rolling 24h) alcanza DROP_THRESHOLD_PCT, dispara
automáticamente el workflow de reentrenamiento (train.yml) vía `gh workflow
run`. Mientras el histórico sea estático, esta caída se mantiene cerca de 0
la mayor parte del tiempo (no hay drift real todavía) — el disparo solo
debería activarse cuando los datos empiecen a cambiar de verdad.
"""

from __future__ import annotations

import os
import subprocess
from datetime import timedelta

import pandas as pd
from dotenv import load_dotenv

from src import db

load_dotenv()

DROP_THRESHOLD_PCT = 5.0
MIN_SAMPLE_SIZE = 48  # al menos un ciclo (12 estaciones x 4 horizontes) evaluado


def _accuracy(frame: pd.DataFrame) -> float:
    return float(100 * max(0.0, 1 - frame["abs_error"].sum() / frame["real"].sum()))


def compute_and_store() -> dict:
    client = db.get_client()
    rows = client.table("evaluations").select("*").execute().data or []
    if len(rows) < MIN_SAMPLE_SIZE:
        print(f"Solo {len(rows)} evaluaciones (mínimo {MIN_SAMPLE_SIZE}); aún no hay señal de drift confiable.")
        return {}

    df = pd.DataFrame(rows)
    df["evaluated_at"] = pd.to_datetime(df["evaluated_at"], utc=True)
    if "abs_error" not in df.columns:
        df["abs_error"] = (df["predicted"] - df["real"]).abs()

    overall = _accuracy(df)
    recent = df[df["evaluated_at"] > df["evaluated_at"].max() - timedelta(hours=24)]
    rolling = _accuracy(recent) if len(recent) >= MIN_SAMPLE_SIZE else overall
    drop = overall - rolling
    triggered = drop >= DROP_THRESHOLD_PCT

    print(f"Muestras: {len(rows)} | accuracy acumulada: {overall:.2f} | rolling 24h: {rolling:.2f} | caída: {drop:.2f} pts")

    db.insert_drift_metric(
        {
            "sample_size": len(rows),
            "accuracy_overall": overall,
            "accuracy_rolling_24h": rolling,
            "drop_pct": drop,
            "triggered_retrain": triggered,
        }
    )

    if triggered:
        print(f"DRIFT: caída de {drop:.2f} pts >= umbral ({DROP_THRESHOLD_PCT}). Disparando reentrenamiento.")
        trigger_retrain()

    return {"overall": overall, "rolling_24h": rolling, "drop_pct": drop, "triggered": triggered}


def trigger_retrain() -> None:
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        print("No estoy corriendo dentro de GitHub Actions (falta GITHUB_REPOSITORY); omito el disparo.")
        return
    subprocess.run(["gh", "workflow", "run", "train.yml", "--repo", repo], check=True)
    print(f"Reentrenamiento disparado en {repo} (train.yml).")


def run() -> None:
    compute_and_store()


if __name__ == "__main__":
    run()
