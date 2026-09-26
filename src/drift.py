"""Detección de drift: accuracy acumulada vs. rolling 24h.

Si la caída (acumulada − rolling 24h) alcanza DROP_THRESHOLD_PCT, dispara
automáticamente el reentrenamiento (train.yml) vía `gh workflow run`.

Por qué 5 puntos, y por qué sobre el promedio móvil y no sobre un lote
suelto. Medido sobre 174 lotes evaluados de este proyecto:

    accuracy por lote: media 88.90, desviación 3.85 (p5 = 85.19, min = 66.21)

- Sobre **un lote suelto**, 5 puntos son apenas 1.3 desviaciones: ocurre por
  puro azar con frecuencia, así que ahí el umbral produciría falsas alarmas.
- Sobre el **promedio móvil de 24 h** (que es lo que se compara aquí), el
  error estándar cae a 3.85/√24 ≈ 0.79, así que 5 puntos son ~6 desviaciones:
  prácticamente imposible por azar.

Es decir: el umbral es conservador por diseño. Prefiere no reentrenar de más
antes que reaccionar a ruido. Si se quisiera detectar antes, 2.5 puntos (≈3σ
del promedio móvil) sigue siendo estadísticamente defendible.
"""

from __future__ import annotations

import os
import subprocess
from datetime import timedelta

import pandas as pd
from dotenv import load_dotenv

from src import db

load_dotenv()

DROP_THRESHOLD_PCT = float(os.getenv("DRIFT_DROP_THRESHOLD_PCT", "5.0"))
MIN_SAMPLE_SIZE = 48  # al menos un ciclo (12 estaciones x 4 horizontes) evaluado


def _accuracy(frame: pd.DataFrame) -> float:
    return float(100 * max(0.0, 1 - frame["abs_error"].sum() / frame["real"].sum()))


def compute_and_store() -> dict:
    rows = db.fetch_all_rows("evaluations", "station_id,predicted,real,abs_error,evaluated_at")
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
