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


REFERENCE_SAMPLE = 480  # 10 ciclos: la "línea base" de un champion recién promovido


def _load_evaluations() -> tuple[pd.DataFrame, str]:
    """Prefiere las evaluaciones reales; cae a simuladas solo si no alcanzan.

    Las simuladas replican historia que el modelo ya vio y dan una lectura
    optimista (medido: 89.5% simulado vs 83.4% real). Usarlas para decidir
    reentrenamientos escondería la degradación de verdad.
    """
    filas = db.fetch_all_rows(
        "evaluations", "station_id,predicted,real,abs_error,evaluated_at,model_version,source"
    )
    df = pd.DataFrame(filas)
    if df.empty:
        return df, "ninguna"

    df["evaluated_at"] = pd.to_datetime(df["evaluated_at"], utc=True)
    if "abs_error" not in df.columns:
        df["abs_error"] = (df["predicted"] - df["real"]).abs()

    reales = df[df["source"] == "real"]
    if len(reales) >= MIN_SAMPLE_SIZE:
        return reales, "real"
    print(f"Solo {len(reales)} evaluaciones reales; se usan las simuladas (lectura optimista).")
    return df[df["source"] == "simulado"], "simulado"


def _reference_accuracy(df: pd.DataFrame) -> tuple[float | None, str | None]:
    """Línea base del champion vigente: sus primeras evaluaciones reales.

    No sirve comparar contra `validation_metric` del modelo, porque esa se
    calculó con validación cruzada sobre histórico y no es comparable con el
    desempeño en competencia (87.2 vs 83.4 para el mismo modelo). La
    referencia honesta es cómo rindió ESE MISMO modelo cuando empezó a
    operar.
    """
    champion = db.get_champion()
    if champion is None:
        return None, None
    suyas = df[df["model_version"] == champion["version"]].sort_values("evaluated_at")
    if len(suyas) < MIN_SAMPLE_SIZE:
        return None, champion["version"]
    return _accuracy(suyas.head(REFERENCE_SAMPLE)), champion["version"]


def compute_and_store() -> dict:
    df, origen = _load_evaluations()
    if len(df) < MIN_SAMPLE_SIZE:
        print(f"Solo {len(df)} evaluaciones (mínimo {MIN_SAMPLE_SIZE}); aún no hay señal confiable.")
        return {}

    overall = _accuracy(df)
    recientes = df[df["evaluated_at"] > df["evaluated_at"].max() - timedelta(hours=24)]
    rolling = _accuracy(recientes) if len(recientes) >= MIN_SAMPLE_SIZE else overall

    # Señal 1: caída súbita (lo reciente contra todo el historial).
    caida_historica = overall - rolling

    # Señal 2: caída contra la línea base del champion (degradación sostenida
    # que la señal 1 no ve, porque el acumulado baja junto con el rolling).
    referencia, version = _reference_accuracy(df)
    caida_referencia = (referencia - rolling) if referencia is not None else 0.0

    caida = max(caida_historica, caida_referencia)
    triggered = caida >= DROP_THRESHOLD_PCT

    print(f"Origen: {origen} | muestras: {len(df)}")
    print(f"  acumulada {overall:.2f} | rolling 24h {rolling:.2f} | caída {caida_historica:.2f} pts")
    if referencia is not None:
        print(f"  referencia de {version}: {referencia:.2f} | caída contra ella {caida_referencia:.2f} pts")
    else:
        print("  sin línea base del champion todavía (necesita más evaluaciones suyas)")

    db.insert_drift_metric({
        "sample_size": len(df),
        "accuracy_overall": overall,
        "accuracy_rolling_24h": rolling,
        "drop_pct": caida,
        "triggered_retrain": triggered,
    })

    if triggered:
        print(f"DRIFT: caída de {caida:.2f} pts >= umbral ({DROP_THRESHOLD_PCT}). Disparando reentrenamiento.")
        trigger_retrain()

    return {
        "overall": overall, "rolling_24h": rolling, "reference": referencia,
        "drop_pct": caida, "triggered": triggered, "source": origen,
    }


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
