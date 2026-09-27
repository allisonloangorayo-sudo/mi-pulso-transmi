"""Revisión del modelo y detección de drift sobre el desempeño REAL.

Cada corrida (cada hora) hace una revisión completa del champion: accuracy
acumulada, rolling 24h, referencia y accuracy por estación. Dispara el
reentrenamiento (train.yml) si ocurre cualquiera de estas cosas:

1. Caída del promedio >= DROP_THRESHOLD_PCT (3 pts) contra la referencia o
   contra el acumulado. Entre 3 y 5 pts se registra como ALERTA; desde 5 como
   CRÍTICA. Ambas reentrenan: 3 pts sobre el promedio móvil de 24 h ya son
   ~4 desviaciones estándar (3.85/√24 ≈ 0.79), no ruido.
2. Una estación cae >= STATION_DROP_PCT contra su propia referencia. El
   promedio diluye a una sola estación rota: 05100 llegó a 7.4% de accuracy
   y el promedio solo marcaba 4.5 pts de caída, por debajo del umbral
   anterior de 5, así que nunca se reentrenó.
3. El champion está viejo: su corte de datos quedó más de MAX_STALENESS_H
   horas (virtuales) detrás del dato más reciente. Medido en la ventana
   competitiva: reentrenar a diario vale +1.5 pts frente a un modelo fijo, y
   cada 6 h otros +0.35 (83.81 → 84.16).

Para no disparar cada hora mientras un reentrenamiento ya está en curso (o
cuando el candidato no logró superar al champion), hay un enfriamiento de
COOLDOWN_H horas entre disparos.
"""

from __future__ import annotations

import os
import subprocess
from datetime import timedelta

import pandas as pd
from dotenv import load_dotenv

from src import db

load_dotenv()

DROP_THRESHOLD_PCT = float(os.getenv("DRIFT_DROP_THRESHOLD_PCT", "3.0"))
CRITICAL_DROP_PCT = float(os.getenv("DRIFT_CRITICAL_DROP_PCT", "5.0"))
STATION_DROP_PCT = float(os.getenv("DRIFT_STATION_DROP_PCT", "10.0"))
MAX_STALENESS_H = float(os.getenv("DRIFT_MAX_STALENESS_H", "6"))
COOLDOWN_H = float(os.getenv("DRIFT_COOLDOWN_H", "2"))
MIN_SAMPLE_SIZE = 48  # al menos un ciclo (12 estaciones x 4 horizontes) evaluado
MIN_STATION_SAMPLE = 24  # 6 ciclos por estación antes de juzgarla sola


def _accuracy(frame: pd.DataFrame) -> float:
    """Métrica oficial: WAPE→accuracy por estación, promedio simple."""
    return float(_accuracy_by_station(frame).mean())


def _accuracy_by_station(frame: pd.DataFrame) -> pd.Series:
    error = frame.groupby("station_id")["abs_error"].sum()
    real = frame.groupby("station_id")["real"].sum()
    return (100 * (1 - error / real)).clip(lower=0)


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


def _reference_frame(df: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    """Evaluaciones que sirven de línea base para juzgar al champion vigente.

    No sirve `validation_metric`: se calcula con validación cruzada sobre
    histórico y no es comparable con la competencia (87.2 vs 83.4 para el
    mismo modelo). Se usan evaluaciones reales, y de dos fuentes:

    - las primeras REFERENCE_SAMPLE del propio champion (cómo arrancó), y
    - las últimas REFERENCE_SAMPLE ANTERIORES a su llegada (cómo rendía el
      modelo al que reemplazó).

    Se queda con la mejor de las dos. Usar solo la primera fue un error: un
    champion que nace malo (v20260926T042832Z arrancó en ~77%) se tomaba a sí
    mismo como referencia y nunca se detectaba su caída frente al anterior.
    """
    champion = db.get_champion()
    if champion is None:
        return None, None
    ordenado = df.sort_values("evaluated_at")
    suyas = ordenado[ordenado["model_version"] == champion["version"]]
    candidatas = []
    if len(suyas) >= MIN_SAMPLE_SIZE:
        candidatas.append(suyas.head(REFERENCE_SAMPLE))
        previas = ordenado[
            (ordenado["model_version"] != champion["version"])
            & (ordenado["evaluated_at"] < suyas["evaluated_at"].min())
        ]
        if len(previas) >= MIN_SAMPLE_SIZE:
            candidatas.append(previas.tail(REFERENCE_SAMPLE))
    if not candidatas:
        return None, champion["version"]
    return max(candidatas, key=_accuracy), champion["version"]


def _hours_since_last_trigger() -> float | None:
    filas = db.get_client().table("drift_metrics").select("computed_at").eq(
        "triggered_retrain", True
    ).order("computed_at", desc=True).limit(1).execute().data
    if not filas:
        return None
    ultimo = pd.Timestamp(filas[0]["computed_at"])
    return (pd.Timestamp.now(tz="UTC") - ultimo).total_seconds() / 3600


def _staleness_hours() -> float | None:
    """Horas (del reloj de los datos) entre el corte del champion y el último
    dato evaluado."""
    champion = db.get_champion()
    if champion is None or not champion.get("data_cutoff"):
        return None
    filas = db.get_client().table("evaluations").select("target_at").order(
        "target_at", desc=True
    ).limit(1).execute().data
    if not filas:
        return None
    return (pd.Timestamp(filas[0]["target_at"]) - pd.Timestamp(champion["data_cutoff"])).total_seconds() / 3600


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

    # Señal 2: caída contra la línea base (degradación sostenida que la señal
    # 1 no ve, porque el acumulado baja junto con el rolling).
    referencia_df, version = _reference_frame(df)
    referencia = _accuracy(referencia_df) if referencia_df is not None else None
    caida_referencia = (referencia - rolling) if referencia is not None else 0.0

    caida = max(caida_historica, caida_referencia)

    # Señal 3: una estación rota, que el promedio diluye.
    por_estacion = _accuracy_by_station(recientes)
    ref_estacion = _accuracy_by_station(referencia_df if referencia_df is not None else df)
    muestras = recientes.groupby("station_id").size()
    caida_estacion = (ref_estacion - por_estacion).where(muestras >= MIN_STATION_SAMPLE).dropna()
    rotas = caida_estacion[caida_estacion >= STATION_DROP_PCT].sort_values(ascending=False)

    # Señal 4: champion viejo.
    antiguedad = _staleness_hours()

    motivos = []
    if caida >= DROP_THRESHOLD_PCT:
        nivel = "CRÍTICA" if caida >= CRITICAL_DROP_PCT else "ALERTA"
        motivos.append(f"caída {nivel} de {caida:.2f} pts (umbral {DROP_THRESHOLD_PCT})")
    if not rotas.empty:
        motivos.append("estaciones degradadas: " + ", ".join(f"{s} −{v:.1f}" for s, v in rotas.items()))
    if antiguedad is not None and antiguedad > MAX_STALENESS_H:
        motivos.append(f"champion con {antiguedad:.0f} h sin datos nuevos (máx {MAX_STALENESS_H:.0f})")

    desde_ultimo = _hours_since_last_trigger()
    en_enfriamiento = desde_ultimo is not None and desde_ultimo < COOLDOWN_H
    triggered = bool(motivos) and not en_enfriamiento

    print(f"=== Revisión del champion {version} ===")
    print(f"Origen: {origen} | muestras: {len(df)}")
    print(f"  acumulada {overall:.2f} | rolling 24h {rolling:.2f} | caída {caida_historica:.2f} pts")
    if referencia is not None:
        print(f"  referencia {referencia:.2f} | caída contra ella {caida_referencia:.2f} pts")
    else:
        print("  sin línea base del champion todavía (necesita más evaluaciones suyas)")
    if antiguedad is not None:
        print(f"  antigüedad del champion: {antiguedad:.1f} h")
    print("  accuracy 24h por estación (vs referencia):")
    for estacion, valor in por_estacion.sort_values().items():
        delta = caida_estacion.get(estacion)
        extra = f"  (−{delta:.1f})" if delta is not None and delta > 0 else ""
        print(f"    {estacion}: {valor:6.2f}{extra}")

    db.insert_drift_metric({
        "sample_size": len(df),
        "accuracy_overall": overall,
        "accuracy_rolling_24h": rolling,
        "drop_pct": caida,
        "triggered_retrain": triggered,
    })

    if motivos and en_enfriamiento:
        print(f"Motivos para reentrenar ({'; '.join(motivos)}), pero hubo un disparo hace "
              f"{desde_ultimo:.1f} h (enfriamiento {COOLDOWN_H} h). No se dispara otra vez.")
    elif triggered:
        print(f"REENTRENAMIENTO: {'; '.join(motivos)}.")
        trigger_retrain()
    else:
        print("Sin motivos para reentrenar.")

    return {
        "overall": overall, "rolling_24h": rolling, "reference": referencia,
        "drop_pct": caida, "triggered": triggered, "source": origen,
        "reasons": motivos, "stations_degraded": list(rotas.index),
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
