"""Revisión del modelo y detección de drift sobre el desempeño REAL.

Cada corrida (cada hora) hace una revisión completa del champion: accuracy
acumulada, rolling 24h, referencia y accuracy por estación. Dispara el
reentrenamiento (train.yml) si ocurre cualquiera de estas cosas:

1. Caída del promedio >= DROP_THRESHOLD_PCT (2 pts) contra la referencia, el
   acumulado o los 6 ciclos anteriores (ver 6). Entre 2 y 5 pts se registra
   como ALERTA; desde 5 como CRÍTICA. Ambas reentrenan.
2. Una estación cae >= STATION_DROP_PCT contra su propia referencia. El
   promedio diluye a una sola estación rota: 05100 llegó a 7.4% de accuracy
   y el promedio solo marcaba 4.5 pts de caída, por debajo del umbral
   anterior de 5, así que nunca se reentrenó.
3. Piso absoluto: la accuracy de los últimos 6 ciclos (la misma ventana de
   la tabla "últimos 6 ciclos" del portal) queda por debajo de
   ACCURACY_FLOOR_PCT (80%), aunque no haya una caída brusca.
4. El champion está viejo: su corte de datos quedó más de MAX_STALENESS_H
   horas (virtuales) detrás del dato más reciente. Medido en la ventana
   competitiva: reentrenar a diario vale +1.5 pts frente a un modelo fijo, y
   cada 6 h otros +0.35 (83.81 → 84.16). Desde que el generador cambia la
   forma de la serie (~cada 2 días virtuales) se baja a 3 h: el GBM pasa de
   ~60% a ~90% en cuanto ve unas horas del régimen nuevo
   (experiments/07_regimen_adaptativo.py).

5. Actualización horaria: si el último entrenamiento (de cualquier estado)
   tiene más de RETRAIN_EVERY_H horas REALES, se reentrena con los datos
   nuevos aunque no haya caída. Así el modelo se actualiza cada hora.
6. Verificación de mejora: accuracy real de los últimos 6 ciclos contra los
   6 anteriores, y por versión de modelo. Se imprime MEJORA / ESTABLE /
   EMPEORA y se guarda en pipeline_state["accuracy_trend"] para el dashboard;
   si empeora >= DROP_THRESHOLD_PCT, reentrena.

Para no disparar dos veces mientras un reentrenamiento ya está en curso (o
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

DROP_THRESHOLD_PCT = float(os.getenv("DRIFT_DROP_THRESHOLD_PCT", "2.0"))
CRITICAL_DROP_PCT = float(os.getenv("DRIFT_CRITICAL_DROP_PCT", "5.0"))
ACCURACY_FLOOR_PCT = float(os.getenv("DRIFT_ACCURACY_FLOOR_PCT", "80.0"))
FLOOR_WINDOW_CYCLES = 6  # igual que la tabla "últimos 6 ciclos" del portal
STATION_DROP_PCT = float(os.getenv("DRIFT_STATION_DROP_PCT", "10.0"))
MAX_STALENESS_H = float(os.getenv("DRIFT_MAX_STALENESS_H", "3"))
COOLDOWN_H = float(os.getenv("DRIFT_COOLDOWN_H", "0.75"))
RETRAIN_EVERY_H = float(os.getenv("DRIFT_RETRAIN_EVERY_H", "1"))
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
        "evaluations", "station_id,predicted,real,abs_error,evaluated_at,model_version,source,cycle_id"
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


def _hours_since_last_training() -> float | None:
    """Horas reales desde el último modelo entrenado (promovido o no)."""
    filas = db.get_client().table("model_versions").select("trained_at").order(
        "trained_at", desc=True
    ).limit(1).execute().data
    if not filas or not filas[0].get("trained_at"):
        return None
    ultimo = pd.Timestamp(filas[0]["trained_at"])
    if ultimo.tzinfo is None:
        ultimo = ultimo.tz_localize("UTC")
    return (pd.Timestamp.now(tz="UTC") - ultimo).total_seconds() / 3600


def accuracy_trend(df: pd.DataFrame) -> dict | None:
    """¿Está mejorando? Últimos 6 ciclos reales contra los 6 anteriores.

    Usa ciclos oficiales (cycle_id), igual que la tabla "últimos 6 ciclos" del
    portal. También resume la accuracy real de cada versión de modelo.
    """
    if "cycle_id" not in df.columns or df["cycle_id"].isna().all():
        return None
    reales = df.dropna(subset=["cycle_id"])
    ciclos = sorted(reales["cycle_id"].unique())
    if len(ciclos) < 2:
        return None
    ultimos = ciclos[-FLOOR_WINDOW_CYCLES:]
    previos = ciclos[-2 * FLOOR_WINDOW_CYCLES:-FLOOR_WINDOW_CYCLES]
    actual = _accuracy(reales[reales["cycle_id"].isin(ultimos)])
    anterior = _accuracy(reales[reales["cycle_id"].isin(previos)]) if previos else None
    delta = None if anterior is None else actual - anterior
    if delta is None:
        veredicto = "SIN_REFERENCIA"
    elif delta >= 0.5:
        veredicto = "MEJORA"
    elif delta <= -DROP_THRESHOLD_PCT:
        veredicto = "EMPEORA"
    else:
        veredicto = "ESTABLE"

    por_version = []
    for version, grupo in reales.groupby("model_version"):
        por_version.append({
            "model_version": version,
            "cycles": int(grupo["cycle_id"].nunique()),
            "accuracy": round(_accuracy(grupo), 2),
            "first_evaluated_at": grupo["evaluated_at"].min().isoformat(),
        })
    por_version.sort(key=lambda r: r["first_evaluated_at"])
    return {
        "last_cycles": ultimos,
        "accuracy_last_6": round(actual, 2),
        "accuracy_prev_6": None if anterior is None else round(anterior, 2),
        "delta": None if delta is None else round(delta, 2),
        "verdict": veredicto,
        "above_floor": actual >= ACCURACY_FLOOR_PCT,
        "by_model_version": por_version[-10:],
    }


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

    # Señal 4: piso absoluto sobre los últimos 6 ciclos.
    ultimos_ciclos = None
    if "cycle_id" in df.columns and df["cycle_id"].notna().any():
        ciclos = sorted(df["cycle_id"].dropna().unique())[-FLOOR_WINDOW_CYCLES:]
        ventana = df[df["cycle_id"].isin(ciclos)]
        if len(ventana) >= MIN_SAMPLE_SIZE:
            ultimos_ciclos = _accuracy(ventana)

    # Señal 5: champion viejo.
    antiguedad = _staleness_hours()

    # Señal 6: actualización horaria con datos nuevos.
    desde_entrenamiento = _hours_since_last_training()

    # Señal 7: verificación de mejora (últimos 6 ciclos vs los 6 anteriores).
    tendencia = accuracy_trend(df) if origen == "real" else None

    motivos = []
    if caida >= DROP_THRESHOLD_PCT:
        nivel = "CRÍTICA" if caida >= CRITICAL_DROP_PCT else "ALERTA"
        motivos.append(f"caída {nivel} de {caida:.2f} pts (umbral {DROP_THRESHOLD_PCT})")
    if not rotas.empty:
        motivos.append("estaciones degradadas: " + ", ".join(f"{s} −{v:.1f}" for s, v in rotas.items()))
    if ultimos_ciclos is not None and ultimos_ciclos < ACCURACY_FLOOR_PCT:
        motivos.append(f"accuracy de los últimos {FLOOR_WINDOW_CYCLES} ciclos {ultimos_ciclos:.2f} "
                       f"< piso de {ACCURACY_FLOOR_PCT:.0f}")
    if antiguedad is not None and antiguedad > MAX_STALENESS_H:
        motivos.append(f"champion con {antiguedad:.0f} h sin datos nuevos (máx {MAX_STALENESS_H:.0f})")
    if desde_entrenamiento is None or desde_entrenamiento >= RETRAIN_EVERY_H:
        horas = "nunca" if desde_entrenamiento is None else f"hace {desde_entrenamiento:.1f} h"
        motivos.append(f"actualización horaria (último entrenamiento {horas})")
    if tendencia and tendencia["verdict"] == "EMPEORA":
        motivos.append(f"los últimos 6 ciclos empeoraron {tendencia['delta']:+.2f} pts "
                       f"(umbral −{DROP_THRESHOLD_PCT:.0f})")

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
    if ultimos_ciclos is not None:
        print(f"  últimos {FLOOR_WINDOW_CYCLES} ciclos {ultimos_ciclos:.2f} (piso {ACCURACY_FLOOR_PCT:.0f})")
    if antiguedad is not None:
        print(f"  antigüedad del champion: {antiguedad:.1f} h")
    if desde_entrenamiento is not None:
        print(f"  último entrenamiento: hace {desde_entrenamiento:.1f} h reales")
    if tendencia:
        previo = tendencia["accuracy_prev_6"]
        print(f"  VERIFICACIÓN: últimos 6 ciclos {tendencia['accuracy_last_6']:.2f}"
              + (f" vs 6 anteriores {previo:.2f} ({tendencia['delta']:+.2f}) → {tendencia['verdict']}"
                 if previo is not None else " (sin ciclos anteriores para comparar)")
              + f" | {'sobre' if tendencia['above_floor'] else 'BAJO'} el piso de {ACCURACY_FLOOR_PCT:.0f}%")
        for fila in tendencia["by_model_version"][-5:]:
            print(f"    {fila['model_version']}: {fila['accuracy']:.2f} en {fila['cycles']} ciclos reales")
        try:
            db.set_state("accuracy_trend", tendencia)
        except Exception as exc:  # noqa: BLE001 - la bitácora no debe tumbar la revisión
            print(f"Aviso: no se pudo guardar accuracy_trend ({exc}).")
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
        "last_cycles": ultimos_ciclos, "reasons": motivos, "stations_degraded": list(rotas.index),
        "trend": tendencia,
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
