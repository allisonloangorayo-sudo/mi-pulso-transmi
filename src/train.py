"""Entrenamiento: ensamble adaptativo al régimen (ver src.adaptive).

Uso:
    python -m src.train

Historia de decisiones (métrica oficial, 4 horizontes):

    producción inicial (1 modelo, lags congelados)          79.24%
    un modelo por estación, boosting MAE                    86.69%  (histórico estable)
    global normalizado por nivel (experiments/06)           84.16%  (ciclos 11-15 sep)

Desde el 18-sep el generador cambia la *forma* de la serie (ciclo diario ->
4 h -> ~8 h) y el modelo por nivel se hundió a 44-58% en esos cambios. La
receta actual detecta el periodo dominante en cada corte y combina un GBM con
pronosticadores simples según su error reciente. Ver
experiments/07_regimen_adaptativo.py.

Promoción: validación cruzada de 2 pliegues temporales cortos (los regímenes
duran ~2 días); el candidato debe superar a los baselines en AMBOS, y luego
gana el duelo contra el champion sobre la misma ventana (duel_against_champion).
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src import adaptive, model
from src.data import fetch_all_observations
from src.features import HORIZONS, build_global_training_design

load_dotenv()

ARTIFACTS_DIR = Path("artifacts")
VALIDATION_DAYS = 2
N_FOLDS = 2
# Historia previa para predecir una ventana: el experto "nivel" usa rezagos de
# 2 semanas (tweek2) y sin ellos cae al último dato conocido.
CONTEXT_DAYS = 16
FEATURES = adaptive.FEATURES


def wape_accuracy_by_station(frame: pd.DataFrame, prediction_col: str) -> pd.Series:
    error = (frame["demand"] - frame[prediction_col]).abs()
    wape = error.groupby(frame["station_id"]).sum() / frame["demand"].groupby(frame["station_id"]).sum()
    return (100 * (1 - wape)).clip(lower=0)


def official_accuracy(frame: pd.DataFrame, prediction_col: str = "prediction") -> float:
    return float(wape_accuracy_by_station(frame, prediction_col).mean())


def temporal_split(df: pd.DataFrame, validation_days: int = VALIDATION_DAYS):
    cutoff = df["observed_at"].max() - pd.Timedelta(days=validation_days)
    return df[df["observed_at"] <= cutoff].copy(), df[df["observed_at"] > cutoff].copy()


def git_commit_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def predict_window(bundle: dict, observations: pd.DataFrame, desde, hasta) -> pd.DataFrame:
    """Predicciones de `bundle` para todo target real en (desde, hasta].

    Entiende cualquier tipo de bundle, así el duelo puede enfrentar al
    candidato con un champion de otra generación sobre las mismas filas.
    Cada fila usa solo datos hasta su propio corte (target - horizonte).
    """
    if bundle.get("kind") == "adaptive_ensemble":
        inicio = desde - pd.Timedelta(days=CONTEXT_DAYS)
        contexto = observations[(observations["observed_at"] > inicio)
                                & (observations["observed_at"] <= hasta)]
        design = adaptive.build_design(contexto)
        design = design[design["observed_at"] > desde - pd.Timedelta(days=1)].reset_index(drop=True)
        design["prediction"] = adaptive.predict(bundle, design)
    else:
        design = build_global_training_design(observations).reset_index(drop=True)
        design = design[(design["observed_at"] > desde) & (design["observed_at"] <= hasta)]
        design = design.reset_index(drop=True)
        design["prediction"] = model.predict(bundle, design)

    mascara = (design["observed_at"] > desde) & (design["observed_at"] <= hasta)
    if "imputed" in design:
        mascara &= ~design["imputed"].astype(bool)
    salida = design[mascara].dropna(subset=["demand", "prediction"])
    return salida[["station_id", "observed_at", "horizon", "demand", "prediction"]].reset_index(drop=True)


def evaluate_baselines(observations: pd.DataFrame, desde, hasta) -> dict[str, float]:
    """Baselines sobre el mismo conjunto: último dato conocido y repetir hace 24 h."""
    design = adaptive.build_design(
        observations[(observations["observed_at"] > desde - pd.Timedelta(days=3))
                     & (observations["observed_at"] <= hasta)]
    )
    design = design[(design["observed_at"] > desde) & ~design["imputed"].astype(bool)]
    resultados = {}
    for nombre, columna in {"ultimo_dato": "a0", "naive_24h": "s96"}.items():
        subset = design.dropna(subset=[columna, "demand"])
        resultados[nombre] = official_accuracy(subset.assign(prediction=subset[columna]))
    return resultados


def cross_validate(observations: pd.DataFrame, design: pd.DataFrame) -> list[dict]:
    max_date = observations["observed_at"].max()
    resultados = []
    for i in range(N_FOLDS):
        corte = max_date - pd.Timedelta(days=VALIDATION_DAYS * (i + 1))
        fin = max_date - pd.Timedelta(days=VALIDATION_DAYS * i)
        print(f"\nPliegue {i}: train hasta {corte}, validación {corte} → {fin}")
        baselines = evaluate_baselines(observations, corte, fin)
        bundle = adaptive.train(design[design["observed_at"] <= corte])
        score = official_accuracy(predict_window(bundle, observations, corte, fin))
        mejor_baseline = max(baselines.values())
        print("  baselines: " + ", ".join(f"{k}={v:.2f}" for k, v in baselines.items()))
        print(f"  candidato: {score:.2f} (mejor baseline: {mejor_baseline:.2f})")
        resultados.append({
            "fold": i,
            "baseline_scores": baselines,
            "candidate_score": score,
            "beats_baseline": score > mejor_baseline,
        })
    return resultados


MIN_FILAS_DUELO = 2000
# Tolerancia del duelo: dos corridas de la misma receta empatan (misma
# semilla, mismos datos), y en ese caso debe ganar el modelo con datos más
# frescos. Exigir "estrictamente mayor" congelaba al champion para siempre.
TOLERANCIA_DUELO = 0.10


def duel_against_champion(
    observations: pd.DataFrame, design: pd.DataFrame, champion: dict
) -> tuple[float, float] | None:
    """Compara la receta del candidato con el champion sobre la MISMA ventana.

    La ventana es todo lo posterior al corte de datos del champion, así que
    ninguno de los dos la vio al entrenar. Lo que se juzga es la *receta*: el
    retador se entrena con los mismos datos que tuvo el champion; si no
    pierde, se promueve el candidato entrenado con TODO el histórico.
    """
    corte = pd.Timestamp(champion["data_cutoff"])
    fin = observations["observed_at"].max()
    from src.predict import load_champion

    bundle_champion, _ = load_champion()
    del_champion = predict_window(bundle_champion, observations, corte, fin)
    if len(del_champion) < MIN_FILAS_DUELO:
        print(f"Solo {len(del_champion)} filas posteriores al champion "
              f"(mínimo {MIN_FILAS_DUELO}): no hay con qué comparar todavía.")
        return None

    retador = adaptive.train(design[design["observed_at"] <= corte])
    del_retador = predict_window(retador, observations, corte, fin)
    llaves = ["station_id", "observed_at", "horizon"]
    comun = del_champion.merge(del_retador[llaves + ["prediction"]], on=llaves, suffixes=("_c", "_r"))
    score_champion = official_accuracy(comun, "prediction_c")
    score_retador = official_accuracy(comun, "prediction_r")

    print(f"\nDuelo sobre {len(comun):,} filas posteriores a {corte}:")
    print(f"  champion {champion['version']}: {score_champion:.2f}")
    print(f"  candidato:                     {score_retador:.2f}")
    return score_retador, score_champion


def double_check(bundle: dict, observations: pd.DataFrame) -> None:
    """Dos inferencias sobre las mismas filas deben coincidir exactamente."""
    fin = observations["observed_at"].max()
    desde = fin - pd.Timedelta(hours=12)
    primera = predict_window(bundle, observations, desde, fin)["prediction"].to_numpy()
    segunda = predict_window(bundle, observations, desde, fin)["prediction"].to_numpy()
    if not np.allclose(primera, segunda, equal_nan=True) or not np.isfinite(primera).all():
        raise RuntimeError(
            "Doble verificación falló: dos inferencias sobre las mismas filas "
            "dieron resultados distintos o no finitos. No se registra este modelo."
        )


def run() -> None:
    observations = fetch_all_observations()
    print(f"Observaciones: {len(observations):,} hasta {observations['observed_at'].max()} "
          f"({int(observations['imputed'].sum())} imputadas)")

    design = adaptive.build_design(observations)
    print(f"Filas de diseño (4 horizontes apilados): {len(design):,}")

    print("\n=== Validación cruzada (2 pliegues temporales) ===")
    folds = cross_validate(observations, design)

    todos_superan = all(f["beats_baseline"] for f in folds)
    peor = min(f["candidate_score"] for f in folds)
    mejor = max(f["candidate_score"] for f in folds)
    print(f"\n¿Supera a los baselines en TODOS los pliegues? {'sí' if todos_superan else 'no'}")
    print(f"Accuracy por pliegue: peor={peor:.2f}, mejor={mejor:.2f}")

    if not todos_superan:
        print("\nNo supera los baselines en algún pliegue: no se promueve ni se registra.")
        return

    print("\n=== Modelo final con todo el histórico ===")
    bundle = adaptive.train(design)
    double_check(bundle, observations)
    print(f"Doble verificación: OK. Ensamble de {len(bundle['experts'])} expertos.")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    version = datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%SZ")
    artifact_path = ARTIFACTS_DIR / f"{version}.joblib"
    bundle = {**bundle, "version": version, "horizons": list(HORIZONS)}
    joblib.dump(bundle, artifact_path, compress=3)
    tamano_mb = artifact_path.stat().st_size / 1e6

    metadata = {
        "version": version,
        "kind": bundle["kind"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_cutoff": str(observations["observed_at"].max()),
        "commit_sha": git_commit_sha(),
        "features": FEATURES,
        "experts": bundle["experts"],
        "cross_validation": folds,
        "validation_metric": peor,
        "artifact_mb": round(tamano_mb, 2),
    }
    (ARTIFACTS_DIR / f"{version}.json").write_text(json.dumps(metadata, indent=2))
    print(f"Guardado: {artifact_path} ({tamano_mb:.1f} MB)")
    print(f"validation_metric (peor pliegue): {peor:.2f}")

    from src import db

    try:
        remote = db.upload_model_artifact(artifact_path, f"{version}.joblib")
        print(f"Subido a Supabase Storage: {remote}")
        db.register_model_version(
            version=version,
            data_cutoff=observations["observed_at"].max(),
            artifact_path=remote,
            commit_sha=metadata["commit_sha"],
            features=FEATURES,
            validation_metric=peor,
            status="candidate",
        )
        print("Registrado en Supabase (status=candidate).")

        champion = db.get_champion()
        if champion is None:
            db.promote(version)
            print(f"PROMOVIDO a champion (era el primero): {version}.")
            return

        duelo = duel_against_champion(observations, design, champion)
        if duelo is None:
            # Sin datos nuevos suficientes para juzgar una receta distinta, no
            # se toca al champion: la novedad por sí sola no es una mejora. Si
            # la receta es la misma, el candidato solo agrega datos frescos.
            if list(champion.get("features") or []) == FEATURES:
                db.promote(version)
                print(f"PROMOVIDO a champion: {version} (misma receta, datos más frescos).")
            else:
                print("No promovido: falta evidencia nueva para comparar recetas.")
            return

        score_retador, score_champion = duelo
        if score_retador >= score_champion - TOLERANCIA_DUELO:
            db.promote(version)
            print(f"PROMOVIDO a champion: {version} "
                  f"({score_retador:.2f} vs {score_champion:.2f} del anterior).")
        else:
            print(f"No promovido: el champion {champion['version']} sigue mejor "
                  f"({score_champion:.2f} vs {score_retador:.2f}) en la misma ventana.")
    except db.SupabaseNotConfigured:
        print("SUPABASE_URL/SUPABASE_KEY no configurados: modelo solo local.")


if __name__ == "__main__":
    run()
