"""Entrenamiento: modelo global normalizado por nivel (ver src.model).

Uso:
    python -m src.train

Decisiones respaldadas por experimentos (métrica oficial, 4 horizontes,
mismo conjunto de validación en todas las variantes):

    producción anterior (1 modelo, lags congelados)      79.24%
    + un modelo por horizonte (elimina el train/serve skew)  84.60%
    + estación como feature                               85.83%
    + un modelo por estación                              86.64%
    + ensamble RF/boosting (48 modelos, ~2 GB)            86.74%
    12 modelos por estación, boosting MAE, 11.8 MB        86.69%

Esas cifras son sobre histórico estable. En la ventana competitiva (ciclos
reales, 11 a 15-sep) aparecieron cambios de nivel (05100 cayó al ~40%) y el
modelo por estación se quedó en 82.48%. Ver experiments/06_nivel_adaptativo.py:

    por estación, crudo                                   82.48%
    ESTA: global normalizado por nivel, ensamble 3 escalas  (ver experimento)

La pérdida MAE se usa porque la métrica oficial (WAPE) es error absoluto.

Promoción: validación cruzada de 2 pliegues temporales; el candidato debe
superar a los baselines en AMBOS, y luego gana el duelo contra el champion
sobre la misma ventana (ver duel_against_champion).
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

from src import model
from src.data import fetch_all_observations
from src.features import GLOBAL_FEATURES, HORIZONS, build_global_training_design

load_dotenv()

ARTIFACTS_DIR = Path("artifacts")
VALIDATION_DAYS = 5
N_FOLDS = 2


def wape_accuracy_by_station(frame: pd.DataFrame, prediction_col: str) -> pd.Series:
    error = (frame["demand"] - frame[prediction_col]).abs()
    wape = error.groupby(frame["station_id"]).sum() / frame["demand"].groupby(frame["station_id"]).sum()
    return (100 * (1 - wape)).clip(lower=0)


def official_accuracy(frame: pd.DataFrame, prediction_col: str = "prediction") -> float:
    return float(wape_accuracy_by_station(frame, prediction_col).mean())


def temporal_split(df: pd.DataFrame, validation_days: int = VALIDATION_DAYS):
    cutoff = df["observed_at"].max() - pd.Timedelta(days=validation_days)
    return df[df["observed_at"] <= cutoff].copy(), df[df["observed_at"] > cutoff].copy()


def temporal_folds(design: pd.DataFrame, n_folds: int = N_FOLDS, validation_days: int = VALIDATION_DAYS):
    max_date = design["observed_at"].max()
    folds = []
    for i in range(n_folds):
        cutoff = max_date - pd.Timedelta(days=validation_days * (i + 1))
        window_end = max_date - pd.Timedelta(days=validation_days * i)
        train = design[design["observed_at"] <= cutoff]
        validation = design[(design["observed_at"] > cutoff) & (design["observed_at"] <= window_end)]
        folds.append((train, validation))
    return folds


def git_commit_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def train_bundle(train_design: pd.DataFrame) -> dict:
    return model.train_global(train_design)


def predict_bundle(bundle: dict, design: pd.DataFrame) -> np.ndarray:
    return model.predict(bundle, design)


def evaluate_baselines(validation: pd.DataFrame) -> dict[str, float]:
    """Baselines sobre el mismo conjunto: repetir el valor de hace 24h / 7d."""
    resultados = {}
    for nombre, columna in {"naive_24h": "sday", "seasonal_7d": "sweek"}.items():
        subset = validation.dropna(subset=[columna])
        resultados[nombre] = official_accuracy(subset.assign(prediction=subset[columna]))
    return resultados


def cross_validate(design: pd.DataFrame) -> list[dict]:
    resultados = []
    for i, (train, validation) in enumerate(temporal_folds(design)):
        if train.empty or validation.empty:
            continue
        print(f"\nPliegue {i}: train hasta {train['observed_at'].max()}, "
              f"validación {validation['observed_at'].min()} → {validation['observed_at'].max()}")
        baselines = evaluate_baselines(validation)
        modelos = train_bundle(train)
        validation = validation.reset_index(drop=True)
        score = official_accuracy(validation.assign(prediction=predict_bundle(modelos, validation)))
        mejor_baseline = max(baselines.values())
        print(f"  baselines: " + ", ".join(f"{k}={v:.2f}" for k, v in baselines.items()))
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


def duel_against_champion(design: pd.DataFrame, champion: dict) -> tuple[float, float] | None:
    """Compara la receta del candidato con el champion sobre la MISMA ventana.

    Por qué existe: la métrica guardada de cada modelo se calculó sobre una
    ventana temporal distinta, y unas ventanas son más difíciles que otras
    (medido: la misma configuración da 86.7% en una semana y 85.2% en la
    siguiente). Comparar esos números sueltos haría que un modelo entrenado
    con datos más frescos pierda solo porque le tocó una ventana difícil, y
    el champion se quedaría congelado para siempre.

    La ventana del duelo es todo lo posterior al corte de datos del champion,
    así que ninguno de los dos la vio al entrenar. Devuelve None si todavía
    no hay suficientes datos nuevos para juzgar.

    Ojo: lo que se juzga es la *receta* (el retador se entrena con los mismos
    datos que tuvo el champion). Si la receta no pierde, se promueve el
    candidato entrenado con TODO el histórico, que además tiene los datos
    nuevos.
    """
    corte_champion = pd.Timestamp(champion["data_cutoff"])
    holdout = design[design["observed_at"] > corte_champion]
    if len(holdout) < MIN_FILAS_DUELO:
        print(f"Solo {len(holdout)} filas posteriores al champion "
              f"(mínimo {MIN_FILAS_DUELO}): no hay con qué comparar todavía.")
        return None

    entrenamiento = design[design["observed_at"] <= corte_champion]
    retador = train_bundle(entrenamiento)
    holdout = holdout.reset_index(drop=True)
    score_retador = official_accuracy(holdout.assign(prediction=predict_bundle(retador, holdout)))

    from src.predict import load_champion

    bundle_champion, _ = load_champion()
    score_champion = official_accuracy(
        holdout.assign(prediction=model.predict(bundle_champion, holdout))
    )

    print(f"\nDuelo sobre {len(holdout):,} filas posteriores a {corte_champion}:")
    print(f"  champion {champion['version']}: {score_champion:.2f}")
    print(f"  candidato:                     {score_retador:.2f}")
    return score_retador, score_champion


def double_check(modelos: dict, design: pd.DataFrame) -> None:
    """Dos inferencias sobre las mismas filas deben coincidir exactamente."""
    muestra = design.head(2000).reset_index(drop=True)
    primera = predict_bundle(modelos, muestra)
    segunda = predict_bundle(modelos, muestra)
    if not np.allclose(primera, segunda):
        raise RuntimeError(
            "Doble verificación falló: dos inferencias sobre las mismas filas "
            "dieron resultados distintos. No se registra este modelo."
        )


def run() -> None:
    observations = fetch_all_observations()
    print(f"Observaciones: {len(observations):,} hasta {observations['observed_at'].max()}")

    design = build_global_training_design(observations)
    print(f"Filas de entrenamiento (4 horizontes apilados): {len(design):,}")

    print("\n=== Validación cruzada (2 pliegues temporales) ===")
    folds = cross_validate(design)
    if not folds:
        print("Sin datos suficientes para validar. No se registra nada.")
        return

    todos_superan = all(f["beats_baseline"] for f in folds)
    peor = min(f["candidate_score"] for f in folds)
    mejor = max(f["candidate_score"] for f in folds)
    print(f"\n¿Supera a los baselines en TODOS los pliegues? {'sí' if todos_superan else 'no'}")
    print(f"Accuracy por pliegue: peor={peor:.2f}, mejor={mejor:.2f}")

    if not todos_superan:
        print("\nNo supera los baselines en algún pliegue: no se promueve ni se registra.")
        return

    print("\n=== Modelo final con todo el histórico ===")
    modelos = train_bundle(design)
    double_check(modelos, design)
    print(f"Doble verificación: OK. Ensamble global de {len(modelos['models'])} escalas.")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    version = datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%SZ")
    artifact_path = ARTIFACTS_DIR / f"{version}.joblib"
    bundle = {**modelos, "version": version, "horizons": list(HORIZONS)}
    joblib.dump(bundle, artifact_path, compress=3)
    tamano_mb = artifact_path.stat().st_size / 1e6

    metadata = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_cutoff": str(observations["observed_at"].max()),
        "commit_sha": git_commit_sha(),
        "features": GLOBAL_FEATURES,
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
            features=GLOBAL_FEATURES,
            validation_metric=peor,
            status="candidate",
        )
        print("Registrado en Supabase (status=candidate).")

        champion = db.get_champion()
        if champion is None:
            db.promote(version)
            print(f"PROMOVIDO a champion (era el primero): {version}.")
            return

        duelo = duel_against_champion(design, champion)
        if duelo is None:
            # Sin datos nuevos suficientes para juzgar una receta distinta, no
            # se toca al champion: la novedad por sí sola no es una mejora. Si
            # la receta es la misma, el candidato solo agrega datos frescos.
            if list(champion.get("features") or []) == GLOBAL_FEATURES:
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
