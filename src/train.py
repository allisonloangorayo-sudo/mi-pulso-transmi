"""Entrenamiento: un modelo por estación, horizonte como feature.

Uso:
    python -m src.train

Decisiones respaldadas por experimentos (métrica oficial, 4 horizontes,
mismo conjunto de validación en todas las variantes):

    producción anterior (1 modelo, lags congelados)      79.24%
    + un modelo por horizonte (elimina el train/serve skew)  84.60%
    + estación como feature                               85.83%
    + un modelo por estación                              86.64%
    + ensamble RF/boosting (48 modelos, ~2 GB)            86.74%
    ESTA: 12 modelos por estación, boosting MAE, 11.8 MB  86.69%

Se eligió la última: empata en accuracy con el ensamble y pesa 99% menos,
lo que importa porque `infer` descarga el artefacto en cada corrida.
La pérdida MAE se usa porque la métrica oficial (WAPE) es error absoluto.

Promoción: validación cruzada de 2 pliegues temporales; el candidato debe
superar a los baselines en AMBOS, y reemplaza al champion solo si su peor
pliegue supera la métrica del champion vigente.
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
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data import fetch_all_observations
from src.features import (
    FEATURE_COLUMNS,
    HORIZONS,
    STEPS_PER_DAY,
    STEPS_PER_WEEK,
    build_training_design,
)

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


def make_model() -> HistGradientBoostingRegressor:
    # loss="absolute_error" alinea el entrenamiento con WAPE, la métrica oficial.
    return HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=600, learning_rate=0.05, random_state=42
    )


def train_bundle(train_design: pd.DataFrame) -> dict[str, HistGradientBoostingRegressor]:
    modelos = {}
    for station_id, grupo in train_design.groupby("station_id", observed=True):
        modelo = make_model()
        modelo.fit(grupo[FEATURE_COLUMNS], grupo["demand"])
        modelos[str(station_id)] = modelo
    return modelos


def predict_bundle(modelos: dict, design: pd.DataFrame) -> np.ndarray:
    salida = np.zeros(len(design))
    for station_id, grupo in design.groupby("station_id", observed=True):
        modelo = modelos.get(str(station_id))
        if modelo is None:
            continue
        salida[design.index.get_indexer(grupo.index)] = modelo.predict(grupo[FEATURE_COLUMNS])
    return np.clip(salida, 0, None)


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

    design = build_training_design(observations)
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
    print(f"Doble verificación: OK. {len(modelos)} modelos (uno por estación).")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    version = datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%SZ")
    artifact_path = ARTIFACTS_DIR / f"{version}.joblib"
    bundle = {
        "version": version,
        "models": modelos,
        "features": FEATURE_COLUMNS,
        "horizons": list(HORIZONS),
    }
    joblib.dump(bundle, artifact_path, compress=3)
    tamano_mb = artifact_path.stat().st_size / 1e6

    metadata = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_cutoff": str(observations["observed_at"].max()),
        "commit_sha": git_commit_sha(),
        "features": FEATURE_COLUMNS,
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
            features=FEATURE_COLUMNS,
            validation_metric=peor,
            status="candidate",
        )
        print("Registrado en Supabase (status=candidate).")

        if db.promote_if_better(version, peor):
            print(f"PROMOVIDO a champion: {version} (peor pliegue {peor:.2f}).")
        else:
            champion = db.get_champion()
            print(f"No promovido: champion vigente {champion['version']} "
                  f"({champion['validation_metric']:.2f}) sigue mejor o igual.")
    except db.SupabaseNotConfigured:
        print("SUPABASE_URL/SUPABASE_KEY no configurados: modelo solo local.")


if __name__ == "__main__":
    run()
