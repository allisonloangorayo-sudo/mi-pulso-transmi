"""Baselines + candidato, con validación cruzada temporal de 2 pliegues.

Uso:
    python -m src.train

Por qué 2 pliegues y no 1: un solo split train/validación puede premiar un
candidato que solo tuvo suerte en esa ventana particular. Con 2 pliegues
temporales independientes exigimos que el candidato supere a los baselines
en AMBOS, y la promoción a champion usa el peor de los dos accuracies (el
caso más conservador), no el mejor.

Genera:
    artifacts/model_<version>.joblib   (candidato final, reentrenado con
                                         TODO el histórico una vez validado)
    artifacts/model_<version>.json     (metadata: features, métricas por
                                         pliegue, cutoff)

No promueve si el peor pliegue no supera al champion vigente (ver
db.promote_if_better) — "la novedad por sí sola no es una mejora".
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
from pulso_transmi import PulsoTransmiClient
from sklearn.ensemble import RandomForestRegressor

from src.features import FEATURE_COLUMNS, STEPS_PER_DAY, STEPS_PER_WEEK, build_features

load_dotenv()

ARTIFACTS_DIR = Path("artifacts")
VALIDATION_DAYS = 7
N_FOLDS = 2


def wape_accuracy_by_station(frame: pd.DataFrame, prediction_col: str) -> pd.Series:
    error = (frame["demand"] - frame[prediction_col]).abs()
    wape = error.groupby(frame["station_id"]).sum() / frame["demand"].groupby(frame["station_id"]).sum()
    return (100 * (1 - wape)).clip(lower=0)


def temporal_split(df: pd.DataFrame, validation_days: int = VALIDATION_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["observed_at"].max() - pd.Timedelta(days=validation_days)
    train = df[df["observed_at"] <= cutoff].copy()
    validation = df[df["observed_at"] > cutoff].copy()
    return train, validation


def temporal_folds(
    df: pd.DataFrame, n_folds: int = N_FOLDS, validation_days: int = VALIDATION_DAYS
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Pliegues temporales retrocediendo una ventana de validación por vez.
    Fold 0 = los últimos `validation_days`; fold 1 = los `validation_days`
    anteriores a esos; etc. Cada fold solo entrena con datos previos a su
    propia ventana (sin fuga de futuro).
    """
    max_date = df["observed_at"].max()
    folds = []
    for i in range(n_folds):
        cutoff = max_date - pd.Timedelta(days=validation_days * (i + 1))
        window_end = max_date - pd.Timedelta(days=validation_days * i)
        train = df[df["observed_at"] <= cutoff]
        validation = df[(df["observed_at"] > cutoff) & (df["observed_at"] <= window_end)]
        folds.append((train, validation))
    return folds


def git_commit_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def evaluate_baselines(features_df: pd.DataFrame, validation: pd.DataFrame) -> dict[str, float]:
    results = {}
    for name, lag in {"naive_24h": STEPS_PER_DAY, "seasonal_7d": STEPS_PER_WEEK}.items():
        col = f"lag_{lag}"
        subset = features_df.loc[validation.index].dropna(subset=[col])
        scored = subset.assign(prediction=subset[col])
        scores = wape_accuracy_by_station(scored, "prediction")
        results[name] = float(scores.mean())
    return results


def train_model(train_df: pd.DataFrame, features_df: pd.DataFrame) -> RandomForestRegressor:
    train_features = features_df.loc[train_df.index].dropna(subset=FEATURE_COLUMNS + ["demand"])
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(train_features[FEATURE_COLUMNS], train_features["demand"])
    return model


def evaluate_model(model: RandomForestRegressor, features_df: pd.DataFrame, validation: pd.DataFrame) -> float:
    subset = features_df.loc[validation.index].dropna(subset=FEATURE_COLUMNS)
    predictions = model.predict(subset[FEATURE_COLUMNS])
    scored = subset.assign(prediction=predictions)
    scores = wape_accuracy_by_station(scored, "prediction")
    return float(scores.mean())


def cross_validate(observations: pd.DataFrame, features_df: pd.DataFrame) -> list[dict]:
    """Entrena y evalúa un candidato por pliegue. Devuelve una lista con el
    resultado de cada pliegue (baselines, accuracy del candidato, si superó
    al mejor baseline)."""
    results = []
    for i, (train_df, validation_df) in enumerate(temporal_folds(observations)):
        print(f"\nPliegue {i}: train hasta {train_df['observed_at'].max()}, "
              f"validación {validation_df['observed_at'].min()} → {validation_df['observed_at'].max()}")
        baseline_scores = evaluate_baselines(features_df, validation_df)
        model = train_model(train_df, features_df)
        candidate_score = evaluate_model(model, features_df, validation_df)
        best_baseline = max(baseline_scores.values())
        print(f"  baselines: {baseline_scores}")
        print(f"  candidato: {candidate_score:.2f} (mejor baseline: {best_baseline:.2f})")
        results.append(
            {
                "fold": i,
                "baseline_scores": baseline_scores,
                "candidate_score": candidate_score,
                "beats_baseline": candidate_score > best_baseline,
            }
        )
    return results


def double_check_predictions(model: RandomForestRegressor, features_df: pd.DataFrame) -> None:
    """Segunda pasada de inferencia sobre las mismas filas: si no coincide
    exactamente con la primera, algo no es determinista (entorno, datos
    mutados a mitad de camino, etc.) y no se debe confiar en el modelo."""
    clean = features_df.dropna(subset=FEATURE_COLUMNS)
    first = model.predict(clean[FEATURE_COLUMNS])
    second = model.predict(clean[FEATURE_COLUMNS])
    if not np.allclose(first, second):
        raise RuntimeError(
            "Doble verificación falló: dos inferencias con las mismas filas dieron "
            "resultados distintos. No se registra este modelo."
        )


def run() -> None:
    with PulsoTransmiClient() as client:
        observations = client.observations_dataframe()

    features_df = build_features(observations)

    print("=== Validación cruzada (2 pliegues temporales) ===")
    fold_results = cross_validate(observations, features_df)

    all_beat_baseline = all(f["beats_baseline"] for f in fold_results)
    worst_fold_score = min(f["candidate_score"] for f in fold_results)
    best_fold_score = max(f["candidate_score"] for f in fold_results)
    print(f"\n¿Supera al mejor baseline en TODOS los pliegues? {'sí' if all_beat_baseline else 'no'}")
    print(f"Accuracy por pliegue: peor={worst_fold_score:.2f}, mejor={best_fold_score:.2f}")

    if not all_beat_baseline:
        print("\nEl candidato no superó los baselines en algún pliegue: no se promueve ni se registra.")
        return

    print("\n=== Reentrenamiento final con todo el histórico ===")
    final_model = train_model(observations, features_df)
    double_check_predictions(final_model, features_df)
    print("Doble verificación de sanidad: OK (dos inferencias coinciden).")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    version = datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%SZ")
    artifact_path = ARTIFACTS_DIR / f"model_{version}.joblib"
    joblib.dump(final_model, artifact_path)

    metadata = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_cutoff": str(observations["observed_at"].max()),
        "commit_sha": git_commit_sha(),
        "features": FEATURE_COLUMNS,
        "cross_validation": fold_results,
        "validation_metric": worst_fold_score,
        "artifact_path": str(artifact_path),
    }
    metadata_path = ARTIFACTS_DIR / f"model_{version}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"\nGuardado: {artifact_path} y {metadata_path}")
    print(f"validation_metric registrado (peor pliegue): {worst_fold_score:.2f}")

    from src import db

    try:
        remote_path = db.upload_model_artifact(artifact_path, f"{version}.joblib")
        print(f"Subido a Supabase Storage: {remote_path}")
        db.register_model_version(
            version=version,
            data_cutoff=observations["observed_at"].max(),
            artifact_path=remote_path,
            commit_sha=metadata["commit_sha"],
            features=FEATURE_COLUMNS,
            validation_metric=worst_fold_score,
            status="candidate",
        )
        print("Registrado en Supabase (model_versions, status=candidate).")

        promoted = db.promote_if_better(version, worst_fold_score)
        if promoted:
            print(f"PROMOVIDO a champion: {version} (accuracy peor-pliegue {worst_fold_score:.2f}).")
        else:
            champion = db.get_champion()
            print(
                f"No promovido: el champion vigente ({champion['version']}, "
                f"{champion['validation_metric']:.2f}) sigue siendo mejor o igual "
                f"que el peor pliegue de este candidato ({worst_fold_score:.2f})."
            )
    except db.SupabaseNotConfigured:
        print("SUPABASE_URL/SUPABASE_KEY no configurados: modelo guardado solo localmente.")


if __name__ == "__main__":
    run()
