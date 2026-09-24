"""Baselines + candidato, con validación temporal (Fase 1 y 3 del taller).

Uso:
    python -m src.train

Genera:
    artifacts/model_<version>.joblib   (candidato, si supera a los baselines)
    artifacts/model_<version>.json     (metadata: features, métrica, cutoff)

No promueve automáticamente a "champion": esa es una decisión explícita
(ver README, sección "Promoción del champion").
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


def wape_accuracy_by_station(frame: pd.DataFrame, prediction_col: str) -> pd.Series:
    error = (frame["demand"] - frame[prediction_col]).abs()
    wape = error.groupby(frame["station_id"]).sum() / frame["demand"].groupby(frame["station_id"]).sum()
    return (100 * (1 - wape)).clip(lower=0)


def temporal_split(df: pd.DataFrame, validation_days: int = VALIDATION_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["observed_at"].max() - pd.Timedelta(days=validation_days)
    train = df[df["observed_at"] <= cutoff].copy()
    validation = df[df["observed_at"] > cutoff].copy()
    return train, validation


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
        print(f"  baseline {name}: accuracy promedio = {results[name]:.2f}")
    return results


def train_candidate(train_df: pd.DataFrame) -> RandomForestRegressor:
    clean = train_df.dropna(subset=FEATURE_COLUMNS + ["demand"])
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(clean[FEATURE_COLUMNS], clean["demand"])
    return model


def evaluate_candidate(model: RandomForestRegressor, features_df: pd.DataFrame, validation: pd.DataFrame) -> float:
    subset = features_df.loc[validation.index].dropna(subset=FEATURE_COLUMNS)
    predictions = model.predict(subset[FEATURE_COLUMNS])
    scored = subset.assign(prediction=predictions)
    scores = wape_accuracy_by_station(scored, "prediction")
    print(f"  candidato RandomForest: accuracy promedio = {scores.mean():.2f}")
    return float(scores.mean())


def run() -> None:
    with PulsoTransmiClient() as client:
        observations = client.observations_dataframe()

    features_df = build_features(observations)
    train_df, validation_df = temporal_split(observations)

    print(f"Train: {len(train_df):,} filas hasta {train_df['observed_at'].max()}")
    print(f"Validation: {len(validation_df):,} filas desde {validation_df['observed_at'].min()}")

    print("\nBaselines:")
    baseline_scores = evaluate_baselines(features_df, validation_df)

    print("\nCandidato:")
    train_features = features_df.loc[train_df.index]
    model = train_candidate(train_features)
    candidate_score = evaluate_candidate(model, features_df, validation_df)

    best_baseline = max(baseline_scores.values())
    promotable = candidate_score > best_baseline
    print(f"\n¿Candidato supera al mejor baseline ({best_baseline:.2f})? {'sí' if promotable else 'no'}")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    version = datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%SZ")
    artifact_path = ARTIFACTS_DIR / f"model_{version}.joblib"
    joblib.dump(model, artifact_path)

    metadata = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_cutoff": str(train_df["observed_at"].max()),
        "commit_sha": git_commit_sha(),
        "features": FEATURE_COLUMNS,
        "validation_metric": candidate_score,
        "baseline_scores": baseline_scores,
        "promotable": promotable,
        "artifact_path": str(artifact_path),
    }
    metadata_path = ARTIFACTS_DIR / f"model_{version}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"\nGuardado: {artifact_path} y {metadata_path}")

    from src import db

    try:
        db.register_model_version(
            version=version,
            data_cutoff=train_df["observed_at"].max(),
            artifact_path=str(artifact_path),
            commit_sha=metadata["commit_sha"],
            features=FEATURE_COLUMNS,
            validation_metric=candidate_score,
            status="candidate",
        )
        print("Registrado en Supabase (model_versions, status=candidate).")
    except db.SupabaseNotConfigured:
        print("SUPABASE_URL/SUPABASE_KEY no configurados: modelo guardado solo localmente.")


if __name__ == "__main__":
    run()
