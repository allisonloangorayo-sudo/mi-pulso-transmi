import pandas as pd
import pytest

from src.features import (
    FEATURE_COLUMNS,
    HORIZONS,
    STEPS_PER_WEEK,
    build_design,
    build_serving_design,
    build_training_design,
)


def _synthetic(stations=("A", "B"), n_periods: int = STEPS_PER_WEEK * 3) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=n_periods, freq="15min", tz="UTC")
    frames = [
        pd.DataFrame({"station_id": s, "observed_at": index, "demand": range(n_periods)})
        for s in stations
    ]
    return pd.concat(frames, ignore_index=True)


def test_training_design_has_all_feature_columns():
    design = build_training_design(_synthetic())
    for column in FEATURE_COLUMNS:
        assert column in design.columns
    assert not design[FEATURE_COLUMNS].isna().any().any()


def test_training_design_stacks_every_horizon():
    design = build_training_design(_synthetic())
    assert set(design["horizon"]) == set(HORIZONS)


@pytest.mark.parametrize("horizon", HORIZONS)
def test_slot_s0_is_the_last_observation_available(horizon):
    """s0 debe ser el dato de `horizon` pasos antes del target — nunca más
    reciente, o estaríamos usando información del futuro."""
    observations = _synthetic(stations=("A",))
    design = build_design(observations, horizon).dropna(subset=["s0"])
    fila = design.iloc[-1]
    esperado = observations[
        observations["observed_at"] == fila["observed_at"] - pd.Timedelta(minutes=15 * horizon)
    ]["demand"].iloc[0]
    assert fila["s0"] == esperado


@pytest.mark.parametrize("horizon", HORIZONS)
def test_serving_matches_training_for_the_same_instant(horizon):
    """Regresión del bug de train/serve skew.

    Las features que produce el camino de servicio para un target futuro deben
    ser idénticas a las que produciría el camino de entrenamiento para ese
    mismo instante. Si alguien vuelve a congelar los lags en el corte, esta
    prueba falla.
    """
    observations = _synthetic(stations=("A",))
    corte = observations["observed_at"].max()
    target = corte + pd.Timedelta(minutes=15 * horizon)

    historia = pd.concat([
        observations,
        pd.DataFrame({"station_id": ["A"], "observed_at": [target], "demand": [999]}),
    ], ignore_index=True)
    esperado = build_design(historia, horizon)
    esperado = esperado[esperado["observed_at"] == target].iloc[0]

    targets = pd.DataFrame([{"station_id": "A", "target_at": target, "horizon": horizon}])
    obtenido = build_serving_design(observations, targets).iloc[0]

    for columna in FEATURE_COLUMNS:
        assert obtenido[columna] == pytest.approx(esperado[columna]), columna
