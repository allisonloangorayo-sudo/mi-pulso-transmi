import numpy as np
import pandas as pd
import pytest

from src import adaptive, model


def _serie(periodos=(96,), pasos=96 * 18, stations=("A", "B"), seed=0):
    """Demanda con un ciclo que cambia de periodo a mitad de la serie."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-01", periods=pasos, freq="15min", tz="UTC")
    tramo = pasos // len(periodos)
    frames = []
    for escala, s in zip((100, 300), stations):
        t = np.arange(pasos)
        p = np.array([periodos[min(i // tramo, len(periodos) - 1)] for i in t])
        base = escala * (1.5 + np.sin(2 * np.pi * t / p))
        frames.append(pd.DataFrame({
            "station_id": s, "observed_at": index,
            "demand": rng.poisson(base).astype(float), "imputed": False,
        }))
    frame = pd.concat(frames, ignore_index=True)
    frame["station_id"] = frame["station_id"].astype("string")
    return frame


def test_period_track_finds_the_new_cycle_after_a_regime_change():
    obs = _serie(periodos=(96, 16), pasos=96 * 6)
    y = obs[obs["station_id"] == "A"]["demand"].to_numpy()
    periodo, _ = adaptive.period_track(y, 16)
    assert periodo[-1] in (16, 32, 48)
    assert periodo[96 * 2] in (95, 96, 97)


@pytest.mark.parametrize("horizon", adaptive.HORIZONS)
def test_features_do_not_look_past_the_cutoff(horizon):
    """Cambiar el futuro no puede cambiar las features del instante objetivo."""
    obs = _serie(pasos=96 * 4, stations=("A",))
    objetivo = obs["observed_at"].iloc[-10]
    original = adaptive.build_design(obs, (horizon,))
    alterado = obs.copy()
    corte = objetivo - pd.Timedelta(minutes=15 * horizon)
    alterado.loc[alterado["observed_at"] > corte, "demand"] *= 7
    cambiado = adaptive.build_design(alterado, (horizon,))
    fila = lambda d: d[d["observed_at"] == objetivo][adaptive.FEATURES].iloc[0]  # noqa: E731
    pd.testing.assert_series_equal(fila(original), fila(cambiado))


@pytest.fixture(scope="module")
def bundle_y_obs():
    obs = _serie(periodos=(96, 32), pasos=96 * 17)
    return adaptive.train(adaptive.build_design(obs)), obs


def test_forecast_targets_covers_every_target_without_gaps(bundle_y_obs):
    bundle, obs = bundle_y_obs
    corte = obs["observed_at"].max()
    targets = pd.DataFrame([
        {"station_id": s, "target_at": corte + pd.Timedelta(minutes=15 * h), "horizon": h}
        for s in ("A", "B") for h in adaptive.HORIZONS
    ]).astype({"station_id": "string"})
    salida = model.forecast_targets(bundle, obs, targets)
    assert len(salida) == len(targets)
    assert np.isfinite(salida["value"]).all()
    assert (salida["value"] >= 0).all()


def test_ensemble_is_deterministic(bundle_y_obs):
    bundle, obs = bundle_y_obs
    design = adaptive.build_design(obs.tail(96 * 6 * 2))
    assert np.allclose(adaptive.predict(bundle, design), adaptive.predict(bundle, design), equal_nan=True)
