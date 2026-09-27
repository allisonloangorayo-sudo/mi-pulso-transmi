import numpy as np
import pandas as pd

from src import model
from src.features import STEPS_PER_WEEK, build_global_training_design


def _synthetic(stations=("A", "B"), weeks: int = 4) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=STEPS_PER_WEEK * weeks, freq="15min", tz="UTC")
    rng = np.random.default_rng(0)
    frames = []
    for escala, s in zip((100, 300), stations):
        base = escala * (1.5 + np.sin(np.arange(len(index)) * 2 * np.pi / 96))
        frames.append(pd.DataFrame({
            "station_id": s, "observed_at": index,
            "demand": rng.poisson(base).astype(int),
        }))
    return pd.concat(frames, ignore_index=True)


def test_global_bundle_predicts_every_row_non_negative():
    design = build_global_training_design(_synthetic())
    bundle = model.train_global(design)
    pred = model.predict(bundle, design.sample(200, random_state=0))
    assert pred.shape == (200,)
    assert (pred >= 0).all()


def test_global_bundle_is_deterministic():
    design = build_global_training_design(_synthetic())
    bundle = model.train_global(design)
    muestra = design.head(300)
    assert np.allclose(model.predict(bundle, muestra), model.predict(bundle, muestra))


def test_global_bundle_follows_a_level_shift():
    """Tras una caída de nivel, la predicción debe acercarse al nivel nuevo,
    no quedarse en el de la semana pasada."""
    obs = _synthetic()
    design = build_global_training_design(obs)
    bundle = model.train_global(design)
    caida = obs.copy()
    ultimos = caida.index[(caida["station_id"] == "B") & (caida.index >= caida[caida["station_id"] == "B"].index[-48])]
    caida.loc[ultimos, "demand"] = (caida.loc[ultimos, "demand"] * 0.4).astype(int)
    fila = build_global_training_design(caida).query("station_id == 'B' and horizon == 1").tail(4)
    pred = model.predict(bundle, fila)
    semana_pasada = fila["tweek"].to_numpy()
    assert (np.abs(pred - fila["demand"].to_numpy()) < np.abs(semana_pasada - fila["demand"].to_numpy())).all()
