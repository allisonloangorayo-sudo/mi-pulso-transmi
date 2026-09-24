import pandas as pd

from src.features import STEPS_PER_WEEK
from src.infer import HORIZONS_MIN, _build_batch, _initial_sim_cutoff


def _synthetic_observations(stations=("A", "B", "C"), n_periods: int = STEPS_PER_WEEK * 2) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=n_periods, freq="15min", tz="UTC")
    frames = [
        pd.DataFrame({"station_id": s, "observed_at": index, "demand": range(n_periods)})
        for s in stations
    ]
    return pd.concat(frames, ignore_index=True)


def test_build_batch_has_one_row_per_station_per_horizon():
    observations = _synthetic_observations()
    cutoff = _initial_sim_cutoff(observations) + pd.Timedelta(hours=1)
    batch = _build_batch(observations, cutoff)

    assert len(batch) == 3 * len(HORIZONS_MIN)
    assert set(batch["station_id"]) == {"A", "B", "C"}
    for minutes in HORIZONS_MIN:
        assert (cutoff + pd.Timedelta(minutes=minutes)) in set(batch["target_at"])


def test_build_batch_has_no_missing_features():
    observations = _synthetic_observations()
    cutoff = _initial_sim_cutoff(observations) + pd.Timedelta(hours=1)
    batch = _build_batch(observations, cutoff)
    assert not batch.isna().any().any()
