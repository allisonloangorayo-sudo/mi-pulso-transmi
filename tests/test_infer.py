import pandas as pd

from src.features import HORIZONS, STEPS_PER_WEEK
from src.infer import _build_targets, _initial_sim_cutoff


def _synthetic(stations=("A", "B", "C"), n_periods: int = STEPS_PER_WEEK * 2) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=n_periods, freq="15min", tz="UTC")
    frames = [
        pd.DataFrame({"station_id": s, "observed_at": index, "demand": range(n_periods)})
        for s in stations
    ]
    return pd.concat(frames, ignore_index=True)


def test_targets_cover_every_station_and_horizon():
    observations = _synthetic()
    cutoff = _initial_sim_cutoff(observations)
    targets = _build_targets(observations, cutoff)

    assert len(targets) == 3 * len(HORIZONS)
    assert set(targets["station_id"]) == {"A", "B", "C"}
    assert set(targets["horizon"]) == set(HORIZONS)


def test_targets_are_15_minutes_apart_after_the_cutoff():
    observations = _synthetic(stations=("A",))
    cutoff = _initial_sim_cutoff(observations)
    targets = _build_targets(observations, cutoff).sort_values("horizon")

    esperados = [cutoff + pd.Timedelta(minutes=15 * h) for h in sorted(HORIZONS)]
    assert list(targets["target_at"]) == esperados
