import pandas as pd

from src.features import FEATURE_COLUMNS, STEPS_PER_DAY, build_features


def _synthetic_observations(n_periods: int = STEPS_PER_DAY * 3) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=n_periods, freq="15min", tz="UTC")
    frames = []
    for station_id in ("A", "B"):
        frames.append(
            pd.DataFrame(
                {
                    "station_id": station_id,
                    "observed_at": index,
                    "demand": range(n_periods),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_build_features_has_expected_columns():
    observations = _synthetic_observations()
    features = build_features(observations)
    for column in FEATURE_COLUMNS:
        assert column in features.columns


def test_lag_features_shift_within_station_only():
    observations = _synthetic_observations()
    features = build_features(observations)
    row = features[(features["station_id"] == "A")].iloc[5]
    expected_lag1 = observations[(observations["station_id"] == "A")].iloc[4]["demand"]
    assert row["lag_1"] == expected_lag1


def test_no_duplicate_station_timestamp_pairs():
    observations = _synthetic_observations()
    assert not observations.duplicated(subset=["station_id", "observed_at"]).any()
