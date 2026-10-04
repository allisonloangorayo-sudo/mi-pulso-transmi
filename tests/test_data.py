import pandas as pd

from src.data import fill_gaps, parse_stream_records


def test_parses_schema_v1_and_v2():
    """Regresión: el stream pasó de `demand` a `measurement.value` (texto) y
    todo lo nuevo quedó NaN, lo que abortó cada envío."""
    filas = [
        {"station_id": "02300", "observed_at": "2026-09-20T12:00:00Z", "demand": 556},
        {
            "station_id": "02300", "observed_at": "2026-09-20T12:15:00Z", "schema_version": 2,
            "measurement": {"value": "546.00", "unit": "passengers", "quality": "observed"},
        },
        {
            "station_id": "02300", "observed_at": "2026-09-20T12:30:00Z", "schema_version": 2,
            "measurement": {"value": None, "unit": "passengers", "quality": "missing"},
        },
    ]
    frame = parse_stream_records(filas)
    assert frame["demand"].tolist()[:2] == [556.0, 546.0]
    assert pd.isna(frame["demand"].iloc[2])
    assert frame["quality"].tolist() == ["observed", "observed", "missing"]


def test_unknown_unit_is_treated_as_missing_not_guessed():
    filas = [{
        "station_id": "02300", "observed_at": "2026-09-20T12:15:00Z", "schema_version": 2,
        "measurement": {"value": "5.46", "unit": "furlongs", "quality": "observed"},
    }]
    assert pd.isna(parse_stream_records(filas)["demand"].iloc[0])


def test_scaled_units_are_converted_to_passengers():
    filas = [{
        "station_id": "02300", "observed_at": "2026-09-20T12:15:00Z", "schema_version": 2,
        "measurement": {"value": "5.46", "unit": "hundreds_of_passengers", "quality": "observed"},
    }]
    assert parse_stream_records(filas)["demand"].iloc[0] == 546.0


def test_fill_gaps_imputes_and_flags_missing_values():
    index = pd.date_range("2026-01-01", periods=6, freq="15min", tz="UTC")
    frame = pd.DataFrame({
        "station_id": "A",
        "observed_at": index.delete(2),            # fila ausente por completo
        "demand": [10.0, 20.0, 40.0, None, 60.0],  # y un valor nulo
    })
    salida = fill_gaps(frame)
    assert len(salida) == 6
    assert not salida["demand"].isna().any()
    assert salida["imputed"].tolist() == [False, False, True, False, True, False]
    assert salida["demand"].iloc[2] == 30.0


def test_fill_gaps_carries_last_value_forward_at_the_end():
    index = pd.date_range("2026-01-01", periods=3, freq="15min", tz="UTC")
    frame = pd.DataFrame({"station_id": "A", "observed_at": index, "demand": [10.0, 20.0, None]})
    salida = fill_gaps(frame)
    assert salida["demand"].iloc[-1] == 20.0
