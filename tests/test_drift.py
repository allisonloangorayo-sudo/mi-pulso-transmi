import pandas as pd

from src.drift import DROP_THRESHOLD_PCT, accuracy_trend


def _evaluaciones(errores_por_ciclo, version="v1"):
    """12 ciclos con 2 estaciones; demanda real 100 y el error indicado."""
    filas = []
    for i, error in enumerate(errores_por_ciclo):
        for estacion in ("A", "B"):
            filas.append({
                "cycle_id": f"cyc_{i:03d}", "station_id": estacion, "model_version": version,
                "real": 100.0, "predicted": 100.0 + error, "abs_error": abs(error),
                "evaluated_at": pd.Timestamp("2026-10-04", tz="UTC") + pd.Timedelta(hours=i),
            })
    return pd.DataFrame(filas)


def test_detecta_mejora():
    tendencia = accuracy_trend(_evaluaciones([20] * 6 + [10] * 6))
    assert tendencia["accuracy_prev_6"] == 80.0
    assert tendencia["accuracy_last_6"] == 90.0
    assert tendencia["verdict"] == "MEJORA"
    assert tendencia["above_floor"]


def test_caida_de_dos_puntos_es_empeora():
    tendencia = accuracy_trend(_evaluaciones([10] * 6 + [10 + DROP_THRESHOLD_PCT] * 6))
    assert tendencia["verdict"] == "EMPEORA"


def test_caida_menor_al_umbral_es_estable():
    tendencia = accuracy_trend(_evaluaciones([10] * 6 + [11] * 6))
    assert tendencia["verdict"] == "ESTABLE"


def test_resume_cada_version_de_modelo():
    df = pd.concat([_evaluaciones([20] * 6, "v1"), _evaluaciones([10] * 6, "v2").assign(
        cycle_id=lambda d: d["cycle_id"].str.replace("cyc_", "cyc_z"),
        evaluated_at=lambda d: d["evaluated_at"] + pd.Timedelta(days=1),
    )])
    versiones = {r["model_version"]: r["accuracy"] for r in accuracy_trend(df)["by_model_version"]}
    assert versiones == {"v1": 80.0, "v2": 90.0}
