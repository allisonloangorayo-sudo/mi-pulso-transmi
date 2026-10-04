import pandas as pd
import pytest

from src import predict


def _ciclo(minutos_para_cerrar: float) -> dict:
    cierre = pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=minutos_para_cerrar)
    return {
        "cycle_id": "cyc_test", "data_cutoff": "2026-09-21T07:00:00Z", "closes_at": cierre.isoformat(),
        "targets": [{"station_id": "02300", "target_at": "2026-09-21T07:15:00Z", "horizon_minutes": 15}],
    }


@pytest.fixture
def entorno(monkeypatch):
    enviados = []

    def modelo_roto():
        raise RuntimeError("Supabase Storage caído")

    def lote_respaldo(cycle):
        return pd.DataFrame({
            "station_id": ["02300"], "target_at": [pd.Timestamp("2026-09-21T07:15:00Z")],
            "horizon": [1], "value": [100.0],
        })

    def enviar(cycle_id, data_cutoff, batch, key, champion):
        enviados.append(champion["version"])
        return {"submission_id": "sub_1", "predictions_received": 1, "expected_predictions": 1,
                "status": "accepted"}

    monkeypatch.setattr(predict, "load_champion", modelo_roto)
    monkeypatch.setattr(predict, "fallback_batch", lote_respaldo)
    monkeypatch.setattr(predict, "submit_predictions", enviar)
    monkeypatch.setattr(predict, "save_receipt", lambda *a, **k: None)
    return monkeypatch, enviados


def test_modelo_roto_sin_envio_previo_entrega_respaldo_y_alerta(entorno):
    monkeypatch, enviados = entorno
    monkeypatch.setattr(predict, "get_current_cycle", lambda: _ciclo(20))
    monkeypatch.setattr(predict, "cycle_has_receipt", lambda cycle_id: False)
    with pytest.raises(predict.FallbackSubmitted):
        predict.run()
    assert enviados == ["respaldo-expertos-simples"]


def test_no_pisa_un_envio_que_ya_existe(entorno):
    monkeypatch, enviados = entorno
    monkeypatch.setattr(predict, "get_current_cycle", lambda: _ciclo(5))
    monkeypatch.setattr(predict, "cycle_has_receipt", lambda cycle_id: True)
    with pytest.raises(RuntimeError, match="Supabase Storage"):
        predict.run()
    assert enviados == []


def test_sin_saber_si_hubo_envio_espera_a_los_ultimos_minutos(entorno):
    monkeypatch, enviados = entorno
    monkeypatch.setattr(predict, "cycle_has_receipt", lambda cycle_id: None)
    monkeypatch.setattr(predict, "get_current_cycle", lambda: _ciclo(20))
    with pytest.raises(RuntimeError, match="Supabase Storage"):
        predict.run()
    assert enviados == []

    monkeypatch.setattr(predict, "get_current_cycle", lambda: _ciclo(8))
    with pytest.raises(predict.FallbackSubmitted):
        predict.run()
    assert enviados == ["respaldo-expertos-simples"]
