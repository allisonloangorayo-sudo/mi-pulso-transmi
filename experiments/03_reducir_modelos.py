"""Experimento v5 — reducir el número de modelos sin perder precisión.

Idea: en vez de 48 modelos (12 estaciones x 4 horizontes), apilar los 4
horizontes en un solo dataset por estación y pasar `horizon` como feature.
Quedan 12 modelos. Las features son "slots" (s0 = lag más reciente
disponible), así que apilar horizontes es coherente.
"""
from __future__ import annotations
import os, sys, time, tempfile, joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

sys.path.insert(0, "/Users/angie/Desktop/mi-pulso-transmi")
load_dotenv("/Users/angie/Desktop/mi-pulso-transmi/.env")
sys.path.insert(0, "/private/tmp/claude-501/-Users-angie-Desktop-Imagenes/e761740c-933f-4acf-84f1-bf1f9e878d7f/scratchpad")
from experimento3 import load_all, design, accuracy, HORIZONS, VAL_DAYS


def stacked(df, rich=True):
    """Apila los 4 horizontes con `horizon` como feature."""
    partes = []
    for h in HORIZONS:
        d, feats = design(df, h, rich=rich, station=False)
        d = d.dropna(subset=feats + ["demand"]).copy()
        d["horizon"] = h
        partes.append(d)
    return pd.concat(partes, ignore_index=True), feats + ["horizon"]


def run(df, *, kind, rich=True, medir_tamano=False):
    d, feats = stacked(df, rich)
    val_start = df["observed_at"].max() - pd.Timedelta(days=VAL_DAYS)
    tr, va = d[d["observed_at"] < val_start], d[d["observed_at"] >= val_start]
    preds, total_mb = [], 0.0
    for sid in d["station_id"].unique():
        trs, vas = tr[tr["station_id"] == sid], va[va["station_id"] == sid]
        if len(trs) == 0 or len(vas) == 0:
            continue
        if kind == "rf":
            m = RandomForestRegressor(n_estimators=150, max_depth=16, min_samples_leaf=5,
                                      n_jobs=-1, random_state=42)
        else:
            m = HistGradientBoostingRegressor(loss="absolute_error", max_iter=600,
                                              learning_rate=0.05, random_state=42)
        m.fit(trs[feats], trs["demand"])
        if medir_tamano:
            f = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
            joblib.dump(m, f.name)
            total_mb += os.path.getsize(f.name) / 1e6
            os.unlink(f.name)
        preds.append(vas.assign(pred=np.clip(m.predict(vas[feats]), 0, None))[
            ["station_id", "observed_at", "demand", "pred", "horizon"]])
    out = pd.concat(preds, ignore_index=True)
    por_h = out.groupby("horizon").apply(accuracy, include_groups=False).round(2).to_dict()
    return accuracy(out), por_h, total_mb


def main():
    t0 = time.time()
    df = load_all()
    print(f"datos: {len(df):,}\n")
    for nombre, kind in [("O. 12 modelos (horizonte=feature), RF", "rf"),
                         ("P. 12 modelos (horizonte=feature), HGB MAE", "hgb")]:
        acc, por_h, mb = run(df, kind=kind, medir_tamano=True)
        print(f"{nombre:46s} {acc:6.2f}%  {por_h}  artefactos: {mb:.1f} MB")
    print(f"\ntiempo: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
