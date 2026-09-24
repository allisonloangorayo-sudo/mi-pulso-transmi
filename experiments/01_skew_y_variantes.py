"""Experimento v2 — comparación justa (mismo conjunto de evaluación en todas
las variantes, los 4 horizontes siempre).

Simula el servicio real: para un target t y horizonte h, solo se usan datos
<= t-h. La variante "actual" reproduce el sesgo train/serve de producción:
el modelo se entrena con alineación h=1 (lag más reciente = 15 min antes del
target) pero se le alimentan features de alineación h (lag más reciente = h
pasos antes), que es exactamente lo que hace build_features_as_of hoy.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

sys.path.insert(0, "/Users/angie/Desktop/mi-pulso-transmi")
load_dotenv("/Users/angie/Desktop/mi-pulso-transmi/.env")
from pulso_transmi import PulsoTransmiClient
from src.http_utils import request_with_retry

BASE, KEY = os.environ["PULSO_API_URL"], os.environ["PULSO_API_KEY"]
DAY, WEEK = 96, 672
HORIZONS = [1, 2, 3, 4]
VAL_DAYS = 5


def load_all():
    with PulsoTransmiClient() as c:
        static = c.observations_dataframe()
        ctx = c.context_dataframe()
    r = request_with_retry("GET", f"{BASE}/v1/stream/observations", params={"limit": 5000},
                           headers={"Authorization": f"Bearer {KEY}"}, timeout=45).json()
    st = pd.DataFrame(r["data"])
    st["observed_at"] = pd.to_datetime(st["observed_at"], utc=True)
    st["station_id"] = st["station_id"].astype("string")
    df = pd.concat([static, st[["station_id", "observed_at", "demand"]]], ignore_index=True)
    df = df.drop_duplicates(subset=["station_id", "observed_at"])
    return df.sort_values(["station_id", "observed_at"]).reset_index(drop=True), ctx


SLOTS = ["s0", "s1", "s2", "s4", "sday", "sweek", "roll4", "roll16"]
CAL = ["hour", "minute", "dow", "is_weekend"]


def design(df, h, ctx=None, station=False):
    """Features en 'slots' semánticos: s0 = lag más reciente disponible (h pasos
    antes del target), s1 = el anterior, etc. Así una matriz de alineación h se
    puede alimentar a un modelo entrenado con otra alineación (eso es el skew)."""
    d = df.sort_values(["station_id", "observed_at"]).copy()
    g = d.groupby("station_id")["demand"]
    for name, off in zip(SLOTS[:6], [h, h + 1, h + 2, h + 4, h + DAY, h + WEEK]):
        d[name] = g.shift(off)
    base = g.shift(h)
    d["roll4"] = base.rolling(4).mean()
    d["roll16"] = base.rolling(16).mean()
    d["hour"] = d["observed_at"].dt.hour
    d["minute"] = d["observed_at"].dt.minute
    d["dow"] = d["observed_at"].dt.dayofweek
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    feats = SLOTS + CAL
    if station:
        d["station_code"] = d["station_id"].astype("category").cat.codes
        feats = feats + ["station_code"]
    if ctx is not None and not ctx.empty:
        cols = [c for c in ctx.columns if c != "observed_at" and pd.api.types.is_numeric_dtype(ctx[c])]
        d = d.merge(ctx[["observed_at"] + cols], on="observed_at", how="left")
        for c in cols:  # imputar para no perder filas del periodo sin contexto
            d[c] = d[c].fillna(d[c].median())
        feats = feats + cols
    return d, feats


def accuracy(frame):
    per = frame.groupby("station_id").apply(
        lambda g: 100 * max(0.0, 1 - (g["demand"] - g["pred"]).abs().sum() / g["demand"].sum()),
        include_groups=False)
    return float(per.mean())


def make_model(kind):
    if kind == "rf":
        return RandomForestRegressor(n_estimators=150, max_depth=14, min_samples_leaf=5,
                                     n_jobs=-1, random_state=42)
    return HistGradientBoostingRegressor(loss="absolute_error", max_iter=400,
                                         learning_rate=0.06, random_state=42)


def run(df, ctx, *, per_horizon, station, use_ctx, kind, train_df=None):
    """Devuelve accuracy oficial evaluando SIEMPRE los 4 horizontes."""
    train_src = df if train_df is None else train_df
    val_start = df["observed_at"].max() - pd.Timedelta(days=VAL_DAYS)
    c = ctx if use_ctx else None
    preds = []

    if per_horizon:
        for h in HORIZONS:
            dtr, feats = design(train_src, h, c, station)
            dtr = dtr.dropna(subset=feats + ["demand"])
            m = make_model(kind)
            m.fit(dtr[dtr["observed_at"] < val_start][feats],
                  dtr[dtr["observed_at"] < val_start]["demand"])
            dva, _ = design(df, h, c, station)
            dva = dva.dropna(subset=feats + ["demand"])
            dva = dva[dva["observed_at"] >= val_start]
            preds.append(dva.assign(pred=np.clip(m.predict(dva[feats]), 0, None),
                                    horizon=h)[["station_id", "observed_at", "demand", "pred", "horizon"]])
    else:
        # un solo modelo entrenado con alineación h=1, servido en las 4 -> skew
        dtr, feats = design(train_src, 1, c, station)
        dtr = dtr.dropna(subset=feats + ["demand"])
        m = make_model(kind)
        m.fit(dtr[dtr["observed_at"] < val_start][feats],
              dtr[dtr["observed_at"] < val_start]["demand"])
        for h in HORIZONS:
            dva, _ = design(df, h, c, station)
            dva = dva.dropna(subset=feats + ["demand"])
            dva = dva[dva["observed_at"] >= val_start]
            preds.append(dva.assign(pred=np.clip(m.predict(dva[feats]), 0, None),
                                    horizon=h)[["station_id", "observed_at", "demand", "pred", "horizon"]])

    out = pd.concat(preds, ignore_index=True)
    por_h = out.groupby("horizon").apply(
        lambda g: accuracy(g), include_groups=False).round(2).to_dict()
    return accuracy(out), len(out), por_h


def main():
    t0 = time.time()
    df, ctx = load_all()
    static_cut = pd.Timestamp("2026-09-09T04:45:00Z")
    df_static_train = df[df["observed_at"] <= static_cut]
    print(f"datos: {len(df):,} obs hasta {df['observed_at'].max()}")
    print(f"validación: últimos {VAL_DAYS} días | contexto: {len(ctx):,} filas\n")

    pruebas = [
        ("A. producción hoy (RF, sin stream, 1 modelo)",
         dict(per_horizon=False, station=False, use_ctx=False, kind="rf", train_df=df_static_train)),
        ("B. + entrenar con stream",
         dict(per_horizon=False, station=False, use_ctx=False, kind="rf")),
        ("C. + un modelo por horizonte (sin skew)",
         dict(per_horizon=True, station=False, use_ctx=False, kind="rf")),
        ("D. + estación como feature",
         dict(per_horizon=True, station=True, use_ctx=False, kind="rf")),
        ("E. + contexto (clima/eventos)",
         dict(per_horizon=True, station=True, use_ctx=True, kind="rf")),
        ("F. + boosting con pérdida MAE",
         dict(per_horizon=True, station=True, use_ctx=True, kind="hgb")),
    ]
    for nombre, kw in pruebas:
        acc, n, por_h = run(df, ctx, **kw)
        print(f"{nombre:44s} {acc:6.2f}%  n={n:,}  por horizonte: {por_h}")
    print(f"\ntiempo: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
