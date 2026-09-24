"""Experimento v3 — empujar por encima de 87%.

Parte de la mejor variante del v2 (modelo por horizonte + estación = 85.83%)
y prueba: features más ricas, modelos por estación, y ensamble.
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
    r = request_with_retry("GET", f"{BASE}/v1/stream/observations", params={"limit": 5000},
                           headers={"Authorization": f"Bearer {KEY}"}, timeout=45).json()
    st = pd.DataFrame(r["data"])
    st["observed_at"] = pd.to_datetime(st["observed_at"], utc=True)
    st["station_id"] = st["station_id"].astype("string")
    df = pd.concat([static, st[["station_id", "observed_at", "demand"]]], ignore_index=True)
    return (df.drop_duplicates(subset=["station_id", "observed_at"])
              .sort_values(["station_id", "observed_at"]).reset_index(drop=True))


def design(df, h, rich=False, station=True):
    d = df.sort_values(["station_id", "observed_at"]).copy()
    g = d.groupby("station_id")["demand"]
    offsets = {"s0": h, "s1": h + 1, "s2": h + 2, "s4": h + 4,
               "sday": h + DAY, "sweek": h + WEEK}
    if rich:
        offsets.update({"s3": h + 3, "s8": h + 8, "sday2": h + 2 * DAY,
                        "sweek2": h + 2 * WEEK, "sdayprev": h + DAY - 1,
                        "sdaynext": h + DAY + 1})
    for name, off in offsets.items():
        d[name] = g.shift(off)
    base = g.shift(h)
    d["roll4"] = base.rolling(4).mean()
    d["roll16"] = base.rolling(16).mean()
    feats = list(offsets) + ["roll4", "roll16"]
    if rich:
        d["roll96"] = base.rolling(DAY).mean()
        d["std16"] = base.rolling(16).std()
        d["trend"] = d["s0"] - d["s4"]
        dayl = g.shift(h + DAY)
        d["day_delta"] = d["s0"] - dayl
        feats += ["roll96", "std16", "trend", "day_delta"]
    d["hour"] = d["observed_at"].dt.hour
    d["minute"] = d["observed_at"].dt.minute
    d["dow"] = d["observed_at"].dt.dayofweek
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    feats += ["hour", "minute", "dow", "is_weekend"]
    if rich:
        d["tod"] = d["hour"] * 4 + d["minute"] // 15
        feats += ["tod"]
    if station:
        d["station_code"] = d["station_id"].astype("category").cat.codes
        feats += ["station_code"]
    return d, feats


def accuracy(frame):
    per = frame.groupby("station_id").apply(
        lambda g: 100 * max(0.0, 1 - (g["demand"] - g["pred"]).abs().sum() / g["demand"].sum()),
        include_groups=False)
    return float(per.mean())


def run(df, *, rich=False, per_station=False, ensemble=False, deep=False, station=True):
    val_start = df["observed_at"].max() - pd.Timedelta(days=VAL_DAYS)
    preds = []
    for h in HORIZONS:
        d, feats = design(df, h, rich=rich, station=station)
        d = d.dropna(subset=feats + ["demand"])
        tr, va = d[d["observed_at"] < val_start], d[d["observed_at"] >= val_start]

        def fit_predict(tr_s, va_s):
            if len(tr_s) == 0 or len(va_s) == 0:
                return None
            rf = RandomForestRegressor(
                n_estimators=300 if deep else 150,
                max_depth=None if deep else 14,
                min_samples_leaf=2 if deep else 5,
                n_jobs=-1, random_state=42)
            rf.fit(tr_s[feats], tr_s["demand"])
            p = rf.predict(va_s[feats])
            if ensemble:
                hg = HistGradientBoostingRegressor(loss="absolute_error", max_iter=500,
                                                   learning_rate=0.05, random_state=42)
                hg.fit(tr_s[feats], tr_s["demand"])
                p = 0.5 * p + 0.5 * hg.predict(va_s[feats])
            return np.clip(p, 0, None)

        if per_station:
            for sid in d["station_id"].unique():
                trs, vas = tr[tr["station_id"] == sid], va[va["station_id"] == sid]
                p = fit_predict(trs, vas)
                if p is not None:
                    preds.append(vas.assign(pred=p, horizon=h)[
                        ["station_id", "observed_at", "demand", "pred", "horizon"]])
        else:
            p = fit_predict(tr, va)
            preds.append(va.assign(pred=p, horizon=h)[
                ["station_id", "observed_at", "demand", "pred", "horizon"]])

    out = pd.concat(preds, ignore_index=True)
    por_h = out.groupby("horizon").apply(accuracy, include_groups=False).round(2).to_dict()
    return accuracy(out), len(out), por_h


def main():
    t0 = time.time()
    df = load_all()
    print(f"datos: {len(df):,} obs hasta {df['observed_at'].max()}\n")
    pruebas = [
        ("D. base v2 (por horizonte + estación)", dict()),
        ("G. + features ricas",                   dict(rich=True)),
        ("H. + RF más profundo",                  dict(rich=True, deep=True)),
        ("I. modelos por estación",               dict(rich=True, per_station=True)),
        ("J. ensamble RF+boosting",               dict(rich=True, deep=True, ensemble=True)),
    ]
    for nombre, kw in pruebas:
        acc, n, por_h = run(df, **kw)
        print(f"{nombre:40s} {acc:6.2f}%  n={n:,}  por horizonte: {por_h}")
    print(f"\ntiempo: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
