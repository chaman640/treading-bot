"""
Forecast engine ki imaandaari ke test — `python test_forecast.py`

1. RANDOM WALK: price me koi pattern nahi. Agar model yahan edge dikhaye, to
   kahin future ka data features me ghus raha hai (look-ahead / leakage).
2. PLANTED SIGNAL: har din ek chhupa hua drift daala gaya hai jo din ke shuru se
   dikhne lagta hai. Model ko ye pakadna chahiye — warna pipeline andha hai.
"""

import os
import tempfile
from datetime import datetime

import numpy as np

os.environ.setdefault("PRED_LOG_FILE", os.path.join(tempfile.gettempdir(), "test_pred.jsonl"))

import app                      # noqa: E402
import forecast as fc           # noqa: E402

COST = 0.001


def synth(drift_per_candle=0.0, days=60, seed=1):
    rng = np.random.default_rng(seed)
    rows, price = [], 1000.0
    day = datetime(2026, 6, 1, 9, 15, tzinfo=fc.IST)
    made = 0
    while made < days:
        if day.weekday() < 5:
            drift = rng.choice([-1, 1]) * drift_per_candle
            for k in range(75):
                t = int(day.timestamp()) + k * 300
                o = price
                c = o * (1 + drift + rng.normal(0, 0.0012))
                h = max(o, c) * (1 + abs(rng.normal(0, 0.0004)))
                l = min(o, c) * (1 - abs(rng.normal(0, 0.0004)))
                rows.append({"t": t, "o": o, "h": h, "l": l, "c": c,
                             "v": int(rng.integers(5e4, 2e5))})
                price = c
            made += 1
        day = day.replace(day=day.day) + (datetime(2026, 1, 2) - datetime(2026, 1, 1))
    return rows


def run(rows):
    d = app.build(rows)
    return fc.forecast(rows, d, 5, app.norm_score, app.build, COST)


def test_random_walk_has_no_edge():
    for seed in (1, 2, 3):
        f = run(synth(0.0, seed=seed))
        q = f["quality"]
        print(f"random seed={seed}: skill {q['logloss_skill_pct']:+.2f}% "
              f"dir {q['dir_sign_pct']} trades {q['trades']}")
        assert q["logloss_skill_pct"] < 1.0, "random data par skill — leakage?"
        assert f["signal"] not in ("LONG", "SHORT"), "random data par trade signal"


def test_planted_signal_is_found():
    f = run(synth(0.0006, seed=7))
    q = f["quality"]
    print(f"planted: skill {q['logloss_skill_pct']:+.2f}% dir {q['dir_sign_pct']} trades {q['trades']}")
    assert q["logloss_skill_pct"] > 2.0, "chhupa pattern model ko nahi dikha"
    assert q["dir_sign_pct"] and q["dir_sign_pct"] > 60


def test_no_future_in_features():
    """Aakhri candle badalne se pichhle candles ke features nahi badalne chahiye."""
    rows = synth(0.0, seed=4)
    d1 = app.build(rows)
    X1, _ = fc.build_features(rows, d1, 5, 6, app.norm_score, app.build)
    rows2 = [dict(r) for r in rows]
    rows2[-1]["c"] *= 1.05
    rows2[-1]["h"] = max(rows2[-1]["h"], rows2[-1]["c"])
    d2 = app.build(rows2)
    X2, _ = fc.build_features(rows2, d2, 5, 6, app.norm_score, app.build)
    assert np.allclose(X1[:-1], X2[:-1]), "future candle ne purane features badal diye"


if __name__ == "__main__":
    test_no_future_in_features()
    print("no-future: OK")
    test_random_walk_has_no_edge()
    test_planted_signal_is_found()
    print("sab test pass")
