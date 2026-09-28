"""
Forecast engine — "agle 30 minute me price kidhar jayega?"

app.py ka signal haath se likhe weights ka vote hai. Ye module usse alag, asli
prediction setup hai:

  1. LABEL   — har purane candle i par: agle candle ke OPEN par ghuso, 30 min baad
               ke CLOSE par niklo. Return cost se bada upar = UP, neeche = DOWN,
               beech me = FLAT (move cost bhi nahi nikalta).
  2. FEATURES — base timeframe ke indicators + 15m/1h/1din ka context (base candles
               ko resample karke, sirf POORE ban chuke candle) + market regime +
               price action + Nifty ke against relative strength.
  3. MODEL   — multinomial logistic regression (numpy). Weights data se seekhe
               jaate hain, haath se nahi.
  4. WALK-FORWARD — pehla aadha data train, baaki 5 tukdon me: har tukde se pehle
               ke data par dobara train, phir us tukde par test. Train aur test ke
               beech purge gap, taaki label test period me na jhaanke.
  5. CALIBRATION — "model ne 60% kaha" ka matlab tabhi 60% hai jab out-of-sample
               me waise predictions sach me ~60% baar sahi nikle. Wahi napa jata hai.

Jo number dikhta hai wo out-of-sample hai. Agar model baseline (bas purani
frequency bol dena) se behtar nahi hai, to seedha "koi edge nahi" dikhta hai.
"""

import json
import math
import os
import threading
from datetime import datetime, timezone, timedelta

import numpy as np

IST = timezone(timedelta(hours=5, minutes=30))
HORIZON_MIN = 30
CLASSES = ("DOWN", "FLAT", "UP")          # y = 0, 1, 2
WARM = 60                                  # EMA50/ADX ko itne candle chahiye
MIN_TRAIN = 250
MIN_SAMPLES = 400
FOLDS = 5
# Regularization. 0.05 par model noise yaad kar leta tha (out-of-sample log-loss
# baseline se 3-7% kharab). 2.0 par wo baseline ke barabar aa jata hai — asli
# signal ho to dikhega, warna model khud "pata nahi" bolega.
L2 = 2.0
SESSION_START = 9 * 60 + 15
SESSION_LEN = 375
LOG_FILE = os.getenv("PRED_LOG_FILE", "predictions.jsonl")
CAL_BINS = (0.0, 0.40, 0.50, 0.60, 1.01)


def horizon_candles(mins):
    """30 min me kitne candle. 60m/1d par 1 candle (tab horizon bhi wahi hota hai)."""
    return max(1, HORIZON_MIN // mins) if mins < HORIZON_MIN else 1


def _dt(t):
    return datetime.fromtimestamp(t, IST)


def _min_of_day(dt):
    return dt.hour * 60 + dt.minute - SESSION_START


# ────────────────────────  MULTI-TIMEFRAME  ────────────────────────

def resample(rows, M, daily=False):
    """Base candles ko M-minute (ya din) ke candle me jodo. 9:15 se aligned."""
    out, key_prev = [], None
    for i, r in enumerate(rows):
        dt = _dt(r["t"])
        key = dt.date() if daily else (dt.date(), _min_of_day(dt) // M)
        if key != key_prev:
            out.append({"t": r["t"], "o": r["o"], "h": r["h"], "l": r["l"],
                        "c": r["c"], "v": r["v"], "last": i})
            key_prev = key
        else:
            b = out[-1]
            b["h"] = max(b["h"], r["h"]); b["l"] = min(b["l"], r["l"])
            b["c"] = r["c"]; b["v"] += r["v"]; b["last"] = i
    return out


def usable_map(rows, buckets, mins, M, daily):
    """
    base index i -> sabse naya higher-TF candle jo i ke close tak POORA ban chuka ho.
    Adhoora higher-TF candle kabhi use nahi hota — warna training me wo future
    dekh leta jo live me nahi dikhta.
    """
    n = len(rows)
    done_at = []
    for bi, b in enumerate(buckets):
        if bi < len(buckets) - 1:
            done_at.append(b["last"])
            continue
        m = _min_of_day(_dt(rows[b["last"]]["t"]))
        end = m + mins
        limit = SESSION_LEN if daily else min((m // M + 1) * M, SESSION_LEN)
        done_at.append(b["last"] if end >= limit else n)
    out, bi = [-1] * n, -1
    for i in range(n):
        while bi + 1 < len(buckets) and done_at[bi + 1] <= i:
            bi += 1
        out[i] = bi
    return out


def _tf_feats(dt, b, c_now):
    if b < 0:
        return [0.0] * 5
    atr = dt["atr"][b]
    e9, e21, e50 = dt["e9"][b], dt["e21"][b], dt["e50"][b]
    if not atr or e21 is None:
        return [0.0] * 5
    stack = 0.0
    if None not in (e9, e50):
        stack = 1.0 if e9 > e21 > e50 else -1.0 if e9 < e21 < e50 else 0.0
    rsi = dt["rsi"][b]
    mh = dt["mhist"][b]
    return [(c_now - e21) / atr, stack,
            (rsi - 50) / 50 if rsi is not None else 0.0,
            mh / atr if mh is not None else 0.0,
            float(dt["st_dir"][b] or 0)]


# ────────────────────────────  REGIME  ────────────────────────────

def regime_at(d, i, vol_ratio):
    a, p, m = d["adx"][i], d["pdi"][i], d["mdi"][i]
    c, e50 = d["c"][i], d["e50"][i]
    if None in (a, p, m, e50):
        trend = "UNKNOWN"
    elif a >= 25 and p > m and c > e50:
        trend = "TRENDING_UP"
    elif a >= 25 and m > p and c < e50:
        trend = "TRENDING_DOWN"
    elif a < 20:
        trend = "RANGING"
    else:
        trend = "WEAK_TREND"
    vol = "HIGH_VOL" if vol_ratio > 1.4 else "LOW_VOL" if vol_ratio < 0.75 else "NORMAL_VOL"
    event = None
    if i >= 20:
        if c > max(d["h"][i - 20:i]):
            event = "BREAKOUT_UP"
        elif c < min(d["l"][i - 20:i]):
            event = "BREAKOUT_DOWN"
    sd, sp = d["st_dir"][i], d["st_dir"][i - 1] if i else None
    if event is None and sd and sp and sd != sp:
        event = "REVERSAL_UP" if sd == 1 else "REVERSAL_DOWN"
    return trend, vol, event


# ───────────────────────────  FEATURES  ───────────────────────────

TF_NAMES = ("15m", "1h", "1din")
TF_PARTS = (("trend", "close vs EMA21"), ("stack", "EMA stack"), ("rsi", "RSI"),
            ("macd", "MACD hist"), ("st", "Supertrend"))

LABELS = {
    "mom_1": "Pichhle candle ka move (ATR)",
    "mom_h": "Pichhle 30 min ka move (ATR)",
    "mom_60": "Pichhle 1 ghante ka move (ATR)",
    "mom_180": "Pichhle 3 ghante ka move (ATR)",
    "day_move": "Aaj open se move (ATR)",
    "gap": "Aaj ka opening gap (ATR)",
    "vwap": "VWAP se doori (ATR)",
    "ema9_21": "EMA 9 − 21 (ATR)",
    "ema21_50": "EMA 21 − 50 (ATR)",
    "ema50": "Close − EMA50 (ATR)",
    "rsi": "RSI (−1..+1)",
    "macd": "MACD histogram (ATR)",
    "macd_slope": "MACD histogram ki dhalan",
    "bb_z": "Bollinger z-score",
    "adx": "ADX / 50",
    "di": "+DI − −DI",
    "st": "Supertrend direction",
    "vol": "Volume vs average (log)",
    "vol_regime": "Volatility vs pichhle 100 candle (log)",
    "body": "Candle body (range ka hissa)",
    "upper_wick": "Upar ki wick",
    "lower_wick": "Neeche ki wick",
    "day_pos": "Aaj ki range me position",
    "pdh": "Kal ke high se doori (ATR)",
    "pdl": "Kal ke low se doori (ATR)",
    "brk": "20-candle breakout",
    "st_flip": "Supertrend abhi palta",
    "tod": "Din ka samay",
    "tod2": "Din ka samay (curve)",
    "score": "Purana indicator score",
    "reg_up": "Regime: trending up",
    "reg_down": "Regime: trending down",
    "reg_range": "Regime: ranging",
    "rsi_x_range": "RSI × ranging",
    "rsi_x_trend": "RSI × trending",
    "bb_x_range": "Bollinger × ranging",
    "macd_x_trend": "MACD × trending",
    "mom_x_hivol": "30-min move × high volatility",
    "rs_h": "Nifty ke against 30 min (relative strength)",
    "rs_day": "Nifty ke against aaj",
    "bench_h": "Nifty ka 30 min move",
}
for _tf in TF_NAMES:
    for _p, _lab in TF_PARTS:
        LABELS[f"{_tf}_{_p}"] = f"{_tf}: {_lab}"

NAMES = list(LABELS.keys())


def build_features(rows, d, mins, h, score_fn, build_fn, bench=None):
    n = len(rows)
    daily = mins >= 375
    dts = [_dt(r["t"]) for r in rows]
    c, o, hi, lo = d["c"], d["o"], d["h"], d["l"]

    # din ke hisaab: aaj ka open/high/low (ab tak), kal ka close/high/low
    day_open, day_hi, day_lo = [0.0] * n, [0.0] * n, [0.0] * n
    prev_c, prev_h, prev_l = [None] * n, [None] * n, [None] * n
    cur_date, dopen, dh, dl = None, 0, 0, 0
    last_day = (None, None, None)
    for i in range(n):
        if daily:
            day_open[i], day_hi[i], day_lo[i] = o[i], hi[i], lo[i]
            if i:
                prev_c[i], prev_h[i], prev_l[i] = c[i - 1], hi[i - 1], lo[i - 1]
            continue
        if dts[i].date() != cur_date:
            if cur_date is not None:
                last_day = (c[i - 1], dh, dl)
            cur_date, dopen, dh, dl = dts[i].date(), o[i], hi[i], lo[i]
        dh, dl = max(dh, hi[i]), min(dl, lo[i])
        day_open[i], day_hi[i], day_lo[i] = dopen, dh, dl
        prev_c[i], prev_h[i], prev_l[i] = last_day

    # volatility regime: ATR% vs pichhle 100 candle ka median
    atrp = [(d["atr"][i] / c[i]) if d["atr"][i] else None for i in range(n)]
    vol_ratio = [1.0] * n
    for i in range(n):
        w = [x for x in atrp[max(0, i - 100):i + 1] if x]
        if atrp[i] and len(w) > 20:
            vol_ratio[i] = atrp[i] / sorted(w)[len(w) // 2]

    # higher timeframe context — sirf poore candle
    tf_ctx = []
    for name, M, is_day in (("15m", 15, False), ("1h", 60, False), ("1din", 375, True)):
        if M <= mins:
            tf_ctx.append(None)
            continue
        bk = resample(rows, M, is_day)
        tf_ctx.append((build_fn(bk), usable_map(rows, bk, mins, M, is_day)))

    bmap = {}
    if bench:
        bmap = {r["t"]: r for r in bench}

    k60 = max(1, math.ceil(60 / mins))
    k180 = max(1, math.ceil(180 / mins))

    X = np.zeros((n, len(NAMES)))
    regimes = [None] * n
    for i in range(WARM, n):
        atr = d["atr"][i]
        if not atr:
            continue
        f = {}
        f["mom_1"] = (c[i] - c[i - 1]) / atr
        f["mom_h"] = (c[i] - c[i - h]) / atr
        f["mom_60"] = (c[i] - c[i - k60]) / atr
        f["mom_180"] = (c[i] - c[i - k180]) / atr if i >= k180 else 0.0
        f["day_move"] = 0.0 if daily else (c[i] - day_open[i]) / atr
        f["gap"] = 0.0 if daily or prev_c[i] is None else (day_open[i] - prev_c[i]) / atr
        vw = d["vwap"][i]
        f["vwap"] = (c[i] - vw) / atr if vw and not daily else 0.0
        e9, e21, e50 = d["e9"][i], d["e21"][i], d["e50"][i]
        f["ema9_21"] = (e9 - e21) / atr
        f["ema21_50"] = (e21 - e50) / atr
        f["ema50"] = (c[i] - e50) / atr
        r = d["rsi"][i]
        f["rsi"] = (r - 50) / 50 if r is not None else 0.0
        mh, mhp = d["mhist"][i], d["mhist"][i - 1]
        f["macd"] = mh / atr if mh is not None else 0.0
        f["macd_slope"] = (mh - mhp) / atr if None not in (mh, mhp) else 0.0
        mid, sd = d["bb_mid"][i], d["bb_sd"][i]
        f["bb_z"] = (c[i] - mid) / sd if mid and sd else 0.0
        a, p, m = d["adx"][i], d["pdi"][i], d["mdi"][i]
        f["adx"] = a / 50 if a is not None else 0.0
        f["di"] = (p - m) / 50 if None not in (p, m) else 0.0
        f["st"] = float(d["st_dir"][i] or 0)
        va = d["vol_avg"][i]
        f["vol"] = math.log(d["v"][i] / va) if va and d["v"][i] > 0 else 0.0
        f["vol_regime"] = math.log(vol_ratio[i])
        rng = hi[i] - lo[i]
        if rng > 0:
            f["body"] = (c[i] - o[i]) / rng
            f["upper_wick"] = (hi[i] - max(o[i], c[i])) / rng
            f["lower_wick"] = (min(o[i], c[i]) - lo[i]) / rng
        drng = day_hi[i] - day_lo[i]
        f["day_pos"] = ((c[i] - day_lo[i]) / drng - 0.5) if drng > 0 and not daily else 0.0
        f["pdh"] = (c[i] - prev_h[i]) / atr if prev_h[i] is not None else 0.0
        f["pdl"] = (c[i] - prev_l[i]) / atr if prev_l[i] is not None else 0.0

        trend, vol, event = regime_at(d, i, vol_ratio[i])
        regimes[i] = (trend, vol, event)
        f["brk"] = 1.0 if event == "BREAKOUT_UP" else -1.0 if event == "BREAKOUT_DOWN" else 0.0
        f["st_flip"] = 1.0 if event == "REVERSAL_UP" else -1.0 if event == "REVERSAL_DOWN" else 0.0
        if not daily:
            tod = _min_of_day(dts[i]) / SESSION_LEN - 0.5
            f["tod"], f["tod2"] = tod, tod * tod
        f["score"] = score_fn(d, i)

        f["reg_up"] = 1.0 if trend == "TRENDING_UP" else 0.0
        f["reg_down"] = 1.0 if trend == "TRENDING_DOWN" else 0.0
        f["reg_range"] = 1.0 if trend == "RANGING" else 0.0
        trending = f["reg_up"] + f["reg_down"]
        f["rsi_x_range"] = f["rsi"] * f["reg_range"]
        f["rsi_x_trend"] = f["rsi"] * trending
        f["bb_x_range"] = f["bb_z"] * f["reg_range"]
        f["macd_x_trend"] = f["macd"] * trending
        f["mom_x_hivol"] = f["mom_h"] * (1.0 if vol == "HIGH_VOL" else 0.0)

        if bmap:
            atr_pct = atr / c[i] * 100
            b0, bh = bmap.get(rows[i]["t"]), bmap.get(rows[i - h]["t"])
            if b0 and bh and bh["c"]:
                b_ret = (b0["c"] / bh["c"] - 1) * 100
                s_ret = (c[i] / c[i - h] - 1) * 100
                f["bench_h"] = b_ret / atr_pct
                f["rs_h"] = (s_ret - b_ret) / atr_pct
            if not daily:
                j = i
                while j > 0 and dts[j - 1].date() == dts[i].date():
                    j -= 1
                bo = bmap.get(rows[j]["t"])
                if b0 and bo and bo["o"]:
                    f["rs_day"] = ((c[i] / day_open[i] - 1) - (b0["c"] / bo["o"] - 1)) * 100 / atr_pct

        for tname, ctx in zip(TF_NAMES, tf_ctx):
            if ctx is None:
                continue
            vals = _tf_feats(ctx[0], ctx[1][i], c[i])
            for (part, _), v in zip(TF_PARTS, vals):
                f[f"{tname}_{part}"] = v

        X[i] = [f.get(k, 0.0) for k in NAMES]
    return X, regimes


def make_labels(rows, d, h, band, daily):
    """fwd = agle candle ke open se, h candle baad ke close tak. Din paar nahi karta."""
    n = len(rows)
    fwd = np.full(n, np.nan)
    y = np.full(n, -1, dtype=int)
    for i in range(n - h):
        if not daily and _dt(rows[i + h]["t"]).date() != _dt(rows[i]["t"]).date():
            continue
        entry = d["o"][i + 1]
        if not entry:
            continue
        r = d["c"][i + h] / entry - 1
        fwd[i] = r
        y[i] = 2 if r > band else 0 if r < -band else 1
    return fwd, y


# ─────────────────────────────  MODEL  ─────────────────────────────

def fit(X, y, lam=None, iters=400):
    lam = L2 if lam is None else lam
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = np.clip((X - mu) / sd, -5, 5)
    n, k = Z.shape
    Y = np.eye(3)[y]
    b = np.log((Y.sum(0) + 1) / (n + 3))
    W = np.zeros((k, 3))
    # step = 1/L, L = softmax loss ka Lipschitz bound — isse GD kabhi phatta nahi
    L = 0.5 * np.linalg.eigvalsh(Z.T @ Z / n).max() + lam
    lr = 1.0 / L
    for _ in range(iters):
        P = _softmax(Z @ W + b)
        G = (P - Y) / n
        W -= lr * (Z.T @ G + lam * W)
        b -= lr * G.sum(0)
    return {"mu": mu, "sd": sd, "W": W, "b": b}


def _softmax(S):
    S = S - S.max(1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(1, keepdims=True)


def predict(model, X):
    Z = np.clip((X - model["mu"]) / model["sd"], -5, 5)
    return _softmax(Z @ model["W"] + model["b"])


def walk_forward(X, y, idx, h):
    """Expanding window. Har fold se pehle dobara train, purge gap ke saath."""
    m = len(y)
    start = max(MIN_TRAIN, m // 2)
    edges = np.linspace(start, m, FOLDS + 1).astype(int)
    P = np.full((m, 3), np.nan)
    B = np.full((m, 3), np.nan)
    for f in range(FOLDS):
        a, b = edges[f], edges[f + 1]
        if b <= a:
            continue
        # train sample ka label (i+h tak) test ke pehle candle se aage na jaye
        tr = np.where(idx + h <= idx[a])[0]
        if len(tr) < MIN_TRAIN:
            continue
        model = fit(X[tr], y[tr])
        P[a:b] = predict(model, X[a:b])
        prior = (np.bincount(y[tr], minlength=3) + 1) / (len(tr) + 3)
        B[a:b] = prior
    return P, B


def wilson(k, n, z=1.645):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def boot_ci(x, runs=2000, seed=42):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (runs, len(x)))].mean(1)
    return float(np.percentile(means, 5)), float(np.percentile(means, 95))


def trade_stats(pnls):
    """Cost ke baad ke trade returns ka poora hisaab — EV, CI, jeet/haar."""
    n = len(pnls)
    if n < 10:
        return {"n": n}
    a = np.asarray(pnls)
    lo, hi = boot_ci(a)
    wins, losses = a[a > 0], a[a <= 0]
    pw = len(wins) / n
    aw = float(wins.mean()) if len(wins) else 0.0
    al = float(losses.mean()) if len(losses) else 0.0
    return {"n": n,
            "avg_pct": round(float(a.mean()) * 100, 4),
            "sd_pct": round(float(a.std(ddof=1)) * 100, 4),
            "ci_low_pct": round(lo * 100, 4), "ci_high_pct": round(hi * 100, 4),
            "win_pct": round(pw * 100, 1),
            "avg_win_pct": round(aw * 100, 4), "avg_loss_pct": round(al * 100, 4),
            # EV = P(jeet)×avg jeet − P(haar)×|avg haar| — cost pehle hi kata hua hai
            "ev_formula": f"{pw:.2f}×{aw * 100:.3f}% − {1 - pw:.2f}×{abs(al) * 100:.3f}%"}


def nonoverlap(mask, pred, fwd, idx, h, cost):
    """Model ki UP/DOWN call par trade, ek waqt me ek hi — overlap se CI jhootha tight hota hai."""
    pnls, last_exit = [], -1
    for r in np.where(mask)[0]:
        if pred[r] == 1 or idx[r] <= last_exit:
            continue
        pnls.append((1 if pred[r] == 2 else -1) * fwd[r] - cost)
        last_exit = idx[r] + h
    return pnls


def oos_quality(P, B, y, fwd, idx, h, cost):
    ok = ~np.isnan(P[:, 0])
    P, B, y, fwd, idx = P[ok], B[ok], y[ok], fwd[ok], idx[ok]
    n = len(y)
    if n < 50:
        return None, None
    eps = 1e-9
    ll_m = -np.mean(np.log(P[np.arange(n), y] + eps))
    ll_b = -np.mean(np.log(B[np.arange(n), y] + eps))
    Y = np.eye(3)[y]
    br_m = np.mean(np.sum((P - Y) ** 2, 1))
    br_b = np.mean(np.sum((B - Y) ** 2, 1))
    pred = P.argmax(1)
    acc = float(np.mean(pred == y))
    base_acc = float(np.mean(B.argmax(1) == y))

    dir_mask = pred != 1
    dir_calls = int(dir_mask.sum())
    dir_hits = int((pred[dir_mask] == y[dir_mask]).sum())
    # direction sahi (FLAT label ko chhod kar): UP bola aur return > 0?
    sign_ok = int(np.sum(np.sign(fwd[dir_mask]) == np.where(pred[dir_mask] == 2, 1, -1)))

    # calibration: jis class ko model ne chuna, uski probability ke bins
    pmax = P.max(1)
    cal = []
    for lo_, hi_ in zip(CAL_BINS[:-1], CAL_BINS[1:]):
        mk = (pmax >= lo_) & (pmax < hi_)
        k = int(mk.sum())
        if k == 0:
            continue
        hits = int((pred[mk] == y[mk]).sum())
        cal.append({"range": f"{int(lo_ * 100)}–{min(100, int(hi_ * 100))}%", "n": k,
                    "said_pct": round(float(pmax[mk].mean()) * 100, 1),
                    "actual_pct": round(100 * hits / k, 1)})

    # trading: model ki har UP/DOWN call par, overlap nahi
    trades = trade_stats(nonoverlap(np.ones(n, bool), pred, fwd, idx, h, cost))

    q = {"oos_n": n,
         "logloss_skill_pct": round((1 - ll_m / ll_b) * 100, 2),
         "brier": round(float(br_m), 4), "base_brier": round(float(br_b), 4),
         "accuracy_pct": round(acc * 100, 1), "base_accuracy_pct": round(base_acc * 100, 1),
         "dir_calls": dir_calls,
         "dir_hit_pct": round(100 * dir_hits / dir_calls, 1) if dir_calls else None,
         "dir_sign_pct": round(100 * sign_ok / dir_calls, 1) if dir_calls else None,
         "class_mix_pct": {c: round(100 * float(np.mean(y == k)), 1)
                           for k, c in enumerate(CLASSES)},
         "calibration": cal, "trades": trades}
    return q, (P, y, fwd, pred, pmax, idx)


# ─────────────────────────  LEVELS / RISK  ─────────────────────────

def levels(rows, d, i, daily):
    c = d["c"][i]
    cands = []
    lb = d["h"][max(0, i - 50):i + 1], d["l"][max(0, i - 50):i + 1]
    cands += [("50-candle high", max(lb[0])), ("50-candle low", min(lb[1]))]
    if d["vwap"][i] and not daily:
        cands.append(("VWAP", d["vwap"][i]))
    if not daily:
        today = _dt(rows[i]["t"]).date()
        j = i
        while j > 0 and _dt(rows[j - 1]["t"]).date() == today:
            j -= 1
        cands += [("Aaj ka high", max(d["h"][j:i + 1])), ("Aaj ka low", min(d["l"][j:i + 1]))]
        if j > 0:
            pday = _dt(rows[j - 1]["t"]).date()
            k = j - 1
            while k > 0 and _dt(rows[k - 1]["t"]).date() == pday:
                k -= 1
            cands += [("Kal ka high", max(d["h"][k:j])), ("Kal ka low", min(d["l"][k:j]))]
    above = sorted([x for x in cands if x[1] > c * 1.0005], key=lambda x: x[1])
    below = sorted([x for x in cands if x[1] < c * 0.9995], key=lambda x: -x[1])
    res = {"name": above[0][0], "price": round(above[0][1], 2)} if above else None
    sup = {"name": below[0][0], "price": round(below[0][1], 2)} if below else None
    return res, sup


def check_quality(rows, mins):
    """Data me gadbad ho to prediction se pehle bata do."""
    warn = []
    n = len(rows)
    ts = [r["t"] for r in rows]
    if len(set(ts)) != n:
        warn.append(f"{n - len(set(ts))} candle duplicate timestamp ke saath")
    if any(ts[i] <= ts[i - 1] for i in range(1, n)):
        warn.append("Candles time order me nahi hain")
    bad = sum(1 for r in rows if r["h"] < max(r["o"], r["c"]) - 1e-9
              or r["l"] > min(r["o"], r["c"]) + 1e-9)
    if bad:
        warn.append(f"{bad} candle me high/low open-close se mel nahi khate")
    jumps = sum(1 for i in range(1, n) if rows[i - 1]["c"]
                and abs(rows[i]["c"] / rows[i - 1]["c"] - 1) > 0.15)
    if jumps:
        warn.append(f"{jumps} jagah ek candle me 15% se bada jump — split/bonus ya galat data?")
    if mins < 375:
        holes = 0
        for i in range(1, n):
            a, b = _dt(ts[i - 1]), _dt(ts[i])
            if a.date() == b.date() and ts[i] - ts[i - 1] > mins * 60 * 1.5:
                holes += 1
        if holes > n * 0.02:
            warn.append(f"Din ke beech {holes} jagah candles gayab hain")
    vols = [r["v"] for r in rows[-200:]]
    if vols and any(vols) and sum(1 for v in vols if v == 0) > len(vols) * 0.2:
        warn.append("Haal ke 20%+ candles me volume zero — volume features par bharosa kam")
    return warn


# ─────────────────────────────  FORECAST  ─────────────────────────────

def forecast(rows, d, mins, score_fn, build_fn, cost, bench=None,
             market_open=True, cost_ratio=None):
    """
    cost = poora round-trip (broker charges + taxes + slippage), fraction me.
    Ye hi UP/DOWN ki band bhi hai: isse chhota move trade karne layak nahi.
    """
    n = len(rows)
    daily = mins >= 375
    h = horizon_candles(mins)
    hmin = h * mins if not daily else None
    hlabel = f"agle {hmin} min" if hmin else "agla din"
    base = {"available": False, "horizon_candles": h, "horizon_label": hlabel,
            "horizon_min": hmin, "band_pct": round(cost * 100, 3)}

    X, regimes = build_features(rows, d, mins, h, score_fn, build_fn, bench)
    fwd, y = make_labels(rows, d, h, cost, daily)
    idx = np.array([i for i in range(WARM, n) if y[i] >= 0 and d["atr"][i]])
    if len(idx) < MIN_SAMPLES:
        base["reason"] = (f"Sirf {len(idx)} labelled examples — model train karne ke liye "
                          f"kam se kam {MIN_SAMPLES} chahiye. Chhota timeframe chuno (5 min).")
        return base

    Xl, yl, fl = X[idx], y[idx], fwd[idx]
    P, B = walk_forward(Xl, yl, idx, h)
    q, oos = oos_quality(P, B, yl, fl, idx, h, cost)
    if q is None:
        base["reason"] = "Walk-forward test ke liye data kam pada."
        return base

    model = fit(Xl, yl)
    i = n - 1
    p = predict(model, X[i:i + 1])[0]
    k = int(p.argmax())
    pred = CLASSES[k]

    # calibration: jab pehle model ne yahi class, isi probability range me boli thi
    Po, yo, fo, predo, pmaxo, ido = oos
    lo_, hi_ = next(((a, b) for a, b in zip(CAL_BINS[:-1], CAL_BINS[1:]) if a <= p[k] < b),
                    (0.0, 1.01))
    mk = (predo == k) & (pmaxo >= lo_) & (pmaxo < hi_)
    basis = "isi class + isi probability range"
    if mk.sum() < 20:
        mk = predo == k
        basis = "isi class (range me itne case nahi the)"
    cn = int(mk.sum())
    ch = int((yo[mk] == k).sum())
    wl, wh = wilson(ch, cn)
    base_rate = float(np.mean(yo == k))
    calib = {"n": cn, "hit_pct": round(100 * ch / cn, 1) if cn else None,
             "low_pct": round(wl * 100, 1), "high_pct": round(wh * 100, 1),
             "base_rate_pct": round(base_rate * 100, 1), "basis": basis}

    # EXPECTED VALUE: jab pehle model ne yahi direction, isi probability range me
    # boli thi, un trades ka asli nateeja (cost ke baad). Probability akele kaafi
    # nahi — 60% jeet bhi paisa dubo sakti hai agar haar jeet se badi ho.
    ev = {"n": 0}
    if k != 1:
        ev = trade_stats(nonoverlap(mk, predo, fo, ido, h, cost))
        ev["basis"] = basis

    # REGIME: har market-regime me model ki UP/DOWN calls ka alag nateeja
    reg_o = np.array([(regimes[j] or ("UNKNOWN",))[0] for j in ido])
    by_regime = []
    for rname in ("TRENDING_UP", "TRENDING_DOWN", "RANGING", "WEAK_TREND"):
        st = trade_stats(nonoverlap(reg_o == rname, predo, fo, ido, h, cost))
        by_regime.append({"regime": rname, **st})
    trend, vol, event = regimes[n - 1] or ("UNKNOWN", "NORMAL_VOL", None)
    cur_reg = next((r for r in by_regime if r["regime"] == trend), {"n": 0})

    # confidence — out-of-sample record se, model ki apni probability se nahi
    skill = q["logloss_skill_pct"] > 0
    ev_ok = ev.get("ci_low_pct") is not None and ev["ci_low_pct"] > 0
    reg_ok = cur_reg.get("avg_pct") is None or cur_reg["avg_pct"] > 0
    if skill and ev_ok and reg_ok and cn >= 20 and wl > base_rate:
        conf = "HIGH"
    elif skill and ev.get("avg_pct") is not None and ev["avg_pct"] > 0:
        conf = "MEDIUM"
    else:
        conf = "LOW"

    # expected range: jab model ne yahi class boli thi, asli return kahan gira
    fr = fo[predo == k] if (predo == k).sum() >= 30 else fl
    q10, q50, q90 = np.percentile(fr, [10, 50, 90])
    close = d["c"][i]
    rng = {"low": round(close * (1 + q10), 2), "mid": round(close * (1 + q50), 2),
           "high": round(close * (1 + q90), 2),
           "basis": ("jab model ne pehle yahi kaha tha, 80% baar price is range me raha"
                     if fr is not fl else "saare purane examples ka 80% range")}

    # why: kaunse features ne is class ko sabse zyada dhakela
    W = model["W"]
    z = np.clip((X[i] - model["mu"]) / model["sd"], -5, 5)
    push = z * (W[:, k] - np.delete(W, k, axis=1).mean(1))
    order = np.argsort(-push)
    why = [{"name": LABELS[NAMES[j]], "value": round(float(X[i, j]), 2)}
           for j in order[:5] if push[j] > 0.02]
    against = [{"name": LABELS[NAMES[j]], "value": round(float(X[i, j]), 2)}
               for j in order[::-1][:3] if push[j] < -0.02]

    # feature importance (standardized coef) aur redundancy — ek hi baat 4 baar to nahi?
    imp = np.abs(W).mean(1)
    importance = [{"name": LABELS[NAMES[j]], "weight": round(float(imp[j]), 3)}
                  for j in np.argsort(-imp)[:8]]
    live = [j for j in range(Xl.shape[1]) if Xl[:, j].std() > 1e-9]
    C = np.corrcoef(Xl[:, live].T)
    pairs = []
    for a_ in range(len(live)):
        for b_ in range(a_ + 1, len(live)):
            if abs(C[a_, b_]) > 0.85:
                pairs.append((abs(C[a_, b_]), live[a_], live[b_], C[a_, b_]))
    pairs.sort(reverse=True)
    redundant = [{"a": LABELS[NAMES[x]], "b": LABELS[NAMES[y_]], "corr": round(float(r), 2)}
                 for _, x, y_, r in pairs[:6]]

    res, sup = levels(rows, d, i, daily)
    atr = d["atr"][i] or close * 0.003
    side = 1 if pred == "UP" else -1 if pred == "DOWN" else 0

    risks = []
    if side == 1 and res and res["price"] < close + 1.5 * atr:
        risks.append(f"Resistance {res['name']} ₹{res['price']} target (1.5×ATR) se pehle aata hai")
    if side == -1 and sup and sup["price"] > close - 1.5 * atr:
        risks.append(f"Support {sup['name']} ₹{sup['price']} target (1.5×ATR) se pehle aata hai")
    if side == 1 and trend == "TRENDING_DOWN" or side == -1 and trend == "TRENDING_UP":
        risks.append("Prediction bade trend ke ulat hai")
    if vol == "HIGH_VOL":
        risks.append("Volatility normal se kaafi zyada — stop jaldi lag sakta hai")
    if cost_ratio and cost_ratio > 35:
        risks.append(f"Cost normal move ka {cost_ratio}% kha jati hai")
    if not market_open:
        risks.append("Market band hai — ye pichhle session ke aakhri candle par bana hai")

    if side == 0:
        sig, why_sig = "NO TRADE", "Model kehta hai move cost se chhota rahega (FLAT)."
    elif conf == "LOW":
        sig, why_sig = "NO TRADE", ("Jab model ne pehle aisa kaha tha, cost ke baad average "
                                    "nateeja zero se upar nahi tha — expected value negative.")
    elif conf == "MEDIUM":
        sig, why_sig = "WATCH", ("Expected value zero se upar dikhta hai par confidence "
                                 "interval zero ko chhoota hai. Bina paise ke track karo.")
    elif not market_open:
        sig, why_sig = "WATCH", "Record achha hai, par market band hai."
    else:
        sig = "LONG" if side == 1 else "SHORT"
        why_sig = ("Expected value ka 90% CI zero se upar, model baseline se behtar, "
                   "aur is regime me bhi nateeja positive.")

    last_t = rows[i]["t"]
    expires = _dt(last_t + 2 * mins * 60) if not daily else None

    base.update({
        "available": True,
        "predicted": pred,
        "probs": {c: round(float(p[j]) * 100, 1) for j, c in enumerate(CLASSES)},
        "calibrated": calib, "confidence": conf,
        "signal": sig, "signal_reason": why_sig,
        "range": rng, "why": why, "against": against,
        "regime": {"trend": trend, "volatility": vol, "event": event},
        "resistance": res, "support": sup, "risks": risks,
        "invalidation": round(close - side * atr, 2) if side else None,
        "quality": q, "samples": int(len(idx)), "features": len(NAMES),
        "ev": ev, "by_regime": by_regime,
        "importance": importance, "redundant": redundant,
        "t_last": last_t,
        "expires_at": expires.strftime("%I:%M %p") if expires else None,
    })
    return base


# ──────────────────────  PREDICTION LOG (live test)  ──────────────────────

_lock = threading.Lock()
_keys = None


def _read_log():
    if not os.path.exists(LOG_FILE):
        return []
    out = []
    with open(LOG_FILE) as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def log_prediction(source, symbol, interval, fc, price, extra=None):
    """Har naye candle ki prediction ek baar likho. Baad me asli nateeje se milayenge."""
    global _keys
    if not fc.get("available"):
        return
    key = f"{source}|{symbol}|{interval}|{fc['t_last']}"
    with _lock:
        if _keys is None:
            _keys = {f"{e['source']}|{e['symbol']}|{e['interval']}|{e['t_last']}"
                     for e in _read_log()}
        if key in _keys:
            return
        _keys.add(key)
        rec = {"source": source, "symbol": symbol, "interval": interval,
               "t_last": fc["t_last"], "h": fc["horizon_candles"], "price": price,
               "pred": fc["predicted"], "probs": fc["probs"], "signal": fc["signal"],
               "confidence": fc["confidence"], "band": fc["band_pct"] / 100,
               "logged_at": int(datetime.now(IST).timestamp()), **(extra or {})}
        with open(LOG_FILE, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


def resolve(e, rows, daily):
    """Logged prediction ka asli return (entry agle open par), ya None agar abhi pata nahi."""
    j = next((k for k in range(len(rows) - 1, -1, -1) if rows[k]["t"] == e["t_last"]), None)
    h = e["h"]
    if j is None or j + h >= len(rows):
        return None
    if not daily and _dt(rows[j + h]["t"]).date() != _dt(rows[j]["t"]).date():
        return None
    return rows[j + h]["c"] / rows[j + 1]["o"] - 1


def track_record(rows, source, symbol, interval, daily):
    """
    Logged predictions ko abhi ke candles se milao. Ye hi sabse imaandaar number
    hai — model ne ye predictions us waqt ki thi, baad me fit nahi hua.
    """
    ents = [e for e in _read_log() if e["source"] == source
            and e["symbol"] == symbol and e["interval"] == interval]
    if not ents:
        return {"logged": 0}
    tmap = {r["t"]: j for j, r in enumerate(rows)}
    resolved = hits = sig_n = 0
    sig_pnl = []
    for e in ents:
        j, h = tmap.get(e["t_last"]), e["h"]
        if j is None or j + h >= len(rows):
            continue
        if not daily and _dt(rows[j + h]["t"]).date() != _dt(rows[j]["t"]).date():
            continue
        r = rows[j + h]["c"] / rows[j + 1]["o"] - 1
        actual = "UP" if r > e["band"] else "DOWN" if r < -e["band"] else "FLAT"
        resolved += 1
        hits += actual == e["pred"]
        if e["signal"] in ("LONG", "SHORT"):
            sig_n += 1
            sig_pnl.append((1 if e["signal"] == "LONG" else -1) * r - e["band"])
    return {"logged": len(ents), "resolved": resolved,
            "hit_pct": round(100 * hits / resolved, 1) if resolved else None,
            "signals": sig_n,
            "signal_avg_pct": round(100 * sum(sig_pnl) / sig_n, 3) if sig_n else None}


def recent_log(limit=100):
    return _read_log()[-limit:]
