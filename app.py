"""
Stock Signal v2 — imaandaar version.

v1 se kya badla:
  • Adhoora (abhi ban raha) candle hata diya jata hai
  • Market khula hai ya nahi, ye check hota hai
  • "Confidence" ka jhootha formula hata diya — ab asli out-of-sample number dikhta hai
  • Report card hit-rate nahi, COST KE BAAD KA P&L dikhata hai
  • Backtest ab walk-forward hai: aadha data tuning ka, aadha test ka (jo tumne kabhi nahi dekha)
  • Trade simulation: entry agle candle ke open par, ATR ka stop-loss aur target
  • Barabari (price nahi hila) ab DOWN ki jeet nahi gini jati
  • RSI/Bollinger ke vote smooth — 69 se 71 par palatte nahi
  • ADX gate: trend kamzor ho to signal fire hi nahi hota
  • Cache ki safai, prev_close fix, live LTP ka use
"""

import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import NormalDist
from datetime import datetime, timezone, timedelta

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import forecast as fc

app = FastAPI(title="Stock Signal")

try:
    import upstox as ups
    app.include_router(ups.router)
    HAS_UPSTOX = True
except Exception:
    HAS_UPSTOX = False

IST = timezone(timedelta(hours=5, minutes=30))
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0 Safari/537.36"}

# interval -> (yahoo range, minutes per candle, label)
# Yahoo 5m/15m/30m ka data sirf pichhle 60 din ka deta hai — "3mo" maangne par
# 422 error aata tha aur 15m/30m chalte hi nahi the.
INTERVALS = {
    "1m":  ("5d",  1,   "1 minute"),
    "5m":  ("60d", 5,   "5 minute"),
    "15m": ("60d", 15,  "15 minute"),
    "30m": ("60d", 30,  "30 minute"),
    "60m": ("1y",  60,  "1 ghante"),
    "1d":  ("5y",  375, "1 din"),
}

# Purane fixed numbers — sirf fallback. Asli cost ab cost_model() se aati hai.
COST_PCT = 0.07 / 100
SLIP_PCT = 0.03 / 100

# ─────────────────────────  COST MODEL  ─────────────────────────
# Intraday equity (NSE), discount-broker jaisa structure. Apne contract note se
# milao aur env se badlo — ye research assumption hai, ground truth nahi.
BROKERAGE_FLAT = float(os.getenv("BROKERAGE_FLAT", "20"))        # ₹ per order
BROKERAGE_PCT = float(os.getenv("BROKERAGE_PCT", "0.03")) / 100  # jo kam ho
STT_SELL = 0.025 / 100          # intraday: sirf sell side
EXCH_PCT = 0.00297 / 100        # NSE transaction charge, dono side
SEBI_PCT = 0.0001 / 100         # ₹10 per crore, dono side
STAMP_BUY = 0.003 / 100         # sirf buy side
GST = 0.18                      # brokerage + exchange + SEBI par


def slippage_pct(rows):
    """
    Liquidity ke hisaab se slippage: pichhle ~20 din ka median roz ka turnover.
    Kam liquid stock me spread + market order ka asar zyada hota hai.
    """
    by_day = {}
    for r in rows:
        dd = datetime.fromtimestamp(r["t"], IST).date()
        by_day[dd] = by_day.get(dd, 0.0) + r["c"] * r["v"]
    vals = sorted(list(by_day.values())[-20:])
    turnover = vals[len(vals) // 2] if vals else 0.0
    cr = turnover / 1e7
    if cr == 0:
        tier = (0.02, "index / volume data nahi")
    elif cr > 500:
        tier = (0.02, f"bahut liquid (₹{cr:,.0f} Cr/din)")
    elif cr > 100:
        tier = (0.03, f"liquid (₹{cr:,.0f} Cr/din)")
    elif cr > 20:
        tier = (0.06, f"theek-thaak liquid (₹{cr:,.0f} Cr/din)")
    else:
        tier = (0.12, f"kam liquid (₹{cr:,.1f} Cr/din)")
    return tier[0] / 100, tier[1]


def cost_model(order_value, rows):
    """Ek round trip (buy + sell) ki poori cost, order value ke % me."""
    v = max(order_value, 1.0)
    brok = 2 * min(BROKERAGE_FLAT, v * BROKERAGE_PCT)
    exch = 2 * v * EXCH_PCT
    sebi = 2 * v * SEBI_PCT
    parts = {"brokerage": brok, "stt": v * STT_SELL, "exchange": exch, "sebi": sebi,
             "stamp": v * STAMP_BUY, "gst": GST * (brok + exch + sebi)}
    charges = sum(parts.values())
    slip, liq = slippage_pct(rows)
    parts["slippage"] = v * slip
    total = charges + parts["slippage"]
    return {"total_pct": total / v, "charges_pct": charges / v, "slip_pct": slip,
            "liquidity": liq, "order_value": round(v),
            "rupees": {k: round(x, 2) for k, x in parts.items()},
            "total_rupees": round(total, 2)}


def bootstrap_ci(pnls, runs=2000, lo=5, hi=95, seed=42):
    """
    Trades ko baar-baar replacement ke saath dobara uthao aur average nikalo.
    Isse pata chalta hai ki jo average dikha wo kitna sthir hai.
    Win-rate ke bajaye seedha P&L par test — payoff asymmetry apne aap handle.
    """
    rnd = random.Random(seed)
    n = len(pnls)
    means = []
    for _ in range(runs):
        means.append(sum(pnls[rnd.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(runs * lo / 100)], means[int(runs * hi / 100)]

_cache = {}
CACHE_TTL = 25


def cached(key, fn):
    now = time.time()
    if len(_cache) > 200:                       # bug 15: cache ki safai
        for k in [k for k, v in list(_cache.items()) if now - v[0] > 300]:
            _cache.pop(k, None)
    hit = _cache.get(key)
    if hit and now - hit[0] < CACHE_TTL:
        return hit[1]
    val = fn()
    _cache[key] = (now, val)
    return val


def market_open(now=None, rows=None):
    """
    NSE: Mon-Fri 9:15 - 15:30 IST.
    FIX 5: chhuttiyon ki hardcoded list rakhne ke bajaye data se pata karte hain —
    agar aaj ka koi candle aaya hi nahi, to market band hai (chhutti ya kuch aur).
    """
    n = now or datetime.now(IST)
    if n.weekday() > 4:
        return False
    if not ((9, 15) <= (n.hour, n.minute) < (15, 30)):
        return False
    if rows:
        last_day = datetime.fromtimestamp(rows[-1]["t"], IST).date()
        if last_day != n.date():
            return False          # aaj ka data hi nahi — chhutti
    return True


# ─────────────────────────────  DATA  ─────────────────────────────

def yahoo_search(q):
    r = requests.get("https://query1.finance.yahoo.com/v1/finance/search",
                     params={"q": q, "quotesCount": 8, "newsCount": 0},
                     headers=UA, timeout=10)
    r.raise_for_status()
    out = []
    for it in r.json().get("quotes", []):
        if it.get("quoteType") not in ("EQUITY", "INDEX", "ETF"):
            continue
        sym = it.get("symbol", "")
        out.append({"symbol": sym,
                    "name": it.get("longname") or it.get("shortname") or sym,
                    "exchange": it.get("exchDisp") or it.get("exchange") or "",
                    "india": sym.endswith(".NS") or sym.endswith(".BO")})
    out.sort(key=lambda x: (not x["india"],))
    return out


def yahoo_candles(symbol, interval):
    rng, mins, _ = INTERVALS[interval]
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                     params={"range": rng, "interval": interval,
                             "includePrePost": "false"},
                     headers=UA, timeout=15)
    if r.status_code != 200:
        raise HTTPException(502, f"Data source ne {r.status_code} bheja")
    res = (r.json().get("chart") or {}).get("result")
    if not res:
        raise HTTPException(404, "Is symbol ka data nahi mila")
    res = res[0]
    meta = res.get("meta", {})
    ts = res.get("timestamp") or []
    q = res["indicators"]["quote"][0]

    rows = []
    for i in range(len(ts)):
        o, h, l, c, v = q["open"][i], q["high"][i], q["low"][i], q["close"][i], q["volume"][i]
        if None in (o, h, l, c):
            continue
        rows.append({"t": ts[i], "o": o, "h": h, "l": l, "c": c, "v": v or 0})

    # BUG 9 FIX: aakhri candle abhi ban raha ho to use hatao.
    # Adhoore candle par RSI/MACD galat aate hain aur signal flicker karta hai.
    dropped = False
    if rows and interval != "1d":
        age = time.time() - rows[-1]["t"]
        if age < mins * 60:
            rows.pop()
            dropped = True

    if len(rows) < 120:
        raise HTTPException(422, "Itne candles nahi mile ki bharosemand test ho sake")

    # BUG 7 FIX: previousClose pehle, chartPreviousClose baad me.
    # chartPreviousClose range ke shuru se pehle ka close hota hai (kai din purana).
    prev = meta.get("previousClose") or meta.get("chartPreviousClose")

    return {"rows": rows, "dropped_forming": dropped,
            "name": meta.get("longName") or meta.get("shortName") or symbol,
            "currency": meta.get("currency", ""),
            "exchange": meta.get("fullExchangeName", ""),
            "ltp": meta.get("regularMarketPrice"),      # BUG 8 FIX: ab use hota hai
            "prev_close": prev,
            "state": meta.get("marketState", "")}


# ─────────────────  INDICATORS (v1 se same — ye sahi the)  ─────────────────

def ema_series(vals, period):
    k = 2 / (period + 1)
    out = [None] * len(vals)
    if len(vals) < period:
        return out
    prev = sum(vals[:period]) / period
    out[period - 1] = prev
    for i in range(period, len(vals)):
        prev = vals[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def sma_series(vals, period):
    out, run = [None] * len(vals), 0.0
    for i, v in enumerate(vals):
        run += v
        if i >= period:
            run -= vals[i - period]
        if i >= period - 1:
            out[i] = run / period
    return out


def stdev_series(vals, period):
    out = [None] * len(vals)
    for i in range(period - 1, len(vals)):
        w = vals[i - period + 1:i + 1]
        m = sum(w) / period
        out[i] = math.sqrt(sum((x - m) ** 2 for x in w) / period)
    return out


def rsi_series(closes, period=14):
    out = [None] * len(closes)
    if len(closes) <= period:
        return out
    g = l_ = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        g += max(d, 0); l_ += max(-d, 0)
    ag, al = g / period, l_ / period
    out[period] = 100 - 100 / (1 + (ag / al if al else 999))
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        ag = (ag * (period - 1) + max(d, 0)) / period
        al = (al * (period - 1) + max(-d, 0)) / period
        out[i] = 100 - 100 / (1 + (ag / al if al else 999))
    return out


def macd_series(closes, fast=12, slow=26, sig=9):
    ef, es = ema_series(closes, fast), ema_series(closes, slow)
    line = [None if (ef[i] is None or es[i] is None) else ef[i] - es[i]
            for i in range(len(closes))]
    start = next((i for i, v in enumerate(line) if v is not None), len(line))
    sg = ema_series(line[start:], sig) if line[start:] else []
    signal = ([None] * start + sg)
    signal += [None] * (len(closes) - len(signal))
    hist = [None if (line[i] is None or signal[i] is None) else line[i] - signal[i]
            for i in range(len(closes))]
    return line, signal, hist


def true_ranges(h, l, c):
    tr = [h[0] - l[0]]
    for i in range(1, len(c)):
        tr.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    return tr


def wilder(vals, period):
    out = [None] * len(vals)
    if len(vals) < period:
        return out
    s = sum(vals[:period]) / period
    out[period - 1] = s
    for i in range(period, len(vals)):
        s = (s * (period - 1) + vals[i]) / period
        out[i] = s
    return out


def atr_series(h, l, c, period=14):
    return wilder(true_ranges(h, l, c), period)


def adx_series(h, l, c, period=14):
    n = len(c)
    pdm, mdm = [0.0], [0.0]
    for i in range(1, n):
        up, dn = h[i] - h[i - 1], l[i - 1] - l[i]
        pdm.append(up if (up > dn and up > 0) else 0.0)
        mdm.append(dn if (dn > up and dn > 0) else 0.0)
    tr_s = wilder(true_ranges(h, l, c), period)
    p_s, m_s = wilder(pdm, period), wilder(mdm, period)
    pdi = [None] * n; mdi = [None] * n; dx = [None] * n
    for i in range(n):
        if tr_s[i] and p_s[i] is not None and m_s[i] is not None and tr_s[i] > 0:
            pdi[i] = 100 * p_s[i] / tr_s[i]
            mdi[i] = 100 * m_s[i] / tr_s[i]
            tot = pdi[i] + mdi[i]
            dx[i] = 100 * abs(pdi[i] - mdi[i]) / tot if tot else 0.0
    adx = [None] * n
    vals = [(i, v) for i, v in enumerate(dx) if v is not None]
    if len(vals) >= period:
        s = sum(v for _, v in vals[:period]) / period
        adx[vals[period - 1][0]] = s
        for i, v in vals[period:]:
            s = (s * (period - 1) + v) / period
            adx[i] = s
    return adx, pdi, mdi


def supertrend_series(h, l, c, period=10, mult=3.0):
    n = len(c)
    atr = atr_series(h, l, c, period)
    dir_, line = [None] * n, [None] * n
    up_prev = dn_prev = None
    for i in range(n):
        if atr[i] is None:
            continue
        mid = (h[i] + l[i]) / 2
        up, dn = mid - mult * atr[i], mid + mult * atr[i]
        if up_prev is not None:
            up = max(up, up_prev) if c[i - 1] > up_prev else up
            dn = min(dn, dn_prev) if c[i - 1] < dn_prev else dn
        prev = dir_[i - 1] if i > 0 and dir_[i - 1] else 1
        if dn_prev is not None and prev == -1 and c[i] > dn_prev:
            d = 1
        elif up_prev is not None and prev == 1 and c[i] < up_prev:
            d = -1
        else:
            d = prev
        dir_[i], line[i] = d, (up if d == 1 else dn)
        up_prev, dn_prev = up, dn
    return dir_, line


def vwap_series(rows):
    out = [None] * len(rows)
    day, pv, vol = None, 0.0, 0.0
    for i, r in enumerate(rows):
        dd = datetime.fromtimestamp(r["t"], IST).date()
        if dd != day:
            day, pv, vol = dd, 0.0, 0.0
        pv += (r["h"] + r["l"] + r["c"]) / 3 * r["v"]
        vol += r["v"]
        out[i] = pv / vol if vol > 0 else r["c"]
    return out


# ────────────────────────────  SCORING  ────────────────────────────

WEIGHTS = {
    "EMA trend":  1.4,
    "Supertrend": 1.3,
    "MACD":       1.2,
    "RSI":        1.0,
    "VWAP":       1.0,
    "ADX":        0.9,
    "Bollinger":  0.8,
    # BUG 13: ye dono sirf abhi kya hua bata rahe the, aage ka kuchh nahi.
    # Weight ghata diya taaki score me bekaar ka agreement na bane.
    "Volume":     0.4,
    "Momentum":   0.4,
}
MAX_SCORE = sum(WEIGHTS.values())

ADX_GATE = 20     # isse kam trend par signal fire hi nahi hoga
THRESHOLD = 0.45  # v1 me 0.30 tha — tab 80% candles par signal banta tha


def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def build(rows):
    c = [r["c"] for r in rows]; h = [r["h"] for r in rows]
    l = [r["l"] for r in rows]; v = [r["v"] for r in rows]
    o = [r["o"] for r in rows]
    mline, msig, mhist = macd_series(c)
    adx, pdi, mdi = adx_series(h, l, c)
    st_dir, st_line = supertrend_series(h, l, c)
    return {"o": o, "c": c, "h": h, "l": l, "v": v,
            "e9": ema_series(c, 9), "e21": ema_series(c, 21), "e50": ema_series(c, 50),
            "macd": mline, "msig": msig, "mhist": mhist,
            "rsi": rsi_series(c), "bb_mid": sma_series(c, 20), "bb_sd": stdev_series(c, 20),
            "atr": atr_series(h, l, c), "adx": adx, "pdi": pdi, "mdi": mdi,
            "st_dir": st_dir, "st_line": st_line,
            "vwap": vwap_series(rows), "vol_avg": sma_series(v, 20)}


def score_at(d, i, detail=False):
    votes, rows = {}, []
    c = d["c"][i]

    def add(name, vote, text):
        votes[name] = clamp(vote)
        if detail:
            rows.append({"name": name, "vote": round(clamp(vote), 2),
                         "reading": text, "weight": WEIGHTS[name]})

    e9, e21, e50 = d["e9"][i], d["e21"][i], d["e50"][i]
    if None not in (e9, e21, e50):
        if e9 > e21 > e50:
            add("EMA trend", 1, "9 > 21 > 50 — upar ka stack")
        elif e9 < e21 < e50:
            add("EMA trend", -1, "9 < 21 < 50 — neeche ka stack")
        else:
            add("EMA trend", 0.4 if e9 > e21 else -0.4, "EMA mila-jula")

    if d["st_dir"][i]:
        add("Supertrend", d["st_dir"][i],
            f"{'Tezi' if d['st_dir'][i] == 1 else 'Mandi'} band, line {d['st_line'][i]:.2f}")

    mh = d["mhist"][i]
    mh_prev = d["mhist"][i - 1] if i else None
    if mh is not None:
        rising = mh_prev is not None and mh > mh_prev
        base = 0.6 if mh > 0 else -0.6
        base += 0.4 if rising else -0.4
        add("MACD", base, f"histogram {mh:+.3f}, {'badh raha' if rising else 'ghat raha'}")

    # BUG 12 FIX: RSI ab smooth. 69 -> 71 par vote nahi palatta.
    r = d["rsi"][i]
    if r is not None:
        if r > 75:
            vote = 1 - (r - 75) / 6.25          # 75:+1  →  81.25:-1, continuous
        elif r < 25:
            vote = -1 + (25 - r) / 6.25
        else:
            vote = (r - 50) / 25                 # 25:-1  50:0  75:+1
        note = "bahut overbought" if r > 75 else "bahut oversold" if r < 25 else "normal"
        add("RSI", vote, f"{r:.1f} — {note}")

    vw = d["vwap"][i]
    if vw:
        gap = (c - vw) / vw * 100
        add("VWAP", clamp(gap / 0.5), f"VWAP se {gap:+.2f}%")

    a, p, m = d["adx"][i], d["pdi"][i], d["mdi"][i]
    if None not in (a, p, m):
        vote = 0.0 if a < ADX_GATE else clamp((p - m) / 25)
        s = "kamzor" if a < ADX_GATE else "strong" if a > 25 else "theek-thaak"
        add("ADX", vote, f"{a:.1f} — trend {s}")

    mid, sd = d["bb_mid"][i], d["bb_sd"][i]
    if mid and sd and sd > 0:
        z = (c - mid) / sd
        vote = (1 - (abs(z) - 2) / 1.0) * (1 if z > 0 else -1) if abs(z) > 2 else clamp(z / 1.5)
        add("Bollinger", vote, f"beech se {z:+.2f}σ")

    va = d["vol_avg"][i]
    if va and va > 0 and i > 0:
        ratio = d["v"][i] / va
        add("Volume", (1 if c >= d["c"][i - 1] else -1) * clamp((ratio - 1) / 1.5),
            f"average ka {ratio:.2f}x")

    if i >= 5 and d["c"][i - 5]:
        chg = (c - d["c"][i - 5]) / d["c"][i - 5] * 100
        add("Momentum", clamp(chg / 0.5), f"5 candle me {chg:+.2f}%")

    total = sum(WEIGHTS[k] * v for k, v in votes.items())
    return (total, rows) if detail else (total, None)


def signal_at(d, i):
    """ADX gate + threshold. Trend kamzor ho to chup raho."""
    a = d["adx"][i]
    if a is None or a < ADX_GATE:
        return "FLAT", 0.0
    total, _ = score_at(d, i)
    norm = total / MAX_SCORE
    if norm >= THRESHOLD:
        return "UP", norm
    if norm <= -THRESHOLD:
        return "DOWN", norm
    return "FLAT", norm


# ──────────────────  TRADE SIMULATION (asli report card)  ──────────────────

def simulate(d, start, end, cost=COST_PCT, slip=SLIP_PCT,
             stop_atr=1.0, target_atr=1.5, max_hold=8):
    """
    Har trade chala kar dekho. Imaandaar rules:
      • Entry AGLE candle ke OPEN par — jo close abhi dekha wo trade nahi kar sakte
      • FIX: agar candle stop ke paar GAP khulta hai, exit us OPEN par hota hai,
        stop ke price par nahi. Gap me stop apne price par lagta hi nahi.
      • Ek candle me stop aur target dono lage to STOP maana jata hai
      • Har trade par round-trip cost + slippage kata jata hai
      • Trades overlap nahi karte
    """
    trades, i = [], start
    while i < end - 2:
        sig, _ = signal_at(d, i)
        if sig == "FLAT" or d["atr"][i] is None:
            i += 1
            continue
        side = 1 if sig == "UP" else -1
        entry = d["o"][i + 1]
        atr = d["atr"][i]
        stop = entry - side * stop_atr * atr
        target = entry + side * target_atr * atr

        exit_px, exit_j, why = None, None, "time"
        for j in range(i + 1, min(i + 1 + max_hold, end)):
            op, hi, lo = d["o"][j], d["h"][j], d["l"][j]

            # GAP: candle stop ke paar khula — exit open par, stop par nahi
            if (op <= stop) if side == 1 else (op >= stop):
                exit_px, exit_j, why = op, j, "gap"
                break
            if (lo <= stop) if side == 1 else (hi >= stop):
                exit_px, exit_j, why = stop, j, "stop"
                break
            if (hi >= target) if side == 1 else (lo <= target):
                exit_px, exit_j, why = target, j, "target"
                break
        if exit_px is None:
            exit_j = min(i + max_hold, end - 1)
            exit_px = d["c"][exit_j]

        gross = side * (exit_px - entry) / entry
        trades.append({"pnl": gross - cost - slip, "gross": gross,
                       "why": why, "side": sig})
        i = exit_j + 1

    if not trades:
        return {"trades": 0}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    eq = peak = 1.0
    dd = 0.0
    for p in pnls:
        eq *= (1 + p)
        peak = max(peak, eq)
        dd = max(dd, (peak - eq) / peak)

    lo_ci, hi_ci = bootstrap_ci(pnls)

    return {
        "trades": len(trades),
        "win_rate": round(100 * len(wins) / len(trades), 1),
        "breakeven_win_rate": round(100 * stop_atr / (stop_atr + target_atr), 1),
        "avg_pnl_pct": round(sum(pnls) / len(pnls) * 100, 4),
        "ci_low_pct": round(lo_ci * 100, 4),
        "ci_high_pct": round(hi_ci * 100, 4),
        "total_return_pct": round((eq - 1) * 100, 2),
        "max_drawdown_pct": round(dd * 100, 1),
        # FIX 4: koi loss na ho to infinity, None nahi — UI use laal nahi dikhayega
        "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses else 999.0,
        "avg_win_pct": round(sum(wins) / len(wins) * 100, 3) if wins else 0,
        "avg_loss_pct": round(sum(losses) / len(losses) * 100, 3) if losses else 0,
        "stopped_out": sum(1 for t in trades if t["why"] == "stop"),
        "gapped_out": sum(1 for t in trades if t["why"] == "gap"),
        "hit_target": sum(1 for t in trades if t["why"] == "target"),
    }


def evaluate(d, cost=COST_PCT, slip=SLIP_PCT):
    """
    BUG 14 + 20 FIX: walk-forward.
    Data do hisson me: pehla 60%, aakhri 40%. Dono par alag-alag nateeja.
    Dhyan do: parameters kisi data se fit NAHI kiye gaye, haath se likhe hain.
    To ye asli walk-forward validation nahi hai — bas do alag daur ka comparison hai.
    Agar dono halves me nateeja bahut alag hai, to strategy stable nahi hai.
    """
    n = len(d["c"])
    warm = 60
    split = warm + int((n - warm) * 0.60)
    return {
        # FIX 3: pehle ise "in_sample"/"out_sample" kehte the. Wo jhooth tha —
        # WEIGHTS, ADX_GATE, THRESHOLD haath se likhe hain, kisi data par fit nahi
        # kiye gaye. To yahan koi training hui hi nahi. Naam seedha kar diya.
        "first_half": simulate(d, warm, split, cost, slip),
        "second_half": simulate(d, split, n, cost, slip),
        "params_fitted": False,
        "split_at": split, "total_candles": n,
        "cost_pct": round((cost + slip) * 100, 3),
    }


def confidence_note(ev):
    """
    FIX 1: pehle win_rate ko 50% se compare karta tha. Wo galat tha —
    stop 1 ATR aur target 1.5 ATR par breakeven win-rate 40% hai, 50% nahi.
    Ab seedha P&L par bootstrap confidence interval lagta hai. Ye payoff ki
    asymmetry apne aap handle kar leta hai.
    """
    o = ev["second_half"]
    n = o.get("trades", 0)
    if n < 30:
        return ("unknown", f"Sirf {n} trades bane — itne se kuch keh nahi sakte. "
                           f"Is signal ka koi measured track record nahi hai.")
    avg, lo = o["avg_pnl_pct"], o["ci_low_pct"]
    if lo > 0:
        return ("positive",
                f"{n} trades, average {avg:+.3f}% per trade (cost+slippage ke baad). "
                f"90% confidence interval ka neecha sira bhi {lo:+.3f}% hai — "
                f"yaani ye sirf sanyog hone ki sambhavna kam hai.")
    if avg > 0:
        return ("weak",
                f"{n} trades, average {avg:+.3f}% — par confidence interval "
                f"{lo:+.3f}% se {o['ci_high_pct']:+.3f}% tak failta hai, jo zero ko "
                f"cross karta hai. Itne data se edge sabit nahi hota.")
    return ("negative",
            f"{n} trades, average {avg:+.3f}% per trade — cost ke baad paisa doobta hai. "
            f"Is signal par mat chalo.")


# ─────────────────────────────  ROUTES  ─────────────────────────────

@app.get("/api/sources")
def sources():
    out = {"yahoo": True, "upstox": False}
    if HAS_UPSTOX:
        out["upstox"] = bool(ups.get_token())
    return out


@app.get("/api/search")
def search(q: str, source: str = "yahoo"):
    if len(q.strip()) < 2:
        return {"results": []}
    if source == "upstox":
        if not HAS_UPSTOX:
            raise HTTPException(400, "Upstox module load nahi hua")
        return {"results": cached(f"us:{q.lower()}", lambda: ups.find(q))}
    return {"results": cached(f"s:{q.lower()}", lambda: yahoo_search(q))}


# relative strength ke liye benchmark
BENCH = {"yahoo": "^NSEI", "upstox": "NSE_INDEX|Nifty 50"}


def bench_rows(source, symbol, interval, fetch_fn):
    b = BENCH.get(source)
    if not b or symbol == b:
        return None
    if source == "yahoo" and not symbol.endswith((".NS", ".BO")):
        return None           # Indian stock nahi — Nifty se tulna bekaar
    try:
        return cached(f"a:{source}:{b}:{interval}", lambda: fetch_fn(b, interval))["rows"]
    except Exception:
        return None


def norm_score(d, i):
    return score_at(d, i)[0] / MAX_SCORE


def prev_close_from_rows(rows):
    """Pichhle trading din ka aakhri close — Upstox prev_close nahi deta."""
    last = datetime.fromtimestamp(rows[-1]["t"], IST).date()
    for r in reversed(rows):
        if datetime.fromtimestamp(r["t"], IST).date() != last:
            return r["c"]
    return None


def fetcher(source):
    if source == "upstox":
        if not HAS_UPSTOX:
            raise HTTPException(400, "Upstox module load nahi hua")
        return ups.upstox_candles
    return yahoo_candles


# ─────────────────────────  RISK GUARD  ─────────────────────────
# Bot ke LONG/SHORT signals ko paper-trade maan kar din ki limits lagti hain.
# Limit tootne par aage ke signals NO TRADE ho jaate hain — us din ke liye.
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "2"))   # capital ka %
MAX_CONSEC_LOSSES = int(os.getenv("MAX_CONSEC_LOSSES", "3"))
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "5"))
KILL_FILE = os.getenv("KILL_SWITCH_FILE", "KILL_SWITCH")


def risk_state(capital):
    today = datetime.now(IST).date()
    ents = [e for e in fc.recent_log(5000) if e.get("signal") in ("LONG", "SHORT")
            and datetime.fromtimestamp(e["logged_at"], IST).date() == today]
    outcomes = []
    for e in ents:
        try:
            fetch = fetcher(e["source"])
            rows = cached(f"a:{e['source']}:{e['symbol']}:{e['interval']}",
                          lambda: fetch(e["symbol"], e["interval"]))["rows"]
            r = fc.resolve(e, rows, e["interval"] == "1d")
        except Exception:
            r = None
        if r is not None:
            side = 1 if e["signal"] == "LONG" else -1
            outcomes.append(e.get("value", capital) * (side * r - e["band"]))
    day_pnl = sum(outcomes)
    consec = 0
    for p in reversed(outcomes):
        if p >= 0:
            break
        consec += 1
    kill = os.path.exists(KILL_FILE)
    reasons = []
    if kill:
        reasons.append("Kill switch ON hai")
    if len(ents) >= MAX_TRADES_PER_DAY:
        reasons.append(f"Aaj {len(ents)} signal ho chuke (limit {MAX_TRADES_PER_DAY})")
    if day_pnl <= -capital * MAX_DAILY_LOSS_PCT / 100:
        reasons.append(f"Aaj ka nuksaan ₹{-day_pnl:,.0f} — limit capital ka {MAX_DAILY_LOSS_PCT}%")
    if consec >= MAX_CONSEC_LOSSES:
        reasons.append(f"Lagatar {consec} haar (limit {MAX_CONSEC_LOSSES})")
    return {"blocked": bool(reasons), "reasons": reasons, "kill_switch": kill,
            "trades_today": len(ents), "resolved_today": len(outcomes),
            "day_pnl": round(day_pnl), "consec_losses": consec,
            "limits": {"max_daily_loss_pct": MAX_DAILY_LOSS_PCT,
                       "max_consec_losses": MAX_CONSEC_LOSSES,
                       "max_trades_per_day": MAX_TRADES_PER_DAY}}


# ─────────────────────────────  CORE  ─────────────────────────────

def core(symbol, interval="5m", source="yahoo", capital=100000.0, risk_pct=1.0, log=True):
    if interval not in INTERVALS:
        raise HTTPException(400, "Ye timeframe support nahi hai")
    fetch_fn = fetcher(source)

    data = cached(f"a:{source}:{symbol}:{interval}", lambda: fetch_fn(symbol, interval))
    rows = data["rows"]
    d = build(rows)
    i = len(rows) - 1

    total, details = score_at(d, i, detail=True)
    norm = total / MAX_SCORE
    sig, _ = signal_at(d, i)
    headline = {"UP": "Tezi ke aasaar", "DOWN": "Mandi ke aasaar",
                "FLAT": "Koi saaf direction nahi"}[sig]

    close = d["c"][i]
    atr = d["atr"][i] or close * 0.003
    _, mins, tf_label = INTERVALS[interval]

    # POSITION SIZING: stop (1×ATR) lage to capital ka sirf risk_pct% jaye.
    # Leverage nahi — order value capital se zyada nahi.
    risk_amt = capital * risk_pct / 100
    qty = int(min(risk_amt / atr, capital / close)) if atr > 0 else 0
    order_value = qty * close if qty > 0 else capital
    costs = cost_model(order_value, rows)
    cost = costs["total_pct"]
    sizing = {"capital": round(capital), "risk_pct": risk_pct, "risk_amt": round(risk_amt),
              "qty": qty, "order_value": round(order_value),
              "max_loss": round(qty * atr + costs["total_rupees"]),
              "capped": qty > 0 and qty == int(capital / close)}

    ev = evaluate(d, costs["charges_pct"], costs["slip_pct"])
    verdict_kind, verdict_text = confidence_note(ev)

    # Cost reality: is timeframe par cost, normal move ka kitna hissa kha jayegi
    move_pct = atr / close * 100
    cost_ratio = round(cost * 100 / move_pct * 100) if move_pct else None

    now = datetime.now(IST)
    is_open = market_open(now, rows)
    stale_min = round((now.timestamp() - rows[i]["t"]) / 60)

    # 30-min forecast — model train + walk-forward. Naya candle aane tak cache.
    fkey = f"f:{source}:{symbol}:{interval}:{rows[i]['t']}:{is_open}:{cost:.6f}"
    hit = _cache.get(fkey)
    if hit:
        forecast = hit[1]
    else:
        try:
            forecast = fc.forecast(rows, d, mins, norm_score, build, cost,
                                   bench=bench_rows(source, symbol, interval, fetch_fn),
                                   market_open=is_open, cost_ratio=cost_ratio)
        except Exception as e:     # forecast fail ho to baaki analysis na ruke
            forecast = {"available": False, "reason": f"Forecast nahi bana: {e}"}
        _cache[fkey] = (time.time(), forecast)

    # risk limits forecast ke upar — cache wale dict ko mat chhedo
    forecast = dict(forecast)
    risk = risk_state(capital)
    if forecast.get("signal") in ("LONG", "SHORT") and risk["blocked"]:
        forecast["signal"] = "NO TRADE"
        forecast["signal_reason"] = "Risk limit: " + "; ".join(risk["reasons"])

    # TRADE PLAN ab forecast ke final signal se — purane indicator vote se nahi.
    # Pehle indicator "UP" bolta tha to plan dikh jata tha, chahe model NO TRADE kahe.
    side = {"LONG": 1, "SHORT": -1}.get(forecast.get("signal"), 0)
    plan = None
    if side and qty > 0:
        plan = {"side": forecast["signal"], "entry": round(close, 2),
                "stop": round(close - side * atr, 2),
                "exit": f"{forecast.get('horizon_label')} baad (time exit) ya stop",
                "qty": qty, "order_value": round(order_value),
                "max_loss": sizing["max_loss"], "risk_pct": round(atr / close * 100, 2)}

    if log:
        fc.log_prediction(source, symbol, interval, forecast, round(close, 2),
                          extra={"qty": qty, "value": round(order_value)})
    track = fc.track_record(rows, source, symbol, interval, mins >= 375)

    prev = data.get("prev_close") or prev_close_from_rows(rows)
    return {
        "symbol": symbol, "name": data["name"], "exchange": data["exchange"],
        "source": source,
        "currency": data.get("currency", ""), "interval": interval,
        "tf_label": tf_label, "horizon_min": mins,
        "market_open": is_open,
        "data_age_min": stale_min,
        "dropped_forming": data.get("dropped_forming", False),
        "last_candle": datetime.fromtimestamp(rows[i]["t"], IST).strftime("%d %b, %I:%M %p"),
        "price": round(close, 2),
        "ltp": round(data["ltp"], 2) if data.get("ltp") else None,
        "day_change_pct": round((close - prev) / prev * 100, 2) if prev else None,
        "signal": sig, "headline": headline,
        "score": round(total, 2), "score_norm": round(norm, 3),
        "max_score": round(MAX_SCORE, 2), "threshold": THRESHOLD,
        "atr": round(atr, 2), "move_pct": round(move_pct, 3),
        "cost_pct": round(cost * 100, 3), "cost_ratio": cost_ratio,
        "costs": costs, "sizing": sizing, "risk": risk,
        "plan": plan,
        "verdict_kind": verdict_kind, "verdict_text": verdict_text,
        "selection_warning": True,
        "evaluation": ev,
        "indicators": details,
        "forecast": {k: v for k, v in forecast.items() if k != "t_last"},
        "track": track,
        "data_warnings": fc.check_quality(rows, mins),
        "spark": [round(x, 2) for x in d["c"][-60:]],
    }


@app.get("/api/analyze")
def analyze(symbol: str, interval: str = "5m", source: str = "yahoo",
            capital: float = 100000, risk_pct: float = 1.0):
    capital = min(max(capital, 1000), 1e9)
    risk_pct = min(max(risk_pct, 0.1), 5.0)
    return core(symbol, interval, source, capital, risk_pct)


@app.get("/api/forecast/log")
def forecast_log(limit: int = 100):
    """Live logged predictions — baad me khud jaanch sakte ho model sach me kaisa raha."""
    return {"entries": fc.recent_log(limit)}


@app.get("/api/risk")
def risk(capital: float = 100000):
    return risk_state(capital)


@app.post("/api/kill")
def kill(on: bool = True):
    """Kill switch: ON hone par koi LONG/SHORT signal nahi niklega jab tak OFF na karo."""
    if on:
        open(KILL_FILE, "w").write(datetime.now(IST).isoformat())
    elif os.path.exists(KILL_FILE):
        os.remove(KILL_FILE)
    return {"kill_switch": os.path.exists(KILL_FILE)}


# ─────────────────────────────  SCANNER  ─────────────────────────────
# Nifty 50 ke har stock par wahi poora forecast chalta hai. Ranking EXPECTED VALUE
# ke neeche wale sire (lower bound) par hoti hai, aur kyunki 50 stock ek saath
# dekhe ja rahe hain, bound ko multiple-testing ke hisaab se kada kiya jata hai
# (Bonferroni). Warna 50 me se 2-3 stock sirf kismat se "shaandaar" dikh jaate.
UNIVERSE = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", "BHARTIARTL.NS",
    "SBIN.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS", "HCLTECH.NS", "M&M.NS", "TITAN.NS",
    "ULTRACEMCO.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "TATASTEEL.NS", "ASIANPAINT.NS",
    "NESTLEIND.NS", "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS", "BAJAJFINSV.NS",
    "JSWSTEEL.NS", "WIPRO.NS", "TECHM.NS", "GRASIM.NS", "HINDALCO.NS", "CIPLA.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "BEL.NS", "TRENT.NS",
    "SHRIRAMFIN.NS", "APOLLOHOSP.NS", "TATACONSUM.NS", "SBILIFE.NS", "HDFCLIFE.NS",
    "INDIGO.NS", "JIOFIN.NS", "ETERNAL.NS", "TMPV.NS",
]
SCAN_TTL = 15 * 60
_scan = {"running": False, "done": 0, "total": 0, "results": [], "interval": None,
         "started": None, "finished": None, "z": None, "tested": 0}
_scan_lock = threading.Lock()


def _summary(sym, res):
    f = res["forecast"]
    ok = f.get("available")
    ev = f.get("ev", {}) if ok else {}
    return {"symbol": sym, "name": res["name"], "price": res["price"],
            "day_change_pct": res["day_change_pct"],
            "predicted": f.get("predicted") if ok else None,
            "prob": f["probs"][f["predicted"]] if ok else None,
            "signal": f.get("signal") if ok else "NO TRADE",
            "confidence": f.get("confidence") if ok else None,
            "skill": f["quality"]["logloss_skill_pct"] if ok else None,
            "regime": f["regime"]["trend"] if ok else None,
            "ev_n": ev.get("n", 0), "ev_avg_pct": ev.get("avg_pct"),
            "ev_sd_pct": ev.get("sd_pct"), "ev_low_pct": ev.get("ci_low_pct"),
            "reason": None if ok else f.get("reason")}


def _run_scan(interval, capital, risk_pct):
    def one(sym):
        try:
            out = _summary(sym, core(sym, interval, "yahoo", capital, risk_pct))
        except Exception as e:
            out = {"symbol": sym, "error": str(getattr(e, "detail", e))[:120], "ev_n": 0}
        with _scan_lock:
            _scan["done"] += 1
            _scan["results"].append(out)

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, UNIVERSE))

    with _scan_lock:
        res = _scan["results"]
        tested = [r for r in res if r.get("ev_n", 0) >= 10 and r.get("ev_sd_pct") is not None]
        N = max(1, len(tested))
        z = NormalDist().inv_cdf(1 - 0.05 / N)       # one-sided, 5% / N
        for r in res:
            r["adj_low_pct"] = None
            r["qualified"] = False
            if r in tested:
                r["adj_low_pct"] = round(r["ev_avg_pct"] - z * r["ev_sd_pct"] / math.sqrt(r["ev_n"]), 4)
                r["qualified"] = (r["adj_low_pct"] > 0 and (r["skill"] or 0) > 0
                                  and r["predicted"] in ("UP", "DOWN"))
        res.sort(key=lambda r: (not r["qualified"],
                                -(r["adj_low_pct"] if r["adj_low_pct"] is not None else -1e9)))
        _scan.update({"running": False, "finished": time.time(), "z": round(z, 2), "tested": N})


@app.get("/api/scan")
def scan_status():
    with _scan_lock:
        out = {k: v for k, v in _scan.items()}
        out["results"] = list(_scan["results"]) if not _scan["running"] else []
    out["age_sec"] = round(time.time() - out["finished"]) if out["finished"] else None
    out["stale"] = out["age_sec"] is None or out["age_sec"] > SCAN_TTL
    return out


@app.post("/api/scan")
def scan_start(interval: str = "5m", capital: float = 100000, risk_pct: float = 1.0):
    if interval not in INTERVALS:
        raise HTTPException(400, "Ye timeframe support nahi hai")
    with _scan_lock:
        if _scan["running"]:
            return {"started": False, "running": True}
        _scan.update({"running": True, "done": 0, "total": len(UNIVERSE), "results": [],
                      "interval": interval, "started": time.time()})
    threading.Thread(target=_run_scan, args=(interval, min(max(capital, 1000), 1e9),
                                             min(max(risk_pct, 0.1), 5.0)),
                     daemon=True).start()
    return {"started": True, "running": True}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/index.html")