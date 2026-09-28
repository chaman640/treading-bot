"""
Upstox data source — app.py me drop karo aur `source=upstox` bhej do.

Ye 4 kaam karta hai:
  1. OAuth login (roz ek baar, manual — SEBI rule hai, automate mat karna)
  2. Access token ko file me cache karna, agle 3:30 AM tak
  3. Instrument master (RELIANCE -> NSE_EQ|INE002A01018) download + cache
  4. Candles laana, usi shape me jo app.py ka build() expect karta hai

Chahiye:
  UPSTOX_API_KEY, UPSTOX_API_SECRET, UPSTOX_REDIRECT_URI  (environment me)
"""

import gzip
import io
import json
import os
import time
from datetime import datetime, timedelta, timezone

import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter(prefix="/api/upstox", tags=["upstox"])

IST = timezone(timedelta(hours=5, minutes=30))
API = "https://api.upstox.com"
INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0 Safari/537.36"}

KEY = os.getenv("UPSTOX_API_KEY", "")
SECRET = os.getenv("UPSTOX_API_SECRET", "")
REDIRECT = os.getenv("UPSTOX_REDIRECT_URI", "http://localhost:8000/api/upstox/callback")

TOKEN_FILE = os.getenv("UPSTOX_TOKEN_FILE", ".upstox_token.json")
INSTR_FILE = ".upstox_nse.json"

# app.py wale interval codes -> Upstox v3 (unit, interval)
UNITS = {
    "1m":  ("minutes", 1),
    "5m":  ("minutes", 5),
    "15m": ("minutes", 15),
    "30m": ("minutes", 30),
    "60m": ("minutes", 60),
    "1d":  ("days", 1),
}


# ──────────────────────────  TOKEN  ──────────────────────────

def _next_expiry():
    """Token agle trading din 3:30 AM IST tak chalta hai."""
    now = datetime.now(IST)
    exp = now.replace(hour=3, minute=30, second=0, microsecond=0)
    if now >= exp:
        exp += timedelta(days=1)
    return exp.timestamp()


def save_token(tok, extra=None):
    json.dump({"access_token": tok, "expires_at": _next_expiry(), **(extra or {})},
              open(TOKEN_FILE, "w"))


def get_token():
    if not os.path.exists(TOKEN_FILE):
        return None
    try:
        d = json.load(open(TOKEN_FILE))
    except Exception:
        return None
    if time.time() >= d.get("expires_at", 0):
        return None
    return d.get("access_token")


def need_token():
    t = get_token()
    if not t:
        raise HTTPException(401, "Upstox token expire ho gaya — /api/upstox/login kholo")
    return t


def headers():
    return {"Accept": "application/json", "Authorization": f"Bearer {need_token()}"}


# ──────────────────────────  OAUTH  ──────────────────────────

@router.get("/login")
def login():
    """Browser me kholo. Upstox par login karoge, wapas callback par aa jaoge."""
    if not KEY:
        raise HTTPException(500, "UPSTOX_API_KEY set nahi hai")
    url = (f"{API}/v2/login/authorization/dialog?response_type=code"
           f"&client_id={KEY}&redirect_uri={REDIRECT}")
    return RedirectResponse(url)


@router.get("/callback", response_class=HTMLResponse)
def callback(code: str = "", error: str = ""):
    if error or not code:
        return f"<p>Login nahi hua: {error or 'code nahi mila'}</p>"
    r = requests.post(
        f"{API}/v2/login/authorization/token",
        headers={"accept": "application/json",
                 "Content-Type": "application/x-www-form-urlencoded"},
        data={"code": code, "client_id": KEY, "client_secret": SECRET,
              "redirect_uri": REDIRECT, "grant_type": "authorization_code"},
        timeout=15,
    )
    js = r.json()
    if "access_token" not in js:
        return f"<p>Token nahi mila: {js}</p>"
    save_token(js["access_token"], {"user": js.get("user_name", "")})
    exp = datetime.fromtimestamp(_next_expiry(), IST).strftime("%d %b %I:%M %p")
    return (f"<body style='font-family:system-ui;padding:40px;background:#0D151E;color:#E3EEF6'>"
            f"<h2>Upstox jud gaya</h2><p>Token {exp} tak chalega.</p>"
            f"<p><a style='color:#31C48D' href='/'>Tool par wapas jao</a></p></body>")


@router.get("/status")
def status():
    t = get_token()
    if not t:
        return {"connected": False, "login_url": "/api/upstox/login"}
    d = json.load(open(TOKEN_FILE))
    return {"connected": True, "user": d.get("user", ""),
            "expires_at": datetime.fromtimestamp(d["expires_at"], IST).isoformat()}


# ─────────────────────  INSTRUMENT MASTER  ─────────────────────
# RELIANCE -> NSE_EQ|INE002A01018
# File roz subah ~6:00 AM refresh hoti hai. Ek baar din me download, phir cache.

_instr = None


def load_instruments(force=False):
    global _instr
    if _instr is not None and not force:
        return _instr

    fresh = (os.path.exists(INSTR_FILE)
             and time.time() - os.path.getmtime(INSTR_FILE) < 20 * 3600)
    if not fresh or force:
        r = requests.get(INSTRUMENTS_URL, headers=UA, timeout=90)
        r.raise_for_status()
        raw = gzip.GzipFile(fileobj=io.BytesIO(r.content)).read()
        data = json.loads(raw)
        eq = [x for x in data if x.get("segment") in ("NSE_EQ", "NSE_INDEX")
              and x.get("instrument_key")]
        slim = [{"key": x["instrument_key"],
                 "sym": (x.get("trading_symbol") or "").upper(),
                 "name": x.get("name") or "",
                 "seg": x["segment"]} for x in eq]
        json.dump(slim, open(INSTR_FILE, "w"))

    _instr = json.load(open(INSTR_FILE))
    return _instr


def find(query, limit=8):
    q = query.strip().upper()
    if not q:
        return []
    rows = load_instruments()
    exact, starts, contains = [], [], []
    for x in rows:
        s, n = x["sym"], x["name"].upper()
        if s == q:
            exact.append(x)
        elif s.startswith(q) or n.startswith(q):
            starts.append(x)
        elif q in n:
            contains.append(x)
        if len(exact) + len(starts) + len(contains) > 400:
            break
    out = (exact + starts + contains)[:limit]
    return [{"symbol": x["key"], "name": x["name"] or x["sym"],
             "exchange": "NSE", "india": True} for x in out]


@router.get("/search")
def search(q: str):
    return {"results": find(q)}


@router.get("/refresh-instruments")
def refresh():
    load_instruments(force=True)
    return {"loaded": len(_instr)}


# ──────────────────────────  CANDLES  ──────────────────────────

def _parse(candles):
    """Upstox: [iso_time, o, h, l, c, volume, oi] — naya pehle aata hai."""
    out = []
    for x in candles:
        dt = datetime.fromisoformat(x[0])
        out.append({"t": int(dt.timestamp()), "o": float(x[1]), "h": float(x[2]),
                    "l": float(x[3]), "c": float(x[4]), "v": int(x[5] or 0)})
    out.sort(key=lambda r: r["t"])
    return out


def _get(url):
    r = requests.get(url, headers=headers(), timeout=20)
    if r.status_code == 401:
        raise HTTPException(401, "Token expire — /api/upstox/login kholo")
    if r.status_code != 200:
        raise HTTPException(502, f"Upstox: {r.status_code} {r.text[:160]}")
    return (r.json().get("data") or {}).get("candles") or []


# Upstox v3 ek request me: 1-15 min -> 1 mahina, 30/60 min -> 1 quarter.
# Forecast model ko training ke liye jitna ho sake utna data chahiye.
DAYS_BACK = {"1m": 28, "5m": 28, "15m": 28, "30m": 85, "60m": 85, "1d": 400}


def upstox_candles(instrument_key, interval="5m", days_back=None):
    """app.py ke yahoo_candles() jaisa hi output deta hai."""
    if interval not in UNITS:
        raise HTTPException(400, "Ye timeframe support nahi hai")
    unit, step = UNITS[interval]
    ik = requests.utils.quote(instrument_key, safe="")

    # aaj ke candles
    rows = _parse(_get(f"{API}/v3/historical-candle/intraday/{ik}/{unit}/{step}"))

    # pichhle dinon ke candles (indicators ko warm-up chahiye)
    to_d = datetime.now(IST).date()
    from_d = to_d - timedelta(days=days_back or DAYS_BACK[interval])
    try:
        old = _parse(_get(f"{API}/v3/historical-candle/{ik}/{unit}/{step}/{to_d}/{from_d}"))
        seen = {r["t"] for r in rows}
        rows = sorted([r for r in old if r["t"] not in seen] + rows, key=lambda r: r["t"])
    except HTTPException:
        pass  # sirf intraday se kaam chala lo

    # Yahoo wala hi fix: aakhri candle abhi ban raha ho to hatao
    dropped = False
    if rows and unit == "minutes" and time.time() - rows[-1]["t"] < step * 60:
        rows.pop()
        dropped = True

    if len(rows) < 120:
        raise HTTPException(422, "Itne candles nahi mile ki analysis ho sake")

    name = instrument_key
    for x in load_instruments():
        if x["key"] == instrument_key:
            name = x["name"] or x["sym"]
            break

    try:
        ltp = quote(instrument_key)
    except Exception:
        ltp = None
    return {"rows": rows, "name": name, "currency": "INR", "exchange": "NSE",
            "last": rows[-1]["c"], "prev_close": None, "state": "",
            "ltp": ltp, "dropped_forming": dropped}


def quote(instrument_key):
    """Live LTP — candle se thoda aage ka price."""
    r = requests.get(f"{API}/v2/market-quote/ltp",
                     headers=headers(), params={"instrument_key": instrument_key}, timeout=10)
    if r.status_code != 200:
        return None
    for v in (r.json().get("data") or {}).values():
        return v.get("last_price")
    return None