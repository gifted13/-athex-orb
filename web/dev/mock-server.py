"""Εικονικός διακομιστής (mock) για ανάπτυξη του UI χωρίς το πραγματικό backend.

Σερβίρει τον φάκελο web/ και απαντά σε ΟΛΑ τα endpoints του συμβολαίου HTTP
(docs/ARCHITECTURE.md) με ρεαλιστικά αλλά ψεύτικα δεδομένα (λειτουργία επίδειξης).
Τα δεδομένα είναι ντετερμινιστικά (σταθερός σπόρος ανά μετοχή).

Χρήση:
    python web/dev/mock-server.py                 # http://127.0.0.1:8765
    python web/dev/mock-server.py --mode empty    # χωρίς δεδομένα (onboarding)
    python web/dev/mock-server.py --latency 400   # τεχνητή καθυστέρηση (ms) για να φανούν τα skeletons
    python web/dev/mock-server.py --api-down      # κάθε /api/* απαντά 503
    python web/dev/mock-server.py --no-backtest   # backtest μη διαθέσιμο
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import math
import random
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"
sys.path.insert(0, str(ROOT))

from fastapi import Body, FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from athex_orb.config import load_config  # noqa: E402
from athex_orb.universe import DEFAULT_UNIVERSE, Stock, normalize_ticker, search as uni_search  # noqa: E402

# ---------------------------------------------------------------------------
# Ρυθμίσεις mock
# ---------------------------------------------------------------------------
OPTS = argparse.Namespace(mode="demo", latency=0, api_down=False, no_backtest=False)

BASE_PRICE = {
    "ETE.AT": 12.4, "EUROB.AT": 3.05, "TPEIR.AT": 6.6, "ALPHA.AT": 3.4, "BOCHGR.AT": 6.1, "OPTIMA.AT": 7.3,
    "PPC.AT": 16.2, "MOH.AT": 24.8, "ELPE.AT": 8.1, "ADMIE.AT": 3.1, "MTLN.AT": 41.5, "CENER.AT": 11.2,
    "VIO.AT": 7.0, "ELHA.AT": 2.6, "TITC.AT": 41.0, "PLAKR.AT": 16.9, "GEKTERNA.AT": 20.6, "AKTR.AT": 6.2,
    "AIA.AT": 12.1, "PPA.AT": 36.0, "OLTH.AT": 30.5, "EYDAP.AT": 6.5, "EYAPS.AT": 9.1, "HTO.AT": 16.3,
    "OPAP.AT": 19.2, "BELA.AT": 29.4, "FOYRK.AT": 4.25, "SAR.AT": 13.1, "KRI.AT": 15.4, "AEGN.AT": 12.3,
    "LAMDA.AT": 7.1,
}
# μέσος ημερήσιος τζίρος (€) — μερικές μικρές εταιρείες κάτω από το όριο ρευστότητας
BASE_ADV = {
    "ETE.AT": 38e6, "EUROB.AT": 30e6, "TPEIR.AT": 28e6, "ALPHA.AT": 25e6, "BOCHGR.AT": 6e6, "OPTIMA.AT": 3.5e6,
    "PPC.AT": 22e6, "MOH.AT": 9e6, "ELPE.AT": 5e6, "ADMIE.AT": 1.2e6, "MTLN.AT": 18e6, "CENER.AT": 4e6,
    "VIO.AT": 1.5e6, "ELHA.AT": 0.9e6, "TITC.AT": 3e6, "PLAKR.AT": 0.12e6, "GEKTERNA.AT": 4.5e6, "AKTR.AT": 2.5e6,
    "AIA.AT": 7e6, "PPA.AT": 0.6e6, "OLTH.AT": 0.18e6, "EYDAP.AT": 0.8e6, "EYAPS.AT": 0.35e6, "HTO.AT": 12e6,
    "OPAP.AT": 14e6, "BELA.AT": 6e6, "FOYRK.AT": 1.1e6, "SAR.AT": 1.3e6, "KRI.AT": 0.55e6, "AEGN.AT": 2.2e6,
    "LAMDA.AT": 2.8e6,
}
NO_DATA = {"EYAPS.AT"}  # μία μετοχή χωρίς δεδομένα, για να ελέγχονται τα null

FEATURE_LABELS = {
    "ret_5": "Απόδοση 5 ημερών", "ret_20": "Απόδοση 20 ημερών", "ret_60": "Απόδοση 60 ημερών",
    "dist_sma50": "Απόσταση από ΚΜΟ 50", "dist_sma200": "Απόσταση από ΚΜΟ 200", "rsi_14": "RSI 14",
    "vol_ratio": "Όγκος έναντι μέσου", "value_z": "Τζίρος (z-score)", "range_pos_252": "Θέση στο εύρος 52 εβδ.",
    "rel_mkt_20": "Σχετική απόδοση έναντι αγοράς", "mkt_trend": "Τάση αγοράς", "atr_pct": "Μεταβλητότητα (ATR)",
}

STATE: dict = {
    "custom": [],          # list[Stock]
    "favorites": ["OPAP.AT", "PPC.AT", "ETE.AT", "MTLN.AT"],
    "jobs": {},            # id -> job dict
    "settings": None,
    "cache": {},
    "seed_bump": 0,        # αλλάζει μετά από κάθε «ενημέρωση» για να φαίνονται αλλαγές
}
LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Βοηθητικά
# ---------------------------------------------------------------------------
def seed_of(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16) + STATE["seed_bump"]


def last_business_day() -> dt.date:
    d = dt.date.today() - dt.timedelta(days=1)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


def business_days(end: dt.date, n: int) -> list[dt.date]:
    out = []
    d = end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= dt.timedelta(days=1)
    return out[::-1]


def business_days_between(start: dt.date, end: dt.date) -> list[dt.date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def add_business_days(d: dt.date, n: int) -> dt.date:
    while n > 0:
        d += dt.timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def iso(d) -> str | None:
    return d.isoformat() if d else None


def r4(x: float) -> float:
    return round(float(x), 6)


def all_stocks() -> list[Stock]:
    return list(DEFAULT_UNIVERSE) + list(STATE["custom"])


def find_stock(ticker: str) -> Stock | None:
    for s in all_stocks():
        if s.ticker == ticker:
            return s
    return None


def mode() -> str:
    return OPTS.mode


def has_data(s: Stock) -> bool:
    return mode() != "empty" and s.ticker not in NO_DATA


# ---------------------------------------------------------------------------
# Συνθετικά δεδομένα ανά μετοχή
# ---------------------------------------------------------------------------
def candles_for(s: Stock, n: int = 520) -> list[dict]:
    key = ("candles", s.ticker, STATE["seed_bump"])
    if key in STATE["cache"]:
        return STATE["cache"][key]
    rng = random.Random(seed_of("px" + s.ticker))
    base = BASE_PRICE.get(s.ticker, rng.uniform(2, 30))
    adv = BASE_ADV.get(s.ticker, rng.uniform(0.2e6, 3e6))
    sigma = rng.uniform(0.011, 0.024)
    drift = rng.uniform(-0.0004, 0.0009)
    days = business_days(last_business_day(), n)
    # ξεκινάμε από τιμή τέτοια ώστε το τέλος να είναι κοντά στη βάση
    price = base * math.exp(-drift * n) * rng.uniform(0.85, 1.15)
    out = []
    regime = 0.0
    for d in days:
        regime = 0.98 * regime + rng.gauss(0, 0.00025)
        gap = rng.gauss(0, sigma * 0.25)
        o = price * (1 + gap)
        ret = drift + regime + rng.gauss(0, sigma)
        c = max(0.05, o * (1 + ret))
        hi = max(o, c) * (1 + abs(rng.gauss(0, sigma * 0.45)))
        lo = min(o, c) * (1 - abs(rng.gauss(0, sigma * 0.45)))
        vol = max(100.0, adv / c * math.exp(rng.gauss(0, 0.45)))
        out.append({"d": iso(d), "o": round(o, 4), "h": round(hi, 4), "l": round(lo, 4),
                    "c": round(c, 4), "v": round(vol)})
        price = c
    # κλιμάκωση ώστε η τελευταία τιμή να είναι κοντά στη ρεαλιστική βάση
    k = base * rng.uniform(0.94, 1.06) / out[-1]["c"]
    for row in out:
        for f in ("o", "h", "l", "c"):
            row[f] = round(row[f] * k, 4)
        row["v"] = round(row["v"] / k)
    STATE["cache"][key] = out
    return out


def atr_pct(candles: list[dict], n: int = 14) -> float:
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i]["h"], candles[i]["l"], candles[i - 1]["c"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = sum(trs[-n:]) / max(1, len(trs[-n:]))
    return atr / candles[-1]["c"]


def adv20(candles: list[dict]) -> float:
    last = candles[-20:]
    return sum(c["c"] * c["v"] for c in last) / len(last)


def orb_for(s: Stock) -> dict | None:
    if not has_data(s):
        return None
    rng = random.Random(seed_of("orb" + s.ticker))
    tilt = rng.uniform(-1, 1)
    p_tp = 0.33 + 0.13 * tilt + rng.uniform(-0.04, 0.04)
    p_sl = 0.44 - 0.12 * tilt + rng.uniform(-0.04, 0.04)
    p_to = max(0.06, 1 - p_tp - p_sl)
    z = p_tp + p_sl + p_to
    p_tp, p_sl, p_to = p_tp / z, p_sl / z, p_to / z
    c = candles_for(s)
    a = atr_pct(c)
    adv = adv20(c)
    tp_ret, sl_ret, timeout_mean = 2.0 * a, 1.0 * a, 0.0015
    cost_rt = 2 * (0.0013 + 0.001 * max(1, min(8, math.sqrt(5e6 / adv)))) + 0.001
    ev_gross = p_tp * tp_ret - p_sl * sl_ret + p_to * timeout_mean
    ev_net = ev_gross - cost_rt
    probs = {"tp": p_tp, "sl": p_sl, "to": p_to}
    return {
        "p_tp": r4(p_tp), "p_sl": r4(p_sl), "p_to": r4(p_to), "ev_net": r4(ev_net),
        "dominant": max(probs, key=probs.get),
        "_ev_gross": ev_gross, "_cost_rt": cost_rt, "_tp_ret": tp_ret, "_sl_ret": sl_ret, "_atr": a, "_adv": adv,
    }


def public_orb(o: dict | None) -> dict | None:
    if o is None:
        return None
    return {k: v for k, v in o.items() if not k.startswith("_")}


def decision_for(s: Stock, o: dict | None) -> tuple[str, str]:
    if o is None:
        return "none", "Δεν υπάρχουν αρκετά δεδομένα για πρόβλεψη."
    if o["_adv"] < 300_000:
        return "blocked", "Χαμηλή ρευστότητα: μέσος τζίρος 20ημ. κάτω από 300.000 €."
    if o["_atr"] > 0.08:
        return "blocked", "Υπερβολική μεταβλητότητα (ATR/τιμή > 8%)."
    if o["ev_net"] >= 0.003:
        return "take", "Θετική αναμενόμενη αξία μετά τα κόστη και όλοι οι έλεγχοι πέρασαν."
    if o["ev_net"] > 0:
        return "skip", "Το καθαρό EV είναι θετικό αλλά κάτω από το ελάχιστο όριο 0,30%."
    return "skip", "Αρνητική αναμενόμενη αξία μετά τα κόστη."


def item_for(s: Stock) -> dict:
    hd = has_data(s)
    o = orb_for(s) if hd else None
    dec, _ = decision_for(s, o)
    if not hd:
        return {"ticker": s.ticker, "name": s.name, "sector": s.sector, "custom": s.custom, "has_data": False,
                "last_close": None, "change_1d": None, "change_5d": None, "spark": [], "adv20_eur": None,
                "orb": None, "decision": "none", "favorite": s.ticker in STATE["favorites"]}
    c = candles_for(s)
    closes = [x["c"] for x in c]
    return {
        "ticker": s.ticker, "name": s.name, "sector": s.sector, "custom": s.custom, "has_data": True,
        "last_close": closes[-1], "change_1d": r4(closes[-1] / closes[-2] - 1),
        "change_5d": r4(closes[-1] / closes[-6] - 1), "spark": [round(x, 4) for x in closes[-60:]],
        "adv20_eur": round(adv20(c), 2), "orb": public_orb(o), "decision": dec,
        "favorite": s.ticker in STATE["favorites"],
    }


def orb_history_for(s: Stock, o: dict, days: list[str]) -> list[dict]:
    rng = random.Random(seed_of("hist" + s.ticker))
    n = len(days)
    a = [0.0] * n
    b = [0.0] * n
    x, y = rng.gauss(0, 0.3), rng.gauss(0, 0.3)
    for i in range(n):
        x = 0.94 * x + rng.gauss(0, 0.09)
        y = 0.94 * y + rng.gauss(0, 0.09)
        a[i], b[i] = x, y
    # ώστε η τελευταία τιμή να ταιριάζει με την τρέχουσα σφαίρα
    lt, ls = math.log(o["p_tp"] / o["p_to"]), math.log(o["p_sl"] / o["p_to"])
    out = []
    for i, d in enumerate(days):
        w = i / max(1, n - 1)
        u = lt + a[i] - a[-1] * w
        v = ls + b[i] - b[-1] * w
        et, es = math.exp(u), math.exp(v)
        z = 1 + et + es
        out.append({"d": d, "p_tp": r4(et / z), "p_sl": r4(es / z), "p_to": r4(1 / z)})
    return out


def agents_for(s: Stock, o: dict | None, item: dict, plan: dict | None, dec: str, reason: str) -> list[dict]:
    rng = random.Random(seed_of("ag" + s.ticker))
    last = last_business_day()
    if o is None:
        base = [
            ("scout", "Ανιχνευτής", "Δεδομένα & ποιότητα", "warn", "Δεν υπάρχουν δεδομένα τιμών για τη μετοχή."),
            ("market", "Χαρτογράφος", "Κλίμα αγοράς & κλάδου", "info", "Το κλίμα της αγοράς είναι ουδέτερο."),
            ("judge", "Κριτής", "Χαρακτηριστικά & μοντέλο", "info", "Χωρίς χαρακτηριστικά, δεν γίνεται πρόβλεψη."),
            ("quant", "Ποσοτικός", "Αναμενόμενη αξία μετά τα κόστη", "info", "Δεν υπολογίζεται EV χωρίς πρόβλεψη."),
            ("guard", "Φύλακας", "Έλεγχοι κινδύνου", "block", "Μπλοκ: δεν υπάρχει επαρκές ιστορικό."),
            ("executor", "Εκτελεστής", "Σχέδιο θέσης (paper)", "info", "Κανένα σχέδιο θέσης."),
        ]
        agents = [{"id": i, "name_el": n, "role_el": r, "status": st, "headline_el": h, "metrics": [], "notes_el": []}
                  for i, n, r, st, h in base]
        agents[4]["checks"] = [{"id": "history", "passed": False, "label_el": "Επαρκές ιστορικό",
                                "detail_el": "0 ημέρες (απαιτούνται ≥ 260)", "blocking": True}]
        agents[2]["drivers"] = []
        return agents

    c = candles_for(s)
    closes = [x["c"] for x in c]
    sma50 = sum(closes[-50:]) / 50
    lo252, hi252 = min(closes[-252:]), max(closes[-252:])
    mkt20 = rng.uniform(-0.04, 0.06)
    regime = "Ανοδικό" if mkt20 > 0.015 else ("Πτωτικό" if mkt20 < -0.02 else "Ουδέτερο")
    sec_rel = rng.uniform(-0.05, 0.05)
    adv = o["_adv"]
    feats = list(FEATURE_LABELS)
    rng.shuffle(feats)
    drivers = [{"feature": f, "label_el": FEATURE_LABELS[f], "contribution": round(rng.uniform(-0.45, 0.45), 3)}
               for f in feats[:6]]
    drivers.sort(key=lambda d: -abs(d["contribution"]))
    checks = [
        {"id": "liquidity", "passed": adv >= 300_000, "label_el": "Ρευστότητα",
         "detail_el": f"Μέσος τζίρος 20ημ. {adv/1e6:.2f} εκ. € (όριο 0,30 εκ. €)".replace(".", ","), "blocking": True},
        {"id": "price", "passed": closes[-1] >= 0.5, "label_el": "Ελάχιστη τιμή",
         "detail_el": "Τιμή πάνω από 0,50 €", "blocking": True},
        {"id": "volatility", "passed": o["_atr"] <= 0.08, "label_el": "Μεταβλητότητα",
         "detail_el": f"ATR/τιμή {o['_atr']*100:.1f}% (όριο 8%)".replace(".", ","), "blocking": True},
        {"id": "history", "passed": True, "label_el": "Επαρκές ιστορικό",
         "detail_el": f"{len(c)} ημέρες (απαιτούνται ≥ 260)", "blocking": True},
        {"id": "crash", "passed": True, "label_el": "Χωρίς κατάρρευση αγοράς",
         "detail_el": "Ο δείκτης δεν έπεσε > 8% τις τελευταίες 10 ημέρες", "blocking": True},
        {"id": "ev", "passed": o["ev_net"] >= 0.003, "label_el": "Ελάχιστο καθαρό EV",
         "detail_el": f"EV καθαρό {o['ev_net']*100:+.2f}% (όριο +0,30%)".replace(".", ","), "blocking": True},
        {"id": "positions", "passed": True, "label_el": "Όρια χαρτοφυλακίου",
         "detail_el": "2 από 5 θέσεις ανοιχτές · 0 στον κλάδο", "blocking": False},
    ]
    guard_status = "ok" if all(ch["passed"] for ch in checks if ch["blocking"]) else (
        "block" if dec == "blocked" else "warn")
    exec_metrics = []
    if plan:
        exec_metrics = [
            {"label_el": "Είσοδος (αναφ.)", "value": plan["entry_ref"], "fmt": "eur"},
            {"label_el": "Στόχος", "value": plan["tp_price"], "fmt": "eur"},
            {"label_el": "Stop", "value": plan["sl_price"], "fmt": "eur"},
            {"label_el": "Ποσό", "value": plan["size_eur"], "fmt": "eur"},
        ]
    return [
        {"id": "scout", "name_el": "Ανιχνευτής", "role_el": "Δεδομένα & ποιότητα", "status": "ok",
         "headline_el": f"Πλήρη δεδομένα έως {last.strftime('%d/%m/%Y')}",
         "metrics": [
             {"label_el": "Τελευταίο κλείσιμο", "value": closes[-1], "fmt": "eur"},
             {"label_el": "Ημέρες ιστορικού", "value": len(c), "fmt": "int"},
             {"label_el": "Μέσος τζίρος 20ημ.", "value": adv, "fmt": "eur", "hint_el": "Μέση αξία συναλλαγών"},
             {"label_el": "Τελευταία ημερομηνία", "value": iso(last), "fmt": "date"},
         ],
         "notes_el": ["Οι τιμές είναι προσαρμοσμένες για μερίσματα και splits.",
                      "Δεν εντοπίστηκαν κενά ή διπλές εγγραφές."]},
        {"id": "market", "name_el": "Χαρτογράφος", "role_el": "Κλίμα αγοράς & κλάδου",
         "status": "info" if regime != "Πτωτικό" else "warn",
         "headline_el": f"{regime} καθεστώς αγοράς · ο κλάδος {'υπεραποδίδει' if sec_rel > 0 else 'υστερεί'}",
         "metrics": [
             {"label_el": "Δείκτης 20ημ.", "value": round(mkt20, 4), "fmt": "pct"},
             {"label_el": "Μεταβλητότητα δείκτη", "value": round(rng.uniform(0.008, 0.016), 4), "fmt": "pct"},
             {"label_el": "Πτώση 10ημ.", "value": round(-rng.uniform(0, 0.04), 4), "fmt": "pct"},
             {"label_el": f"{s.sector} vs αγορά", "value": round(sec_rel, 4), "fmt": "pct"},
         ],
         "notes_el": [f"Καθεστώς: {regime.lower()} (δείκτης έναντι ΚΜΟ 100).",
                      "Ο κλάδος συγκρίνεται με τον μέσο όρο των μετοχών του ίδιου κλάδου."]},
        {"id": "judge", "name_el": "Κριτής", "role_el": "Χαρακτηριστικά & μοντέλο", "status": "info",
         "headline_el": ("Τα χαρακτηριστικά ευνοούν τον στόχο" if o["dominant"] == "tp"
                         else "Τα χαρακτηριστικά γέρνουν προς το stop" if o["dominant"] == "sl"
                         else "Αναμένεται πλάγια κίνηση"),
         "metrics": [
             {"label_el": "RSI 14", "value": round(rng.uniform(32, 70), 1), "fmt": "num"},
             {"label_el": "Απόσταση από ΚΜΟ 50", "value": round(closes[-1] / sma50 - 1, 4), "fmt": "pct"},
             {"label_el": "ATR / τιμή", "value": round(o["_atr"], 4), "fmt": "pct"},
             {"label_el": "Θέση στο εύρος 52 εβδ.", "value": round((closes[-1] - lo252) / (hi252 - lo252), 4),
              "fmt": "pct"},
         ],
         "drivers": drivers,
         "notes_el": ["Λογιστική παλινδρόμηση 3 καταστάσεων, walk-forward.",
                      "Οι συνεισφορές αφορούν το log-odds στόχου έναντι stop."]},
        {"id": "quant", "name_el": "Ποσοτικός", "role_el": "Αναμενόμενη αξία μετά τα κόστη",
         "status": "ok" if o["ev_net"] >= 0.003 else "warn",
         "headline_el": f"EV καθαρό {o['ev_net']*100:+.2f}% ανά συναλλαγή".replace(".", ","),
         "metrics": [
             {"label_el": "EV μικτό", "value": round(o["_ev_gross"], 5), "fmt": "pct"},
             {"label_el": "Κόστος (μετ' επιστροφής)", "value": round(o["_cost_rt"], 5), "fmt": "pct",
              "hint_el": "Προμήθειες, χρεώσεις, spread, επίδραση"},
             {"label_el": "EV καθαρό", "value": o["ev_net"], "fmt": "pct"},
             {"label_el": "Νεκρό σημείο P(στόχος)", "value": round(min(0.95, o["p_tp"] - o["ev_net"] / (o["_tp_ret"] + o["_sl_ret"])), 4), "fmt": "pct"},
         ],
         "notes_el": [f"Στόχος +{o['_tp_ret']*100:.2f}% · Stop −{o['_sl_ret']*100:.2f}%".replace(".", ","),
                      "Το spread εκτιμάται από τη ρευστότητα της μετοχής."]},
        {"id": "guard", "name_el": "Φύλακας", "role_el": "Έλεγχοι κινδύνου", "status": guard_status,
         "headline_el": ("Όλοι οι έλεγχοι πέρασαν" if guard_status == "ok"
                         else "Μπλοκ: " + reason.split(":")[0].lower() if guard_status == "block"
                         else "Το EV δεν περνά το ελάχιστο όριο"),
         "metrics": [], "checks": checks,
         "notes_el": ["Οι έλεγχοι με ✕ που είναι δεσμευτικοί μπλοκάρουν τη θέση."]},
        {"id": "executor", "name_el": "Εκτελεστής", "role_el": "Σχέδιο θέσης (paper)",
         "status": "ok" if dec == "take" else "info",
         "headline_el": ("Αγορά στο άνοιγμα της επόμενης συνεδρίασης" if dec == "take"
                         else "Καμία θέση — παρακολούθηση"),
         "metrics": exec_metrics,
         "notes_el": ["Paper trading: δεν αποστέλλεται καμία εντολή.",
                      "Έξοδος στο πρώτο από: στόχος, stop ή λήξη χρόνου."]},
    ]


def stock_payload(s: Stock, days: int = 250) -> dict:
    item = item_for(s)
    base = {"ticker": s.ticker, "name": s.name, "sector": s.sector, "custom": s.custom,
            "favorite": s.ticker in STATE["favorites"], "has_data": item["has_data"],
            "as_of": iso(last_business_day()) if item["has_data"] else None}
    if not item["has_data"]:
        base.update({"candles": [], "orb_history": [], "orb": None, "plan": None, "decision": "none",
                     "decision_reason_el": "Δεν υπάρχουν δεδομένα για αυτή τη μετοχή ακόμη.",
                     "agents": agents_for(s, None, item, None, "none", ""), "outcomes": []})
        return base
    o = orb_for(s)
    c = candles_for(s)[-days:]
    dec, reason = decision_for(s, o)
    entry = c[-1]["c"]
    size = min(2000.0, 0.01 * o["_adv"]) if dec == "take" else min(2000.0, 0.01 * o["_adv"])
    plan = {"entry_ref": entry, "tp_price": round(entry * (1 + o["_tp_ret"]), 4),
            "sl_price": round(entry * (1 - o["_sl_ret"]), 4), "horizon_days": 10,
            "size_eur": round(size, 2), "shares": round(size / entry, 2)}
    if dec == "blocked":
        plan["size_eur"] = None
        plan["shares"] = None
    rng = random.Random(seed_of("out" + s.ticker))
    outcomes = []
    d = last_business_day() - dt.timedelta(days=16)
    for _ in range(20):
        lab = rng.choices([0, 1, 2], weights=[o["p_tp"], o["p_sl"], o["p_to"]])[0]
        g = (o["_tp_ret"] if lab == 0 else -o["_sl_ret"] if lab == 1 else rng.uniform(-0.01, 0.012))
        outcomes.append({"signal_date": iso(d), "label": lab, "gross_ret": r4(g)})
        d -= dt.timedelta(days=rng.choice([3, 5, 7, 8, 12]))
    base.update({
        "candles": c,
        "orb_history": orb_history_for(s, o, [x["d"] for x in c]),
        "orb": public_orb(o), "plan": plan, "decision": dec, "decision_reason_el": reason,
        "agents": agents_for(s, o, item, plan, dec, reason), "outcomes": outcomes,
    })
    return base


# ---------------------------------------------------------------------------
# Backtest / ημερολόγιο
# ---------------------------------------------------------------------------
def metrics_from(eq: list[float], rng: random.Random, trades: int | None) -> dict:
    rets = [eq[i] / eq[i - 1] - 1 for i in range(1, len(eq))]
    mean = sum(rets) / len(rets)
    sd = math.sqrt(sum((r - mean) ** 2 for r in rets) / len(rets))
    peak, mdd = eq[0], 0.0
    for v in eq:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    years = len(eq) / 252
    total = eq[-1] / eq[0] - 1
    m = {"total_return": r4(total), "cagr": r4((1 + total) ** (1 / years) - 1), "max_drawdown": r4(mdd),
         "sharpe": round(mean / sd * math.sqrt(252), 3) if sd else None, "volatility": r4(sd * math.sqrt(252))}
    if trades is None:
        m.update({"n_trades": None, "win_rate": None, "avg_win": None, "avg_loss": None, "profit_factor": None,
                  "avg_bars_held": None, "total_costs_eur": 0.0, "exposure": 1.0})
    else:
        m.update({"n_trades": trades, "win_rate": r4(rng.uniform(0.38, 0.47)), "avg_win": r4(rng.uniform(0.03, 0.045)),
                  "avg_loss": r4(-rng.uniform(0.017, 0.024)), "profit_factor": round(rng.uniform(0.95, 1.25), 3),
                  "avg_bars_held": round(rng.uniform(5.2, 7.4), 2), "total_costs_eur": round(trades * rng.uniform(9, 14), 2),
                  "exposure": r4(rng.uniform(0.55, 0.8))})
    return m


def backtest_payload() -> dict:
    if OPTS.no_backtest or mode() == "empty":
        return {"available": False}
    key = ("bt", STATE["seed_bump"])
    if key in STATE["cache"]:
        return STATE["cache"][key]
    rng = random.Random(seed_of("backtest"))
    days = business_days_between(dt.date(2019, 1, 2), last_business_day())
    cap = 10000.0
    s, b, u = [cap], [cap], [cap]
    for _ in days[1:]:
        m = rng.gauss(0.00028, 0.0105)
        b.append(b[-1] * (1 + m))
        u.append(u[-1] * (1 + m + 0.00008))      # ισοσταθμική λίστα: + μερίσματα (δείκτης τιμών)
        s.append(s[-1] * (1 + 0.45 * m + rng.gauss(0.00009, 0.0055)))
    n = len(days)
    med, p05, p95 = [], [], []
    for i in range(n):
        t = i / 252
        center = cap * (1 + 0.012 * t + 0.035 * math.sin(i / 160.0) * min(1, t))
        spread = cap * 0.11 * math.sqrt(t + 1e-9)
        med.append(center)
        p05.append(center - spread * 0.95)
        p95.append(center + spread * 1.05)
    equity = [{"d": iso(d), "strategy": round(s[i], 2), "buy_hold": round(b[i], 2), "buy_hold_universe": round(u[i], 2),
               "random_median": round(med[i], 2), "random_p05": round(p05[i], 2), "random_p95": round(p95[i], 2)}
              for i, d in enumerate(days)]
    peak, dd = s[0], []
    for i, d in enumerate(days):
        peak = max(peak, s[i])
        dd.append({"d": iso(d), "strategy": r4(s[i] / peak - 1)})
    n_trades = 418
    stocks = [x for x in DEFAULT_UNIVERSE if x.ticker not in NO_DATA]
    trades = []
    for k in range(300):
        st = rng.choice(stocks)
        i = int((k + 1) / 301 * (n - 15))
        lab = rng.choices([0, 1, 2], weights=[0.34, 0.44, 0.22])[0]
        ep = BASE_PRICE.get(st.ticker, 10) * rng.uniform(0.8, 1.2)
        a = rng.uniform(0.012, 0.035)
        g = 2 * a if lab == 0 else (-a if lab == 1 else rng.uniform(-0.015, 0.02))
        held = rng.randint(1, 10) if lab != 2 else 10
        size = round(rng.uniform(800, 2000), 2)
        cost = round(size * rng.uniform(0.004, 0.009), 2)
        net_pnl = size * g - cost
        trades.append({
            "ticker": st.ticker, "sector": st.sector, "signal_date": iso(days[i]), "entry_date": iso(days[i + 1]),
            "entry_price": round(ep, 4), "exit_date": iso(days[min(n - 1, i + held)]),
            "exit_price": round(ep * (1 + g), 4), "label": lab, "size_eur": size, "gross_ret": r4(g),
            "cost_eur": cost, "net_pnl_eur": round(net_pnl, 2), "net_ret": r4(net_pnl / size),
            "p_tp": r4(rng.uniform(0.3, 0.55)), "p_sl": r4(rng.uniform(0.25, 0.45)), "p_to": r4(rng.uniform(0.1, 0.25)),
            "ev_net": r4(rng.uniform(0.003, 0.012)),
        })
    trades[-1]["label"] = -1  # ανοιχτή θέση στο τέλος
    calib = []
    for j in range(8):
        lo, hi = 0.15 + j * 0.05, 0.2 + j * 0.05
        pred = (lo + hi) / 2 + rng.uniform(-0.01, 0.01)
        calib.append({"bin_lo": round(lo, 3), "bin_hi": round(hi, 3), "predicted": r4(pred),
                      "realized": r4(max(0, min(1, pred + rng.uniform(-0.07, 0.05)))),
                      "n": int(rng.uniform(40, 900) * math.exp(-((j - 3) ** 2) / 6))})
    ms = metrics_from(s, rng, n_trades)
    mb = metrics_from(b, rng, None)
    mu = metrics_from(u, rng, None)
    mr = metrics_from(med, rng, n_trades)
    mr["total_return"] = r4(med[-1] / cap - 1)
    pct = 0.71
    payload = {
        "available": True, "generated_at": dt.datetime.now().replace(microsecond=0).isoformat(),
        "period": {"start": iso(days[0]), "end": iso(days[-1])},
        "params": {"horizon_days": 10, "k_tp": 2.0, "k_sl": 1.0, "capital_eur": cap, "position_eur_max": 2000,
                   "max_positions": 5, "random_runs": 200, "model_kind": "logreg"},
        "metrics": {"strategy": ms, "buy_hold": mb, "buy_hold_universe": mu, "random_median": mr},
        "skill": {"n": 41250, "n_dates": 1475, "logloss_model": 0.9412, "logloss_base": 0.9338, "brier_model": 0.552,
                  "brier_base": 0.548, "improvement": -0.0079, "p_value": 0.94, "ok": False, "block_days": 20},
        "equity": equity, "drawdown": dd, "trades": trades, "calibration": calib,
        "by_label": {"tp": 142, "sl": 183, "to": 93},
        "verdict": {"edge": False, "percentile": pct, "p_value": 0.29, "level": "none", "skill_ok": False,
                    "text_el": (f"Η στρατηγική ξεπέρασε το {round(pct*100)}% των τυχαίων στρατηγικών — όχι αρκετά "
                                "για να αποκλειστεί η τύχη. Εκτός δείγματος οι πιθανότητες του μοντέλου δεν είναι "
                                "καλύτερες από τις σταθερές ιστορικές συχνότητες.")},
    }
    STATE["cache"][key] = payload
    return payload


def journal_payload() -> dict:
    if mode() == "empty":
        return {"items": [], "summary": {"total": 0}}
    rng = random.Random(seed_of("journal"))
    stocks = [x for x in DEFAULT_UNIVERSE if x.ticker not in NO_DATA]
    items = []
    days = business_days(last_business_day(), 30)
    k = 0
    for di, d in enumerate(reversed(days)):
        for st in rng.sample(stocks, 3):
            k += 1
            take = rng.random() < 0.45
            age = di
            if age <= 1:
                status = "pending" if age == 0 else "open"
            elif age < 10 and rng.random() < 0.4:
                status = "open"
            else:
                status = rng.choices(["tp", "sl", "timeout"], weights=[0.36, 0.42, 0.22])[0]
            if not take and status == "open":
                status = rng.choice(["tp", "sl", "timeout"]) if age > 1 else "pending"
            ref = BASE_PRICE.get(st.ticker, 10) * rng.uniform(0.95, 1.05)
            a = rng.uniform(0.012, 0.03)
            g = {"tp": 2 * a, "sl": -a, "timeout": rng.uniform(-0.01, 0.012)}.get(status)
            items.append({
                "id": k, "signal_date": iso(d), "ticker": st.ticker, "name": st.name,
                "decision": "take" if take else "skip",
                "reason_el": ("Θετικό EV και όλοι οι έλεγχοι πέρασαν" if take else
                              rng.choice(["Αρνητικό EV μετά τα κόστη", "Χαμηλή ρευστότητα",
                                          "EV κάτω από το ελάχιστο όριο", "Όριο θέσεων στον κλάδο"])),
                "p_tp": r4(rng.uniform(0.25, 0.55)), "p_sl": r4(rng.uniform(0.25, 0.5)),
                "p_to": r4(rng.uniform(0.1, 0.3)), "ev_net": r4(rng.uniform(0.003, 0.012) if take else rng.uniform(-0.01, 0.003)),
                "entry_ref": round(ref, 4), "tp_ret": r4(2 * a), "sl_ret": r4(a),
                "size_eur": round(rng.uniform(900, 2000), 2) if take else None, "status": status,
                "entry_date": iso(add_business_days(d, 1)) if status != "pending" else None,
                "entry_price": round(ref * rng.uniform(0.99, 1.01), 4) if status != "pending" else None,
                "exit_date": iso(add_business_days(d, rng.randint(2, 10))) if g is not None else None,
                "exit_price": round(ref * (1 + g), 4) if g is not None else None,
                "gross_ret": r4(g) if g is not None else None,
                "net_ret": r4(g - 0.006) if g is not None else None,
                "created_at": f"{iso(d)}T18:30:00",
            })
    summary = {"total": len(items)}
    for dec in ("take", "skip"):
        sub = [i for i in items if i["decision"] == dec]
        res = [i for i in sub if i["status"] in ("tp", "sl", "timeout")]
        wins = [i for i in res if (i["net_ret"] or 0) > 0]
        summary[dec] = {"n": len(sub), "resolved": len(res),
                        "tp": sum(1 for i in res if i["status"] == "tp"),
                        "sl": sum(1 for i in res if i["status"] == "sl"),
                        "timeout": sum(1 for i in res if i["status"] == "timeout"),
                        "hit_rate": r4(len(wins) / len(res)) if res else None,
                        "avg_net_ret": r4(sum(i["net_ret"] for i in res) / len(res)) if res else None}
    summary["pending"] = sum(1 for i in items if i["status"] == "pending")
    summary["open"] = sum(1 for i in items if i["status"] == "open")
    return {"items": items, "summary": summary}


def settings_payload() -> dict:
    if STATE["settings"] is None:
        cfg = load_config(use_user_settings=False)
        d = cfg.to_dict()
        STATE["settings"] = {"portfolio": d["portfolio"], "barriers": d["barriers"], "guard": d["guard"],
                             "costs": d["costs"], "model": {"kind": d["model"]["kind"]}}
    return STATE["settings"]


# ---------------------------------------------------------------------------
# Εργασίες (jobs) — προσομοίωση προόδου
# ---------------------------------------------------------------------------
JOB_STEPS = {
    "refresh": ["Λήψη δεδομένων τιμών…", "Έλεγχος ποιότητας δεδομένων…", "Υπολογισμός χαρακτηριστικών…",
                "Ετικέτες τριπλού φραγμού…", "Εκπαίδευση μοντέλου (walk-forward)…", "Υπολογισμός σημερινών σημάτων…",
                "Ενημέρωση ημερολογίου…"],
    "backtest": ["Προσομοίωση χαρτοφυλακίου…", "Αγορά & διακράτηση…", "Τυχαίες είσοδοι (Monte Carlo)…",
                 "Βαθμονόμηση πιθανοτήτων…", "Σύνταξη ετυμηγορίας…"],
    "demo": ["Δημιουργία συνθετικών τιμών…", "Υπολογισμός χαρακτηριστικών…", "Εκπαίδευση μοντέλου…",
             "Υπολογισμός σημάτων…", "Backtest…"],
}
JOB_SECONDS = {"refresh": 7.0, "backtest": 5.0, "demo": 6.0}


def job_view(j: dict) -> dict:
    if j["status"] in ("queued", "running"):
        el = time.time() - j["_t0"]
        frac = min(1.0, el / JOB_SECONDS[j["type"]])
        steps = JOB_STEPS[j["type"]]
        if frac >= 1.0:
            j["status"], j["progress"] = "done", 1.0
            j["message_el"] = "Ολοκληρώθηκε"
            j["finished_at"] = dt.datetime.now().replace(microsecond=0).isoformat()
            on_job_done(j)
        else:
            j["status"] = "running" if el > 0.4 else "queued"
            j["progress"] = round(frac, 3)
            j["message_el"] = steps[min(len(steps) - 1, int(frac * len(steps)))]
    return {k: v for k, v in j.items() if not k.startswith("_")}


def on_job_done(j: dict) -> None:
    if j["type"] == "demo":
        OPTS.mode = "demo"
    if j["type"] in ("refresh", "demo"):
        STATE["seed_bump"] += 1
        STATE["cache"].clear()
    if j["type"] == "backtest":
        OPTS.no_backtest = False
        STATE["cache"] = {k: v for k, v in STATE["cache"].items() if k[0] != "bt"}


# ---------------------------------------------------------------------------
# Εφαρμογή
# ---------------------------------------------------------------------------
app = FastAPI(title="ATHEX Orb — mock")


@app.middleware("http")
async def chaos(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        if OPTS.api_down:
            return JSONResponse({"detail": "Ο διακομιστής δεν είναι διαθέσιμος (mock)."}, status_code=503)
        if OPTS.latency:
            await asyncio.sleep(OPTS.latency / 1000)
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/api/status")
def api_status():
    m = mode()
    with LOCK:
        jobs = [job_view(j) for j in STATE["jobs"].values()]
    n_with = 0 if m == "empty" else sum(1 for s in all_stocks() if has_data(s))
    last = last_business_day()
    return {
        "mode": m,
        "data_last_date": None if m == "empty" else iso(last),
        "data_updated_at": None if m == "empty" else dt.datetime.combine(last + dt.timedelta(days=1), dt.time(7, 42)).isoformat(),
        "model_trained_until": None if m == "empty" else iso(last - dt.timedelta(days=12)),
        "n_tickers": len(all_stocks()), "n_with_data": n_with,
        "horizon_days": 10, "k_tp": 2.0, "k_sl": 1.0,
        "jobs": [j for j in jobs if j["status"] in ("queued", "running")],
        "disclaimer_el": ("Εργαλείο έρευνας — όχι επενδυτική συμβουλή. Οι αποδόσεις του παρελθόντος "
                          "δεν εγγυώνται μελλοντικές. Δεν εκτελεί εντολές."),
    }


@app.get("/api/universe")
def api_universe():
    return {"as_of": None if mode() == "empty" else iso(last_business_day()),
            "items": [item_for(s) for s in all_stocks()]}


@app.get("/api/search")
def api_search(q: str = ""):
    found = uni_search(all_stocks(), q, limit=20)
    suggest = None
    t = normalize_ticker(q)
    if q and t and not any(s.ticker == t for s in all_stocks()) and q.strip().isascii():
        suggest = {"ticker": t, "valid": True}
    return {"items": [item_for(s) for s in found], "suggest": suggest}


@app.post("/api/universe")
def api_add(body: dict = Body(...)):
    t = normalize_ticker(str(body.get("ticker", "")))
    if not t:
        return {"ok": False, "error_el": "Μη έγκυρο σύμβολο μετοχής."}
    if find_stock(t):
        return {"ok": False, "error_el": f"Η μετοχή {t} υπάρχει ήδη στη λίστα."}
    if t.startswith("FAIL"):
        return {"ok": False, "error_el": f"Δεν βρέθηκαν δεδομένα για το {t} στο Yahoo Finance."}
    s = Stock(t, body.get("name") or t.removesuffix(".AT"), body.get("sector") or "Άλλο", [], custom=True)
    STATE["custom"].append(s)
    return {"ok": True, "item": item_for(s)}


@app.delete("/api/universe/{ticker}")
def api_remove(ticker: str):
    before = len(STATE["custom"])
    STATE["custom"] = [s for s in STATE["custom"] if s.ticker != ticker]
    if len(STATE["custom"]) == before:
        raise HTTPException(404, "Μόνο μετοχές που προσθέσατε εσείς μπορούν να αφαιρεθούν.")
    if ticker in STATE["favorites"]:
        STATE["favorites"].remove(ticker)
    return {"ok": True}


@app.get("/api/favorites")
def api_favs():
    return {"tickers": list(STATE["favorites"])}


@app.put("/api/favorites/{ticker}")
def api_fav_add(ticker: str):
    if not find_stock(ticker):
        raise HTTPException(404, f"Η μετοχή {ticker} δεν υπάρχει στη λίστα.")
    if ticker not in STATE["favorites"]:
        STATE["favorites"].append(ticker)
    return {"tickers": list(STATE["favorites"])}


@app.delete("/api/favorites/{ticker}")
def api_fav_del(ticker: str):
    if ticker in STATE["favorites"]:
        STATE["favorites"].remove(ticker)
    return {"tickers": list(STATE["favorites"])}


@app.get("/api/stock/{ticker}")
def api_stock(ticker: str, days: int = 250):
    s = find_stock(ticker)
    if s is None:
        raise HTTPException(404, f"Η μετοχή {ticker} δεν βρέθηκε.")
    return stock_payload(s, max(20, min(days, 500)))


@app.get("/api/signals")
def api_signals():
    if mode() == "empty":
        return {"as_of": None, "items": []}
    items = []
    for s in all_stocks():
        if not has_data(s):
            continue
        o = orb_for(s)
        dec, reason = decision_for(s, o)
        entry = candles_for(s)[-1]["c"]
        items.append({"ticker": s.ticker, "name": s.name, "sector": s.sector, "decision": dec, "reason_el": reason,
                      "p_tp": o["p_tp"], "p_sl": o["p_sl"], "p_to": o["p_to"], "ev_net": o["ev_net"],
                      "entry_ref": entry, "tp_price": round(entry * (1 + o["_tp_ret"]), 4),
                      "sl_price": round(entry * (1 - o["_sl_ret"]), 4),
                      "size_eur": round(min(2000.0, 0.01 * o["_adv"]), 2) if dec == "take" else None})
    items.sort(key=lambda x: (x["decision"] != "take", -(x["ev_net"] or 0)))
    return {"as_of": iso(last_business_day()), "items": items}


@app.get("/api/backtest")
def api_backtest():
    return backtest_payload()


@app.get("/api/journal")
def api_journal():
    return journal_payload()


@app.get("/api/settings")
def api_settings():
    return settings_payload()


@app.put("/api/settings")
def api_settings_put(body: dict = Body(...)):
    cur = settings_payload()
    retrain = False
    for section, values in body.items():
        if section in cur and isinstance(values, dict):
            for k, v in values.items():
                if k in cur[section] and cur[section][k] != v:
                    cur[section][k] = v
                    if section in ("barriers", "model"):
                        retrain = True
    return {"ok": True, "retrain_needed": retrain, "settings": cur}


@app.post("/api/jobs")
def api_jobs_post(body: dict = Body(...)):
    typ = body.get("type")
    if typ not in JOB_STEPS:
        raise HTTPException(400, "Άγνωστος τύπος εργασίας.")
    with LOCK:
        for j in STATE["jobs"].values():
            job_view(j)
            if j["status"] in ("queued", "running"):
                if j["type"] == typ:
                    raise HTTPException(409, "Η εργασία εκτελείται ήδη.")
        jid = uuid.uuid4().hex[:10]
        STATE["jobs"][jid] = {"id": jid, "type": typ, "status": "queued", "progress": 0.0,
                              "message_el": "Σε αναμονή…", "started_at": dt.datetime.now().replace(microsecond=0).isoformat(),
                              "finished_at": None, "error": None, "_t0": time.time()}
    return {"job_id": jid}


@app.get("/api/jobs")
def api_jobs():
    with LOCK:
        return {"items": [job_view(j) for j in list(STATE["jobs"].values())[-10:]][::-1]}


@app.get("/api/jobs/{jid}")
def api_job(jid: str):
    with LOCK:
        j = STATE["jobs"].get(jid)
        if not j:
            raise HTTPException(404, "Η εργασία δεν βρέθηκε.")
        return job_view(j)


@app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "DELETE"])
def api_404(rest: str):
    raise HTTPException(404, "Άγνωστο endpoint.")


app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def main() -> None:
    ap = argparse.ArgumentParser(description="Mock διακομιστής για το UI του ATHEX Orb")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--mode", choices=["demo", "live", "empty"], default="demo")
    ap.add_argument("--latency", type=int, default=0, help="τεχνητή καθυστέρηση σε ms")
    ap.add_argument("--api-down", action="store_true", help="κάθε /api/* απαντά 503")
    ap.add_argument("--no-backtest", action="store_true", help="backtest μη διαθέσιμο")
    args = ap.parse_args()
    OPTS.mode, OPTS.latency, OPTS.api_down, OPTS.no_backtest = args.mode, args.latency, args.api_down, args.no_backtest
    import uvicorn
    print(f"ATHEX Orb mock: http://{args.host}:{args.port}  (mode={args.mode})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
