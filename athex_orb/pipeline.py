"""Ενορχήστρωση: δεδομένα → χαρακτηριστικά → ετικέτες → walk‑forward → σήματα → ημερολόγιο → αρχεία κατάστασης.

Ο διακομιστής δεν υπολογίζει τίποτα βαρύ: διαβάζει τα αρχεία JSON που γράφει αυτό το module στο
data/state/ (βλ. ARCHITECTURE.md). Για ταχύτητα κρατάμε και pickles (panel, ετικέτες, προβλέψεις,
μοντέλο, δείκτη) ώστε:
* το backtest να τρέχει χωρίς νέα εκπαίδευση,
* μια αλλαγή ρυθμίσεων φύλακα/κόστους/χαρτοφυλακίου ή μια νέα μετοχή να ενημερώνει τα σήματα σε
  1–3 δευτερόλεπτα (`recompute_signals`), με το ήδη εκπαιδευμένο μοντέλο.

Συνέπεια ρυθμίσεων: οι πιθανότητες του μοντέλου αφορούν τα όρια (k_tp, k_sl, ορίζοντας) με τα οποία
εκπαιδεύτηκε. Αν ο χρήστης αλλάξει όρια χωρίς νέα ενημέρωση, οι γρήγοροι επανυπολογισμοί και το backtest
χρησιμοποιούν τα όρια της εκπαίδευσης (`cfg_for_model`) — ποτέ μείγμα παλιού μοντέλου με νέα όρια.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import pickle
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import benchmarks, executor, explain, features, journal, market, orb_model, quant, scout
from . import labels as labels_mod
from .guard import fmt_num, fmt_pct
from .universe import Stock, add_custom, load_universe

STATE_LOCK = threading.RLock()          # εγγραφές αρχείων κατάστασης (εργασίες + γρήγοροι επανυπολογισμοί)
DEMO_SEED = 7
N_CANDLES = 520                         # ~2 χρόνια συνεδριάσεων ανά μετοχή στο αρχείο της
N_SPARK = 60
N_OUTCOMES = 20
N_BACKTEST_TRADES = 300
PICKLES = ("panel", "labels", "preds", "model", "index", "prices")
FINGERPRINT_KEYS = {
    "barriers": ("k_tp", "k_sl", "horizon_days", "atr_days"),
    "model": ("kind", "min_train_days", "retrain_every_days", "train_window_days"),
}
SETTINGS_SECTIONS = ("portfolio", "barriers", "guard", "costs", "model", "backtest")
DISCLAIMER_EL = ("Εργαλείο έρευνας — όχι επενδυτική συμβουλή. Οι αποδόσεις του παρελθόντος δεν εγγυώνται "
                 "μελλοντικές. Δεν εκτελεί εντολές.")


class PipelineError(RuntimeError):
    """Σφάλμα με μήνυμα στα ελληνικά, κατάλληλο για εμφάνιση στον χρήστη."""


# ---------------------------------------------------------------- διαδρομές & αρχεία

def state_dir(cfg) -> Path:
    return Path(cfg.data_dir) / "state"


def stocks_dir(cfg) -> Path:
    return state_dir(cfg) / "stocks"


def _stock_file(cfg, ticker: str) -> Path:
    return stocks_dir(cfg) / f"{scout._safe_name(ticker)}.json"


def _clean(obj):
    """Μετατροπή σε τύπους JSON: NaN/άπειρο → None, numpy → Python, ημερομηνίες → 'YYYY-MM-DD'."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, np.integer)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, np.datetime64, datetime)):
        return explain.date_iso(obj)
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray, pd.Index)):
        return [_clean(v) for v in obj]
    if obj is pd.NaT:
        return None
    return str(obj)


RAW_KEYS = ("candles", "orb_history")    # λίστες που χτίζονται ήδη με απλούς τύπους (χωρίς NaN)


def _write_json(path: Path, obj, raw_keys=()) -> None:
    """Ατομική εγγραφή JSON. `raw_keys`: κλειδιά του dict που είναι ήδη καθαρά (παραλείπεται το _clean)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw_keys and isinstance(obj, dict):
        data = _clean({k: v for k, v in obj.items() if k not in raw_keys})
        data.update({k: obj[k] for k in raw_keys if k in obj})
    else:
        data = _clean(obj)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def write_state(cfg, name: str, obj) -> None:
    """Γράφει το state/<name>.json (ατομικά)."""
    _write_json(state_dir(cfg) / f"{name}.json", obj)


def load_state(cfg, name: str) -> dict | None:
    """Διαβάζει το state/<name>.json (None αν λείπει ή είναι κατεστραμμένο)."""
    p = state_dir(cfg) / f"{name}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_stock_state(cfg, ticker: str) -> dict | None:
    try:
        return json.loads(_stock_file(cfg, ticker).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_pickle(cfg, name: str, obj) -> None:
    p = state_dir(cfg) / f"{name}.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex[:6]}.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(p)


def load_pickle(cfg, name: str):
    p = state_dir(cfg) / f"{name}.pkl"
    if not p.exists():
        return None
    try:
        with open(p, "rb") as fh:
            return pickle.load(fh)
    except Exception:  # παλιά/κατεστραμμένη cache ⇒ σαν να μην υπάρχει
        return None


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sub(progress, lo: float, hi: float):
    """Υπο‑πρόοδος: αντιστοιχίζει [0, 1] → [lo, hi] της συνολικής προόδου."""
    def report(fraction: float, message_el: str) -> None:
        if progress is not None:
            try:
                progress(lo + (hi - lo) * min(max(float(fraction), 0.0), 1.0), message_el)
            except Exception:  # η ένδειξη προόδου δεν πρέπει ποτέ να σταματά τη δουλειά
                pass
    return report


# ---------------------------------------------------------------- ρυθμίσεις & «αποτυπώματα»

def fingerprint(cfg) -> dict:
    """Οι ρυθμίσεις από τις οποίες εξαρτάται το εκπαιδευμένο μοντέλο."""
    return {sec: {k: getattr(getattr(cfg, sec), k) for k in keys} for sec, keys in FINGERPRINT_KEYS.items()}


def settings_hash(cfg) -> str:
    d = cfg.to_dict()
    blob = json.dumps({s: d[s] for s in SETTINGS_SECTIONS}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def cfg_for_model(cfg, fp: dict | None):
    """Αντίγραφο του cfg με τα όρια/μοντέλο με τα οποία εκπαιδεύτηκε το αποθηκευμένο μοντέλο."""
    out = copy.deepcopy(cfg)
    for sec, values in (fp or {}).items():
        obj = getattr(out, sec, None)
        if obj is None or not isinstance(values, dict):
            continue
        for k, v in values.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
    return out


def needs_retrain(cfg, meta: dict | None) -> bool:
    if not meta or not meta.get("fingerprint"):
        return False
    return _clean(fingerprint(cfg)) != meta["fingerprint"]


# ---------------------------------------------------------------- μορφή δεδομένων για το UI

def _r(x, nd: int = 6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _group(frame: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    """{ticker: γραμμές της μετοχής με index = ημερομηνία}."""
    if frame is None or not len(frame):
        return {}
    return {str(t): g.droplevel("ticker") for t, g in frame.groupby(level="ticker", sort=False)}


def _candles(df: pd.DataFrame, n: int = N_CANDLES) -> list[dict]:
    tail = df.tail(n)
    d = tail.index.strftime("%Y-%m-%d").tolist()
    o, h, lo, c, v = (tail[k].to_numpy("float64") for k in ("open", "high", "low", "close", "volume"))
    return [{"d": d[i], "o": _r(o[i], 4), "h": _r(h[i], 4), "l": _r(lo[i], 4), "c": _r(c[i], 4),
             "v": int(v[i]) if math.isfinite(v[i]) else None} for i in range(len(d))]


def _orb_history(pred_frame: pd.DataFrame | None, live: dict | None, live_date, n: int = N_CANDLES) -> list[dict]:
    out = []
    if pred_frame is not None and len(pred_frame):
        tail = pred_frame.tail(n)
        d = tail.index.strftime("%Y-%m-%d").tolist()
        p = tail[["p_tp", "p_sl", "p_to"]].to_numpy("float64")
        out = [{"d": d[i], "p_tp": _r(p[i, 0], 4), "p_sl": _r(p[i, 1], 4), "p_to": _r(p[i, 2], 4)}
               for i in range(len(d)) if np.isfinite(p[i]).all()]
    if live is not None and live_date is not None:
        ld = explain.date_iso(live_date)
        point = {"d": ld, "p_tp": _r(live["p_tp"], 4), "p_sl": _r(live["p_sl"], 4), "p_to": _r(live["p_to"], 4)}
        if out and out[-1]["d"] == ld:
            out[-1] = point                # η τελευταία ημέρα: η πρόβλεψη του τελικού μοντέλου (ίδια με τη Σφαίρα)
        elif not out or out[-1]["d"] < ld:
            out.append(point)
    return out


def _outcomes(lab_frame: pd.DataFrame | None, pred_frame: pd.DataFrame | None, n: int = N_OUTCOMES) -> list[dict]:
    """Οι τελευταίες `n` επιλυμένες εκβάσεις χωρίς επικάλυψη (κάθε «συναλλαγή» κλείνει πριν την επόμενη)."""
    if lab_frame is None or not len(lab_frame):
        return []
    res = lab_frame[lab_frame["label"].notna() & lab_frame["exit_date"].notna()]
    if not len(res):
        return []
    sig = res.index.to_numpy(dtype="datetime64[us]")
    ex = pd.DatetimeIndex(res["exit_date"]).to_numpy(dtype="datetime64[us]")
    lab = res["label"].to_numpy("float64")
    gross = res["gross_ret"].to_numpy("float64")
    p_tp = None
    if pred_frame is not None and len(pred_frame):
        p_tp = pred_frame["p_tp"].reindex(res.index).to_numpy("float64")
    picked = []
    last_sig = None
    for i in range(len(res) - 1, -1, -1):
        if last_sig is None or ex[i] <= last_sig:
            item = {"signal_date": explain.date_iso(sig[i]), "label": int(lab[i]), "gross_ret": _r(gross[i]),
                    "exit_date": explain.date_iso(ex[i])}
            if p_tp is not None:
                item["p_tp"] = _r(p_tp[i], 4)
            picked.append(item)
            last_sig = sig[i]
            if len(picked) >= n:
                break
    return picked[::-1]


def _public_orb(a: dict) -> dict | None:
    orb = a.get("orb")
    if orb is None:
        return None
    be = (a.get("ev") or {}).get("breakeven_p_tp")
    return {"p_tp": _r(orb["p_tp"]), "p_sl": _r(orb["p_sl"]), "p_to": _r(orb["p_to"]),
            "ev_net": _r(orb.get("ev_net")), "dominant": orb["dominant"],
            "breakeven_p_tp": _r(be) if be is not None and math.isfinite(be) and 0 <= be <= 1 else None}


def _public_plan(plan: dict | None) -> dict | None:
    if plan is None:
        return None
    return {"entry_ref": _r(plan["entry_ref"], 4), "tp_price": _r(plan["tp_price"], 4),
            "sl_price": _r(plan["sl_price"], 4), "horizon_days": plan["horizon_days"],
            "size_eur": _r(plan["size_eur"], 2), "shares": _r(plan["shares"], 4),
            "tp_ret": _r(plan["tp_ret"]), "sl_ret": _r(plan["sl_ret"])}


def _item(stock: Stock, df: pd.DataFrame | None, a: dict | None) -> dict:
    """Γραμμή του `/api/universe` (χωρίς το `favorite`)."""
    base = {"ticker": stock.ticker, "name": stock.name, "sector": stock.sector, "custom": bool(stock.custom),
            "has_data": False, "last_close": None, "change_1d": None, "change_5d": None, "spark": [],
            "adv20_eur": None, "orb": None, "decision": "none", "as_of": None}
    if df is None or not len(df):
        return base
    c = df["close"].to_numpy("float64")
    base.update({
        "has_data": True,
        "last_close": _r(c[-1], 4),
        "change_1d": _r(c[-1] / c[-2] - 1.0) if len(c) > 1 else None,
        "change_5d": _r(c[-1] / c[-6] - 1.0) if len(c) > 5 else None,
        "spark": [_r(x, 4) for x in c[-N_SPARK:]],
        "as_of": explain.date_iso(df.index[-1]),
    })
    if a is not None:
        base["adv20_eur"] = _r(a.get("adv20_eur"), 2)
        base["orb"] = _public_orb(a)
        base["decision"] = a["decision"]
    return base


def _signal(stock: Stock, a: dict) -> dict:
    orb = a.get("orb") or {}
    plan = a.get("plan") or {}
    cost = a["costs"]["total"] if a.get("costs") else None
    return {
        "ticker": stock.ticker, "name": stock.name, "sector": stock.sector, "decision": a["decision"],
        "reason_el": a.get("reason_el") or "", "p_tp": _r(orb.get("p_tp")), "p_sl": _r(orb.get("p_sl")),
        "p_to": _r(orb.get("p_to")), "ev_net": _r(orb.get("ev_net")), "dominant": orb.get("dominant"),
        "entry_ref": _r(a.get("entry_ref"), 4), "tp_price": _r(plan.get("tp_price"), 4),
        "sl_price": _r(plan.get("sl_price"), 4), "size_eur": _r(plan.get("size_eur"), 2),
        "shares": _r(plan.get("shares"), 4), "tp_ret": _r(a.get("tp_ret")), "sl_ret": _r(a.get("sl_ret")),
        "cost_rt": _r(cost), "adv20_eur": _r(a.get("adv20_eur"), 2),
    }


def _signal_sort_key(s: dict):
    order = {"take": 0, "skip": 1, "blocked": 2, "none": 3}
    ev = s.get("ev_net")
    return (order.get(s["decision"], 4), -(ev if ev is not None else -1e9), s["ticker"])


def empty_stock_payload(cfg, stock: Stock, reason_el: str | None = None, context: dict | None = None) -> dict:
    """Το `/api/stock/{ticker}` για μετοχή χωρίς δεδομένα (χωρίς το `favorite`)."""
    agents = explain.agent_report(stock.ticker, None, None, None, None, stock, cfg, context=context)
    return {"ticker": stock.ticker, "name": stock.name, "sector": stock.sector, "custom": bool(stock.custom),
            "has_data": False, "as_of": None, "candles": [], "orb_history": [], "orb": None, "plan": None,
            "decision": "none",
            "decision_reason_el": reason_el or "Δεν υπάρχουν ακόμη δεδομένα τιμών για αυτή τη μετοχή.",
            "agents": agents, "outcomes": [], "notes_el": []}


# ---------------------------------------------------------------- δημοσίευση (σήματα + αρχεία κατάστασης)

def _market_now(panel: pd.DataFrame, as_of) -> dict | None:
    """Το κλίμα της αγοράς την τελευταία συνεδρίαση (ίδιο για όλες τις μετοχές): καθεστώς + βασικά μεγέθη."""
    if as_of is None or not len(panel) or "regime" not in panel.columns:
        return None
    try:
        last = panel.xs(as_of, level="date")
    except KeyError:
        return None
    if not len(last):
        return None
    reg = last["regime"].dropna()
    regime = str(reg.iloc[0]) if len(reg) else "neutral"
    out = {"regime": regime, "regime_el": market.REGIME_LABELS_EL.get(regime, regime)}
    for k in ("mkt_ret_20", "mkt_trend", "mkt_drawdown_10", "mkt_vol_20"):
        if k in last.columns:
            v = last[k].dropna()
            out[k] = _r(v.iloc[0]) if len(v) else None
    return out


def publish(cfg, stocks: list[Stock], prices: dict[str, pd.DataFrame], panel: pd.DataFrame, labels: pd.DataFrame,
            preds: pd.DataFrame, model, *, mode: str, extra_meta: dict | None = None, progress=None,
            record_journal: bool = True) -> dict:
    """Βήματα 4–6: σήματα της τελευταίας συνεδρίασης, ημερολόγιο, αρχεία κατάστασης. Επιστρέφει σύνοψη."""
    report = _sub(progress, 0.0, 1.0)
    report(0.0, "Υπολογισμός σημερινών σημάτων…")
    dates = panel.index.get_level_values("date") if len(panel) else pd.DatetimeIndex([])
    as_of = dates.max() if len(dates) else None
    g_panel, g_labels, g_preds = _group(panel), _group(labels), _group(preds)
    sectors = {s.ticker: s.sector for s in stocks}
    peers = pd.Series([s.sector for s in stocks if s.ticker in prices]).value_counts().to_dict()
    meta_in = extra_meta or {}
    context = {"mode": mode, "source": meta_in.get("source"), "index_ok": meta_in.get("index_ok"),
               "index_note_el": meta_in.get("index_note_el"), "as_of": explain.date_iso(as_of)}

    # ήδη ανοιχτές εικονικές θέσεις (από προηγούμενες ημέρες)
    open_pos = []
    if as_of is not None:
        for e in journal.open_positions(cfg, mode=mode, before=as_of):
            open_pos.append({"ticker": e["ticker"], "sector": e.get("sector") or sectors.get(e["ticker"], "Άλλο"),
                             "size_eur": e.get("size_eur")})

    assessments: dict[str, dict] = {}
    live: dict[str, dict | None] = {}
    last_rows = {s.ticker: g_panel[s.ticker].iloc[-1] for s in stocks
                 if s.ticker in g_panel and len(g_panel[s.ticker])}
    batch = explain.live_predictions(model, last_rows) if model is not None else {}
    for s in stocks:
        frame = g_panel.get(s.ticker)
        if frame is None or not len(frame):
            continue
        row = last_rows[s.ticker]
        pred = batch.get(s.ticker)
        if pred is None and model is None:
            pf = g_preds.get(s.ticker)       # χωρίς τελικό μοντέλο: η τελευταία πρόβλεψη walk‑forward (αν υπάρχει)
            if pf is not None and len(pf) and pf.index[-1] == frame.index[-1]:
                pred = pf.iloc[-1].to_dict()
        live[s.ticker] = pred
        assessments[s.ticker] = explain.prepare_assessment(
            s.ticker, row, pred, cfg, as_of=as_of, model_ready=model is not None or pred is not None,
            row_date=frame.index[-1])
    explain.allocate(list(assessments.values()), sectors, open_pos, cfg)

    report(0.35, "Σύνταξη αναφορών των πρακτόρων…")
    items, signals, counts = [], [], {"take": 0, "skip": 0, "blocked": 0, "none": 0}
    sdir = stocks_dir(cfg)
    sdir.mkdir(parents=True, exist_ok=True)
    written = set()
    for s in stocks:
        df = prices.get(s.ticker)
        a = assessments.get(s.ticker)
        items.append(_item(s, df, a))
        ctx = dict(context, sector_peers=int(peers.get(s.sector, 0)))
        if a is None:
            reason = None
            if df is not None and len(df):
                reason = "Υπάρχουν τιμές, αλλά η μετοχή δεν συμπεριλήφθηκε στον τελευταίο υπολογισμό — τρέξτε «Ενημέρωση»."
            payload = empty_stock_payload(cfg, s, reason, ctx)
            if df is not None and len(df):
                payload["candles"] = _candles(df)
            _write_json(_stock_file(cfg, s.ticker), payload, RAW_KEYS)
            written.add(_stock_file(cfg, s.ticker).name)
            continue
        counts[a["decision"]] = counts.get(a["decision"], 0) + 1
        frame, lab, pf = g_panel.get(s.ticker), g_labels.get(s.ticker), g_preds.get(s.ticker)
        agents = explain.agent_report(s.ticker, frame, lab, live.get(s.ticker), model, s, cfg, assessment=a,
                                      context=ctx)
        notes = []
        if pf is None or not len(pf):
            notes.append("Το ιστορικό της Σφαίρας (walk-forward) θα υπολογιστεί στην επόμενη πλήρη ενημέρωση.")
        payload = {
            "ticker": s.ticker, "name": s.name, "sector": s.sector, "custom": bool(s.custom), "has_data": True,
            "as_of": a.get("row_date"), "candles": _candles(df),
            "orb_history": _orb_history(pf, live.get(s.ticker), frame.index[-1] if a.get("current") else None),
            "orb": _public_orb(a), "plan": _public_plan(a.get("plan")), "decision": a["decision"],
            "decision_reason_el": a.get("reason_el") or "", "agents": agents,
            "outcomes": _outcomes(lab, pf), "notes_el": notes, "mode": mode,
        }
        _write_json(_stock_file(cfg, s.ticker), payload, RAW_KEYS)
        written.add(_stock_file(cfg, s.ticker).name)
        signals.append(_signal(s, a))
    # αρχεία μετοχών που αφαιρέθηκαν από τη λίστα (όχι όσες προστέθηκαν στο μεταξύ από τον χρήστη)
    current = {_stock_file(cfg, s.ticker).name for s in load_universe(cfg.data_dir)}
    for p in sdir.glob("*.json"):
        if p.name not in written and p.name not in current:
            p.unlink(missing_ok=True)
    signals.sort(key=_signal_sort_key)

    report(0.75, "Ενημέρωση ημερολογίου…")
    n_rec = n_res = 0
    if record_journal and as_of is not None:
        rows = []
        for s in stocks:
            a = assessments.get(s.ticker)
            if a is None or a.get("orb") is None or a.get("tp_ret") is None or not a.get("current"):
                continue
            take = a["decision"] == "take"
            cost = a["costs"]["total"]
            if take and a.get("size_eur"):
                cost = quant.round_trip_cost_pct(a["size_eur"], a.get("adv20_eur"), a.get("entry_ref"), cfg,
                                                 vol=a.get("vol_20"))["total"]
            rows.append({"ticker": s.ticker, "sector": s.sector, "decision": "take" if take else "skip",
                         "blocked": a["decision"] == "blocked",
                         "reason_el": a.get("reason_el"), "p_tp": a["orb"]["p_tp"], "p_sl": a["orb"]["p_sl"],
                         "p_to": a["orb"]["p_to"], "ev_net": a["orb"].get("ev_net"), "entry_ref": a.get("entry_ref"),
                         "tp_ret": a["tp_ret"], "sl_ret": a["sl_ret"], "size_eur": a.get("size_eur") if take else None,
                         "horizon_days": cfg.barriers.horizon_days, "cost_rt": cost})
        n_rec = journal.record(cfg, as_of, rows, mode=mode)
        n_res = journal.resolve(cfg, prices, mode=mode)

    report(0.9, "Αποθήκευση αποτελεσμάτων…")
    as_of_s = explain.date_iso(as_of)
    market_now = _market_now(panel, as_of)
    write_state(cfg, "universe", {"as_of": as_of_s, "items": items})
    write_state(cfg, "signals", {"as_of": as_of_s, "items": signals, "counts": counts, "mode": mode})
    p_dates = preds.index.get_level_values("date") if preds is not None and len(preds) else pd.DatetimeIndex([])
    meta = dict(meta_in)
    meta.update({
        "generated_at": _now_iso(),
        "mode": mode,
        "as_of": as_of_s,
        "model": model.info() if model is not None else None,
        "preds": {"n": int(len(preds)) if preds is not None else 0,
                  "first_date": explain.date_iso(p_dates.min()) if len(p_dates) else None,
                  "last_date": explain.date_iso(p_dates.max()) if len(p_dates) else None},
        "n_tickers": len(stocks),
        "n_with_data": sum(1 for it in items if it["has_data"]),
        "counts": counts,
        "market": market_now,
        "journal": {"recorded": n_rec, "resolved": n_res},
        "settings_hash": settings_hash(cfg),
    })
    meta.setdefault("fingerprint", _clean(fingerprint(cfg)))
    write_state(cfg, "meta", meta)
    report(1.0, "Τα σήματα ενημερώθηκαν")
    return {"as_of": as_of_s, "counts": counts, "n_with_data": meta["n_with_data"], "n_tickers": len(stocks),
            "journal": meta["journal"], "model": meta["model"]}


# ---------------------------------------------------------------- πλήρης ενημέρωση

def _load_prices_for(cfg, stocks):
    prices, index_df = scout.load_prices(cfg, [s.ticker for s in stocks])
    return prices, index_df


INDEX_MAX_LAG_SESSIONS = 2      # ο δείκτης μπορεί να υστερεί το πολύ 2 συνεδριάσεις από τις μετοχές


def usable_index(index_df: pd.DataFrame | None, prices: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame | None, str | None]:
    """Ο επίσημος δείκτης αν είναι ενημερωμένος, αλλιώς (None, ελληνική εξήγηση).

    Ένα παλιό GD.AT.csv (π.χ. αποτυχημένη λήψη του δείκτη ενώ οι μετοχές ήρθαν) θα «πάγωνε» τα
    χαρακτηριστικά αγοράς, το καθεστώς και τον έλεγχο απότομης πτώσης, και θα έκοβε την αγορά & διακράτηση.
    """
    if index_df is None or not len(index_df):
        return None, "Ο Γενικός Δείκτης δεν ήταν διαθέσιμος: η «αγορά» είναι ισοσταθμικός δείκτης των μετοχών της λίστας."
    ends = [df.index[-1] for df in prices.values() if df is not None and len(df)]
    if not ends:
        return index_df, None
    ref = max(ends)
    last = index_df.index[-1]
    lag = len(pd.bdate_range(last, ref)) - 1 if last < ref else 0
    if lag > INDEX_MAX_LAG_SESSIONS:
        return None, (f"Ο Γενικός Δείκτης έχει δεδομένα μόνο έως {explain.date_el(last)} (οι μετοχές έως "
                      f"{explain.date_el(ref)}): η «αγορά» είναι ισοσταθμικός δείκτης των μετοχών της λίστας.")
    return index_df, None


def run_refresh(cfg, progress=None, demo: bool = False, stocks: list[Stock] | None = None) -> dict:
    """Πλήρης ενημέρωση: 1 δεδομένα (ή επίδειξη) 2 panel + ετικέτες 3 walk‑forward 4 σήματα
    5 ημερολόγιο 6 αρχεία κατάστασης. Επιστρέφει σύνοψη· σε αδυναμία πετά `PipelineError` (ελληνικά)."""
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    report = _sub(progress, 0.0, 1.0)
    stocks = list(stocks) if stocks is not None else load_universe(cfg.data_dir)
    if not stocks:
        raise PipelineError("Η λίστα μετοχών είναι άδεια.")

    # 1. δεδομένα
    fetch = {"ok": [], "failed": []}
    if demo:
        report(0.0, "Δημιουργία συνθετικών δεδομένων επίδειξης…")
        scout.write_demo_data(cfg, stocks, seed=DEMO_SEED)
        journal.clear(cfg, "demo")          # νέα επίδειξη ⇒ καθαρό ημερολόγιο επίδειξης (τα πραγματικά μένουν)
        fetch["ok"] = [s.ticker for s in stocks]
    else:
        report(0.0, "Λήψη δεδομένων τιμών…")
        fetch = scout.update_cache(cfg, stocks, progress=_sub(progress, 0.0, 0.25))
        if not fetch.get("ok"):
            src = "τα αρχεία CSV (data/csv)" if cfg.data.source == "csv" else "το Yahoo Finance"
            hint = "ελέγξτε τη σύνδεση στο διαδίκτυο" if cfg.data.source != "csv" else \
                "ελέγξτε ότι υπάρχουν αρχεία data/csv/<ΣΥΜΒΟΛΟ>.csv με κεφαλίδα date,open,high,low,close,volume"
            probs = list((fetch.get("problems") or {}).values())
            detail = f" Πρόβλημα: {probs[0]}" if probs else ""
            now = scout.data_mode(cfg)
            if now == "empty":
                raise PipelineError(f"Δεν ήταν δυνατή η λήψη δεδομένων τιμών από {src} — {hint} ή δοκιμάστε "
                                    f"τα δεδομένα επίδειξης.{detail}")
            keep = "τα δεδομένα επίδειξης" if now == "demo" else \
                f"τα αποθηκευμένα δεδομένα (έως {explain.date_el(scout.data_meta(cfg).get('last_date'))})"
            raise PipelineError(f"Δεν ήταν δυνατή η λήψη νέων δεδομένων από {src} — παραμένουν {keep}.{detail}")
    timings["data"] = time.perf_counter() - t0
    mode = scout.data_mode(cfg)
    dmeta = scout.data_meta(cfg)

    # 2. panel + ετικέτες
    t1 = time.perf_counter()
    report(0.26, "Φόρτωση τιμών…")
    prices, index_df = _load_prices_for(cfg, stocks)
    if not prices:
        raise PipelineError("Δεν υπάρχουν δεδομένα τιμών για καμία μετοχή της λίστας.")
    index_df, index_note = usable_index(index_df, prices)
    index_ok = index_df is not None
    if not index_ok:
        index_df = market.equal_weight_index(prices)
    report(0.27, "Υπολογισμός χαρακτηριστικών…")
    panel = features.build_panel(prices, index_df, stocks, cfg)
    report(0.29, "Ετικέτες τριπλού φραγμού (στόχος / stop / λήξη)…")
    lab = labels_mod.triple_barrier(prices, panel, cfg)
    timings["panel_labels"] = time.perf_counter() - t1

    # 3. walk‑forward
    t2 = time.perf_counter()
    report(0.30, "Εκπαίδευση μοντέλου (walk-forward)…")
    preds, model = orb_model.walk_forward(panel, lab, cfg, progress=_sub(progress, 0.30, 0.85))
    timings["walk_forward"] = time.perf_counter() - t2

    with STATE_LOCK:
        for name, obj in (("panel", panel), ("labels", lab), ("preds", preds), ("model", model), ("index", index_df),
                          ("prices", prices)):
            save_pickle(cfg, name, obj)
        t3 = time.perf_counter()
        extra = {
            "data_last_date": dmeta.get("last_date"), "data_updated_at": dmeta.get("updated_at"),
            "source": dmeta.get("source"), "index_ok": index_ok, "index_note_el": index_note,
            "failed": fetch.get("failed", []), "data_problems": fetch.get("problems") or {},
            "fingerprint": _clean(fingerprint(cfg)), "preds_id": uuid.uuid4().hex[:12],
        }
        summary = publish(cfg, stocks, prices, panel, lab, preds, model, mode=mode, extra_meta=extra,
                          progress=_sub(progress, 0.85, 1.0))
        timings["signals"] = time.perf_counter() - t3
        timings["total"] = time.perf_counter() - t0
        meta = load_state(cfg, "meta") or {}
        meta["timings"] = {k: round(v, 2) for k, v in timings.items()}
        write_state(cfg, "meta", meta)
    summary.update({"mode": mode, "failed": fetch.get("failed", []), "index_ok": index_ok,
                    "timings": meta["timings"]})
    report(1.0, "Η ενημέρωση ολοκληρώθηκε")
    return summary


# ---------------------------------------------------------------- γρήγορος επανυπολογισμός

def load_context(cfg, stocks: list[Stock] | None = None) -> dict | None:
    """Οι αποθηκευμένες caches της τελευταίας ενημέρωσης (+ τιμές από τα CSV). None αν λείπουν.

    Αν από τότε άλλαξε το σύνολο των μετοχών με δεδομένα (π.χ. νέα μετοχή), το panel και οι ετικέτες
    ξαναϋπολογίζονται (φθηνό)· οι προβλέψεις walk‑forward μένουν ως έχουν μέχρι την επόμενη ενημέρωση.
    """
    meta = load_state(cfg, "meta")
    panel, preds = load_pickle(cfg, "panel"), load_pickle(cfg, "preds")
    if meta is None or panel is None or preds is None:
        return None
    lab, model, index_df = load_pickle(cfg, "labels"), load_pickle(cfg, "model"), load_pickle(cfg, "index")
    stocks = list(stocks) if stocks is not None else load_universe(cfg.data_dir)
    mcfg = cfg_for_model(cfg, meta.get("fingerprint"))
    prices = _load_prices_cached(cfg, stocks)
    if not prices:
        return None
    if index_df is None or not len(index_df):
        index_df = market.equal_weight_index(prices)
    have = set(panel.index.get_level_values("ticker").unique()) if len(panel) else set()
    if lab is None or not len(panel):
        panel = features.build_panel(prices, index_df, stocks, mcfg)
        lab = labels_mod.triple_barrier(prices, panel, mcfg)
    elif set(prices) != have:
        panel, lab = _splice_panel(panel, lab, prices, index_df, stocks, have, mcfg)
    if len(preds):
        preds = preds[preds.index.get_level_values("ticker").isin(list(prices))]
    return {"meta": meta, "cfg": mcfg, "stocks": stocks, "prices": prices, "panel": panel, "labels": lab,
            "preds": preds, "model": model, "index": index_df}


def _load_prices_cached(cfg, stocks: list[Stock]) -> dict[str, pd.DataFrame]:
    """Τιμές από το prices.pkl της τελευταίας ενημέρωσης· από τα CSV μόνο όσες άλλαξαν μετά από αυτήν."""
    cached = load_pickle(cfg, "prices") or {}
    pkl = state_dir(cfg) / "prices.pkl"
    pkl_t = pkl.stat().st_mtime_ns if pkl.exists() else -1
    out, missing = {}, []
    for s in stocks:
        csv = scout._csv_path(cfg, s.ticker)
        if not csv.exists():
            continue
        if s.ticker in cached and csv.stat().st_mtime_ns <= pkl_t:
            out[s.ticker] = cached[s.ticker]
        else:
            missing.append(s.ticker)
    if missing:
        fresh, _ = scout.load_prices(cfg, missing)
        out.update(fresh)
    result = {s.ticker: out[s.ticker] for s in stocks if s.ticker in out}
    if missing and result:
        with STATE_LOCK:                 # την επόμενη φορά χωρίς ανάγνωση CSV
            save_pickle(cfg, "prices", result)
    return result


def _splice_panel(panel, lab, prices, index_df, stocks, have: set, mcfg):
    """Ξαναϋπολογίζει panel/ετικέτες μόνο για τους κλάδους που άλλαξαν (νέα ή αφαιρεμένη μετοχή).

    Το `sector_rel_20` εξαρτάται μόνο από τις μετοχές του ίδιου κλάδου και τα χαρακτηριστικά αγοράς από
    τον (αποθηκευμένο) δείκτη, άρα οι υπόλοιποι κλάδοι μένουν ακριβώς ίδιοι.
    """
    sector_of = {s.ticker: s.sector for s in stocks}
    changed = (set(prices) ^ have)
    sectors = {sector_of.get(t, market.OTHER_SECTOR) for t in changed}
    affected = [t for t in prices if sector_of.get(t, market.OTHER_SECTOR) in sectors]
    keep_t = [t for t in prices if t not in affected and t in have]
    tick = panel.index.get_level_values("ticker")
    keep = np.asarray(tick.isin(keep_t))
    parts_p, parts_l = [panel[keep]], [lab[keep]]
    if affected:
        sub = {t: prices[t] for t in affected}
        sp = features.build_panel(sub, index_df, stocks, mcfg)
        parts_p.append(sp)
        parts_l.append(labels_mod.triple_barrier(sub, sp, mcfg))
    new_panel = pd.concat(parts_p).sort_index()
    new_lab = pd.concat(parts_l).reindex(new_panel.index)
    return new_panel, new_lab


def recompute_signals(cfg, record_journal: bool = False) -> dict | None:
    """Ξαναϋπολογίζει σήματα και αρχεία κατάστασης με το αποθηκευμένο μοντέλο (χωρίς εκπαίδευση).

    Χρήσιμο μετά από αλλαγή ρυθμίσεων φύλακα/κόστους/χαρτοφυλακίου ή προσθήκη/αφαίρεση μετοχής.
    None αν δεν έχει γίνει ποτέ ενημέρωση. Δεν γράφει στο ημερολόγιο (εκτός αν ζητηθεί ρητά): οι
    αποφάσεις του ημερολογίου καταγράφονται μόνο από την πλήρη ενημέρωση με νέα δεδομένα, ώστε μια αλλαγή
    ρυθμίσεων αφού φανούν οι τιμές της επόμενης συνεδρίασης να μην ξαναγράφει τις αποφάσεις.
    """
    with STATE_LOCK:
        ctx = load_context(cfg)
        if ctx is None:
            return None
        meta = ctx["meta"]
        extra = {k: meta.get(k) for k in ("data_last_date", "data_updated_at", "source", "index_ok", "index_note_el",
                                           "failed", "data_problems", "fingerprint", "preds_id", "timings")}
        dm = scout.data_meta(cfg)
        extra["data_last_date"] = dm.get("last_date") or extra["data_last_date"]
        extra["data_updated_at"] = dm.get("updated_at") or extra["data_updated_at"]
        out = publish(ctx["cfg"], ctx["stocks"], ctx["prices"], ctx["panel"], ctx["labels"], ctx["preds"],
                      ctx["model"], mode=scout.data_mode(cfg), extra_meta=extra, record_journal=record_journal)
        # το hash ρυθμίσεων αφορά τις ρυθμίσεις του χρήστη (όχι το αντίγραφο με τα όρια της εκπαίδευσης)
        m = load_state(cfg, "meta") or {}
        m["settings_hash"] = settings_hash(cfg)
        write_state(cfg, "meta", m)
        return out


def add_stock(cfg, stock: Stock) -> dict:
    """Προσθήκη μετοχής χρήστη: δεδομένα + κατάσταση (γρήγορη διαδρομή, χωρίς νέα εκπαίδευση).

    Επιστρέφει {"ok", "item"?, "error_el"?, "note_el"?}. Σε επίδειξη τα δεδομένα είναι συνθετικά· αλλιώς
    λήψη από την πηγή — αν αποτύχει, η μετοχή ΔΕΝ προστίθεται. Χωρίς καθόλου δεδομένα (καμία ενημέρωση
    ακόμη) προστίθεται στη λίστα και τα δεδομένα της έρχονται με την πρώτη ενημέρωση.
    """
    stock = Stock(**{**stock.to_dict(), "custom": True})
    if cfg.data.index_ticker and stock.ticker == cfg.data.index_ticker:
        return {"ok": False, "error_el": f"Το {stock.ticker} είναι ο Γενικός Δείκτης — δεν προστίθεται ως μετοχή."}
    mode = scout.data_mode(cfg)
    if mode == "empty":
        add_custom(cfg.data_dir, stock)
        return {"ok": True, "item": _item(stock, None, None),
                "note_el": "Η μετοχή προστέθηκε· τα δεδομένα της θα ληφθούν στην πρώτη ενημέρωση."}
    try:
        got = scout.add_stock_data(cfg, stock)
    except Exception:  # δίκτυο/πηγή: ποτέ δεν «ρίχνουμε» τον διακομιστή
        got = False
    if not got:
        src = "τα αρχεία CSV (data/csv)" if cfg.data.source == "csv" else "το Yahoo Finance"
        return {"ok": False, "error_el": f"Δεν βρέθηκαν δεδομένα για το {stock.ticker} από {src}. "
                                         "Ελέγξτε το σύμβολο (π.χ. INTEK.AT)."}
    add_custom(cfg.data_dir, stock)
    item = None
    if load_state(cfg, "meta") is not None:
        recompute_signals(cfg)
        uni = load_state(cfg, "universe") or {}
        item = next((it for it in uni.get("items", []) if it["ticker"] == stock.ticker), None)
    if item is None:
        prices, _ = scout.load_prices(cfg, [stock.ticker])
        item = _item(stock, prices.get(stock.ticker), None)
    return {"ok": True, "item": item}


def remove_stock(cfg, ticker: str) -> None:
    """Αφαιρεί τα αρχεία κατάστασης/τιμών μιας μετοχής χρήστη και ενημερώνει λίστα και σήματα."""
    with STATE_LOCK:
        _stock_file(cfg, ticker).unlink(missing_ok=True)
        if ticker != cfg.data.index_ticker:              # ο δείκτης δεν σβήνεται ποτέ από εδώ
            scout._csv_path(cfg, ticker).unlink(missing_ok=True)
        for name in ("universe", "signals"):
            st = load_state(cfg, name)
            if st and isinstance(st.get("items"), list):
                st["items"] = [it for it in st["items"] if it.get("ticker") != ticker]
                write_state(cfg, name, st)


# ---------------------------------------------------------------- backtest

def run_backtest_job(cfg, progress=None) -> dict:
    """Backtest της στρατηγικής + αγορά & διακράτηση + τυχαίες στρατηγικές → state/backtest.json."""
    t0 = time.perf_counter()
    report = _sub(progress, 0.0, 1.0)
    report(0.0, "Φόρτωση προβλέψεων…")
    ctx = load_context(cfg)
    if ctx is None or ctx["preds"] is None or not len(ctx["preds"]):
        raise PipelineError("Δεν υπάρχουν προβλέψεις του μοντέλου — τρέξτε πρώτα «Ενημέρωση».")
    bcfg = ctx["cfg"]
    prices, panel, preds, lab = ctx["prices"], ctx["panel"], ctx["preds"], ctx["labels"]
    index_df = ctx["index"] if ctx["index"] is not None else market.equal_weight_index(prices)
    capital = float(bcfg.portfolio.capital_eur)

    report(0.03, "Προσομοίωση χαρτοφυλακίου…")
    bctx = executor.prepare(prices, panel, preds, lab, bcfg)
    if bctx is None:
        raise PipelineError("Δεν υπάρχουν αρκετά δεδομένα για backtest.")
    res = executor.run_backtest(prices, panel, preds, lab, bcfg, context=bctx)
    report(0.1, "Αγορά & διακράτηση δείκτη…")
    bh = benchmarks.buy_and_hold(index_df, bctx.cal[0], bctx.cal[-1], capital)
    bhm = benchmarks.metrics(bh, None, capital)
    # ισοσταθμικό χαρτοφυλάκιο των ίδιων μετοχών: ίδια δεδομένα (προσαρμοσμένα για μερίσματα) και ίδια
    # μεροληψία επιβίωσης με τη στρατηγική — ο δείκτης τιμών μόνος του θα ήταν πιο «εύκολος» αντίπαλος
    bhu = benchmarks.buy_and_hold(market.equal_weight_index(prices), bctx.cal[0], bctx.cal[-1], capital)
    bhum = benchmarks.metrics(bhu, None, capital)
    runs = int(bcfg.backtest.random_runs)
    rb = benchmarks.random_benchmark(prices, panel, preds, lab, bcfg, runs, progress=_sub(progress, 0.12, 0.9),
                                     context=bctx)
    report(0.91, "Έλεγχος δεξιότητας του μοντέλου εκτός δείγματος…")
    sk = benchmarks.skill(preds, lab, seed=int(bcfg.backtest.seed))
    report(0.93, "Βαθμονόμηση πιθανοτήτων…")
    calib = benchmarks.calibration(preds, lab, bins=10)
    rm = benchmarks.metrics(rb["equity_median"], None, capital)
    # η διάμεσος καμπύλη (σημείο προς σημείο) δεν την παρήγαγε καμία εκτέλεση: οι μετρικές είναι οι
    # διάμεσοι των μετρικών κάθε τυχαίας εκτέλεσης· όσα δεν ορίζονται μένουν κενά
    rm.update({k: v for k, v in (rb.get("metrics_median") or {}).items() if k in rm})
    rm["exposure"] = None
    rm["n_trades"] = int(np.median(rb["n_trades"])) if rb["n_trades"] else 0
    rm["total_costs_eur"] = float(np.median(rb["costs_eur"])) if rb.get("costs_eur") else None
    wr = np.asarray(rb.get("win_rates") or [], dtype="float64")
    rm["win_rate"] = float(np.nanmedian(wr)) if np.isfinite(wr).any() else None
    for k in ("avg_win", "avg_loss", "profit_factor", "avg_bars_held"):
        rm[k] = None
    bhm["n_trades"] = None                             # ο δείκτης δεν «συναλλάσσεται»
    bhum["n_trades"] = None
    report(0.96, "Σύνταξη ετυμηγορίας…")
    verdict = benchmarks.verdict(res["metrics"], rb["final_returns"], bhm, bh_universe_metrics=bhum,
                                 skill_result=sk)

    eq = res["equity"]
    cal = bctx.cal
    bh_al = bh.reindex(cal).ffill() if len(bh) else pd.Series(np.nan, index=cal)
    bhu_al = bhu.reindex(cal).ffill() if len(bhu) else pd.Series(np.nan, index=cal)

    def al(s):
        return s.reindex(cal) if s is not None and len(s) else pd.Series(np.nan, index=cal)

    med, p05, p95 = al(rb["equity_median"]), al(rb["equity_p05"]), al(rb["equity_p95"])
    dstr = cal.strftime("%Y-%m-%d").tolist()
    ev, bv, uv, mv, lv, hv = (x.to_numpy("float64") for x in (eq.reindex(cal), bh_al, bhu_al, med, p05, p95))
    equity = [{"d": dstr[i], "strategy": _r(ev[i], 2), "buy_hold": _r(bv[i], 2), "buy_hold_universe": _r(uv[i], 2),
               "random_median": _r(mv[i], 2), "random_p05": _r(lv[i], 2), "random_p95": _r(hv[i], 2)}
              for i in range(len(cal))]
    peak = np.maximum.accumulate(np.concatenate(([capital], ev)))[1:]
    dd = ev / peak - 1.0
    drawdown = [{"d": dstr[i], "strategy": _r(dd[i])} for i in range(len(cal))]

    trades = res["trades"]
    names = {s.ticker: s.name for s in ctx["stocks"]}
    tr_items = []
    if len(trades):
        t = trades.tail(N_BACKTEST_TRADES)
        for r in t.to_dict("records"):
            tr_items.append({
                "ticker": r["ticker"], "name": names.get(r["ticker"], r["ticker"]), "sector": r["sector"],
                "signal_date": explain.date_iso(r["signal_date"]), "entry_date": explain.date_iso(r["entry_date"]),
                "entry_price": _r(r["entry_price"], 4), "exit_date": explain.date_iso(r["exit_date"]),
                "exit_price": _r(r["exit_price"], 4), "label": int(r["label"]), "size_eur": _r(r["size_eur"], 2),
                "gross_ret": _r(r["gross_ret"]), "cost_eur": _r(r["cost_eur"], 2),
                "net_pnl_eur": _r(r["net_pnl_eur"], 2), "net_ret": _r(r["net_ret"]), "p_tp": _r(r["p_tp"], 4),
                "p_sl": _r(r["p_sl"], 4), "p_to": _r(r["p_to"], 4), "ev_net": _r(r["ev_net"]),
                "bars_held": int(r["bars_held"]),
            })
    lab_counts = trades["label"].value_counts().to_dict() if len(trades) else {}
    by_label = {"tp": int(lab_counts.get(0, 0)), "sl": int(lab_counts.get(1, 0)), "to": int(lab_counts.get(2, 0)),
                "open": int(lab_counts.get(executor.LABEL_MARKED, 0))}
    meta = ctx["meta"]
    p = bcfg.portfolio
    payload = {
        "available": True,
        "generated_at": _now_iso(),
        "mode": scout.data_mode(cfg),
        "period": {"start": dstr[0] if dstr else None, "end": dstr[-1] if dstr else None},
        "params": {"horizon_days": bcfg.barriers.horizon_days, "k_tp": bcfg.barriers.k_tp,
                   "k_sl": bcfg.barriers.k_sl, "capital_eur": p.capital_eur, "position_eur_max": p.position_eur_max,
                   "max_positions": p.max_positions, "max_per_sector": p.max_per_sector,
                   "max_adv_fraction": p.max_adv_fraction, "min_ev_net": bcfg.guard.min_ev_net,
                   "min_adv_eur": bcfg.guard.min_adv_eur, "commission_pct": bcfg.costs.commission_pct,
                   "commission_min_eur": bcfg.costs.commission_min_eur, "random_runs": rb["runs"],
                   "model_kind": bcfg.model.kind, "seed": bcfg.backtest.seed},
        "metrics": {"strategy": res["metrics"], "buy_hold": bhm, "buy_hold_universe": bhum, "random_median": rm},
        "skill": sk,
        "equity": equity,
        "drawdown": drawdown,
        "trades": tr_items,
        "n_trades_total": int(len(trades)),
        "calibration": calib,
        "by_label": by_label,
        "verdict": verdict,
        "random": {"runs": rb["runs"], "method_el": rb.get("method_el"),
                   "final_returns": [_r(x) for x in rb["final_returns"]],
                   "n_trades_median": rm["n_trades"]},
        "notes_el": _backtest_notes(bcfg, meta, scout.data_mode(cfg), sk),
        "settings_hash": settings_hash(cfg),
        "preds_id": meta.get("preds_id"),
    }
    payload["timing_s"] = round(time.perf_counter() - t0, 2)
    with STATE_LOCK:
        write_state(cfg, "backtest", payload)
    report(1.0, "Το backtest ολοκληρώθηκε")
    return {"verdict": verdict, "metrics": {"strategy": res["metrics"], "buy_hold": bhm, "buy_hold_universe": bhum,
                                            "random_median": rm},
            "skill": sk, "n_trades": int(len(trades)), "timing_s": payload["timing_s"], "period": payload["period"]}


def _backtest_notes(cfg, meta: dict, mode: str, skill: dict | None = None) -> list[str]:
    notes = [
        "Είσοδος στο άνοιγμα της επόμενης συνεδρίασης, έξοδος στο πρώτο από στόχο / stop / λήξη χρόνου — "
        "ακριβώς όπως στις ετικέτες του μοντέλου.",
        "Κάθε συναλλαγή πληρώνει προμήθεια, χρεώσεις, φόρο, spread και επίδραση στην τιμή.",
        "Οι προβλέψεις είναι εκτός δείγματος (walk-forward): κάθε μοντέλο είδε μόνο αποτελέσματα που είχαν "
        "κριθεί πριν από την ημερομηνία πρόβλεψης.",
        f"Οι τυχαίες στρατηγικές τηρούν τους ίδιους κανόνες με τη στρατηγική ({fmt_num(cfg.backtest.random_runs)} "
        "εκτελέσεις)· οι μετρικές τους είναι οι διάμεσοι των μετρικών κάθε εκτέλεσης.",
        "Αγορά & διακράτηση: ο δείκτης και ένα ισοσταθμικό χαρτοφυλάκιο των ίδιων μετοχών, χωρίς κόστη. Ο δείκτης "
        "του Yahoo είναι δείκτης τιμών (χωρίς μερίσματα), ενώ οι τιμές των μετοχών είναι προσαρμοσμένες για "
        "μερίσματα — γι' αυτό η στρατηγική πρέπει να νικά και το ισοσταθμικό χαρτοφυλάκιο.",
        "Μεροληψία επιβίωσης: η λίστα περιέχει εταιρείες που είναι εισηγμένες σήμερα. Όσες διαγράφηκαν ή "
        "κατέρρευσαν στο μεταξύ λείπουν, οπότε οι αποδόσεις του παρελθόντος φαίνονται καλύτερες απ' ό,τι ήταν.",
    ]
    if skill and math.isfinite(skill.get("logloss_model") or float("nan")):
        verdict_word = "καλύτερες" if skill.get("ok") else "όχι καλύτερες"
        notes.append(f"Δεξιότητα εκτός δείγματος: οι πιθανότητες του μοντέλου είναι {verdict_word} από τις σταθερές "
                     f"ιστορικές συχνότητες (log-loss {fmt_num(skill['logloss_model'], 3)} έναντι "
                     f"{fmt_num(skill['logloss_base'], 3)}, {fmt_num(skill.get('n') or 0)} προβλέψεις, "
                     f"p = {fmt_num(skill.get('p_value'), 3)} με επαναδειγματοληψία σε μπλοκ ημερομηνιών).")
    if mode == "demo":
        notes.insert(0, "ΕΠΙΔΕΙΞΗ: συνθετικά δεδομένα — τα αποτελέσματα δεν λένε τίποτα για την πραγματική αγορά.")
    if meta.get("index_ok") is False:
        notes.append(meta.get("index_note_el") or
                     "Ο Γενικός Δείκτης δεν ήταν διαθέσιμος: η σύγκριση γίνεται με ισοσταθμικό δείκτη των μετοχών.")
    return notes


def run_demo(cfg, progress=None, stocks: list[Stock] | None = None) -> dict:
    """Δεδομένα επίδειξης + πλήρης ενημέρωση + backtest."""
    ref = run_refresh(cfg, progress=_sub(progress, 0.0, 0.72), demo=True, stocks=stocks)
    bt = run_backtest_job(cfg, progress=_sub(progress, 0.72, 1.0))
    return {"refresh": ref, "backtest": bt}


def verdict_line(bt: dict) -> str:
    """Σύντομη ελληνική περίληψη αποτελέσματος backtest (για το CLI)."""
    v = bt["verdict"]
    m = bt["metrics"]
    bhu = (m.get("buy_hold_universe") or {}).get("total_return")
    return (f"Στρατηγική {fmt_pct(m['strategy']['total_return'], 1, sign=True)} · αγορά & διακράτηση "
            f"{fmt_pct(m['buy_hold']['total_return'], 1, sign=True)} (ισοσταθμική λίστα "
            f"{fmt_pct(bhu, 1, sign=True)}) · διάμεσος τυχαίων "
            f"{fmt_pct(m['random_median']['total_return'], 1, sign=True)} · percentile "
            f"{fmt_pct(v['percentile'], 0)} · συναλλαγές {fmt_num(bt['n_trades'])} → {v['level'].upper()}")
