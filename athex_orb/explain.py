"""Οι «έξι πράκτορες»: αξιολόγηση μιας μετοχής και η εξήγησή της για το UI.

Δύο επίπεδα:
* `assess(...)` — η ουσία της απόφασης για την τελευταία συνεδρίαση: πιθανότητες της Σφαίρας,
  κόστη & αναμενόμενη αξία (Ποσοτικός), έλεγχοι (Φύλακας), απόφαση, σχέδιο θέσης (Εκτελεστής).
  Το `allocate(...)` εφαρμόζει τους ελέγχους χαρτοφυλακίου σε όλες τις υποψήφιες μιας ημέρας, με
  σειρά ev_net (ίδια λογική με το backtest).
* `agent_report(...)` — οι έξι κάρτες (Ανιχνευτής, Χαρτογράφος, Κριτής, Ποσοτικός, Φύλακας,
  Εκτελεστής) με ελληνικά κείμενα και «ωμούς» αριθμούς + υπόδειξη μορφοποίησης (`fmt`) για το UI.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import guard, quant
from .features import FEATURE_COLUMNS, FEATURE_FMT, FEATURE_LABELS_EL
from .guard import fmt_num, fmt_pct
from .labels import LABEL_SL, LABEL_TO, LABEL_TP
from .market import REGIME_LABELS_EL, REGIME_TREND_BAND

AGENT_IDS = ["scout", "market", "judge", "quant", "guard", "executor"]
AGENT_NAMES_EL = {
    "scout": ("Ανιχνευτής", "Δεδομένα & ρευστότητα"),
    "market": ("Χαρτογράφος", "Κλίμα αγοράς & κλάδου"),
    "judge": ("Κριτής", "Χαρακτηριστικά & μοντέλο"),
    "quant": ("Ποσοτικός", "Αναμενόμενη αξία μετά τα κόστη"),
    "guard": ("Φύλακας", "Έλεγχοι κινδύνου"),
    "executor": ("Εκτελεστής", "Σχέδιο θέσης (εικονικό)"),
}
DOMINANT_KEYS = {0: "tp", 1: "sl", 2: "to"}
DECISIONS = ("take", "skip", "blocked", "none")
HIGH_LIQUIDITY_EUR = quant.SPREAD_REF_ADV_EUR     # πάνω από αυτό το spread είναι το βασικό
N_DRIVERS = 6
BASE_RATE_DAYS = 250

# σύντομες ονομασίες χαρακτηριστικών για τίτλους
SHORT_LABELS_EL = {
    "ret_1": "απόδοση ημέρας", "ret_5": "απόδοση 5ημ.", "ret_20": "απόδοση 20ημ.", "ret_60": "απόδοση 60ημ.",
    "dist_sma20": "απόσταση από ΚΜΟ 20", "dist_sma50": "απόσταση από ΚΜΟ 50",
    "dist_sma200": "απόσταση από ΚΜΟ 200", "rsi_14": "RSI 14", "atr_pct": "εύρος κίνησης (ATR)",
    "vol_20": "μεταβλητότητα 20ημ.", "vol_ratio": "όγκος", "value_z": "τζίρος", "gap_1": "χάσμα ανοίγματος",
    "range_pos_252": "θέση στο εύρος 52 εβδ.", "rel_mkt_20": "απόδοση έναντι αγοράς",
    "sector_rel_20": "απόδοση έναντι κλάδου", "mkt_ret_20": "απόδοση αγοράς", "mkt_vol_20": "μεταβλητότητα αγοράς",
    "mkt_trend": "τάση αγοράς",
}
# ετικέτες μετρικών (μεσαίου μήκους — χωρούν στις κάρτες)
METRIC_LABELS_EL = {k: v[0].upper() + v[1:] for k, v in SHORT_LABELS_EL.items()}
METRIC_LABELS_EL.update({"rsi_14": "RSI 14", "atr_pct": "ATR / τιμή", "vol_ratio": "Όγκος / μέσος 20ημ.",
                         "value_z": "Τζίρος (z-score)"})
SOURCE_EL = {
    "synthetic": "συνθετικά δεδομένα επίδειξης",
    "yahoo": "Yahoo Finance",
    "csv": "αρχεία CSV του χρήστη",
}


# ---------------------------------------------------------------- βοηθητικά

def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _ok(x) -> bool:
    return math.isfinite(_f(x))


def _val(x):
    """Τιμή για JSON: πεπερασμένος αριθμός ή None."""
    v = _f(x)
    return v if math.isfinite(v) else None


def _get(row, key):
    if row is None:
        return float("nan")
    try:
        return row[key]
    except (KeyError, IndexError):
        return float("nan")


def eur_short(x) -> str:
    """Ποσό σε € για τίτλους: «4,2 εκ. €», «350 χιλ. €», «1.250 €», «4,50 €»."""
    v = _f(x)
    if not math.isfinite(v):
        return "—"
    a = abs(v)
    if a >= 1e6:
        return f"{fmt_num(v / 1e6, 1)} εκ. €"
    if a >= 1e4:
        return f"{fmt_num(v / 1e3, 0)} χιλ. €"
    if a >= 100:
        return f"{fmt_num(v, 0)} €"
    return f"{fmt_num(v, 2)} €"


def _lc(text: str) -> str:
    """Πρώτο γράμμα πεζό (για συνένωση φράσεων), εκτός αν η λέξη είναι ακρωνύμιο (π.χ. «ATR», «EV»)."""
    if not text:
        return text
    first = text.split(" ", 1)[0]
    if len(first) > 1 and first[:2].isupper():
        return text
    return text[0].lower() + text[1:]


def date_el(d) -> str:
    if d is None:
        return "—"
    try:
        ts = pd.Timestamp(d)
    except (TypeError, ValueError):
        return "—"
    return "—" if pd.isna(ts) else ts.strftime("%d/%m/%Y")


def date_iso(d) -> str | None:
    if d is None:
        return None
    try:
        ts = pd.Timestamp(d)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(ts) else ts.strftime("%Y-%m-%d")


def _metric(label_el: str, value, fmt: str, hint_el: str | None = None) -> dict:
    if fmt in ("pct", "eur", "num", "int"):
        value = _val(value)
        if fmt == "int" and value is not None:
            value = int(round(value))
    m = {"label_el": label_el, "value": value, "fmt": fmt}
    if hint_el:
        m["hint_el"] = hint_el
    return m


def _agent(aid: str, status: str, headline: str, metrics=None, notes=None, **extra) -> dict:
    name, role = AGENT_NAMES_EL[aid]
    out = {"id": aid, "name_el": name, "role_el": role, "status": status, "headline_el": headline,
           "metrics": list(metrics or []), "notes_el": [n for n in (notes or []) if n]}
    out.update(extra)
    return out


def ticker_frame(frame: pd.DataFrame | None, ticker: str) -> pd.DataFrame | None:
    """Οι γραμμές μιας μετοχής (index = ημερομηνίες) από panel/labels/preds με MultiIndex (date, ticker)."""
    if frame is None or not len(frame):
        return None
    if isinstance(frame.index, pd.MultiIndex):
        if ticker not in frame.index.get_level_values("ticker"):
            return None
        return frame.xs(ticker, level="ticker")
    return frame


def dominant(p_tp, p_sl, p_to) -> str:
    return DOMINANT_KEYS[int(np.argmax([p_tp, p_sl, p_to]))]


def orb_dict(pred) -> dict | None:
    """{"p_tp","p_sl","p_to","dominant"} από γραμμή/λεξικό πρόβλεψης (None αν λείπει)."""
    if pred is None:
        return None
    p = [_f(_get(pred, k)) for k in ("p_tp", "p_sl", "p_to")]
    if not all(math.isfinite(x) for x in p):
        return None
    return {"p_tp": p[0], "p_sl": p[1], "p_to": p[2], "dominant": dominant(*p)}


def live_prediction(model, row) -> dict | None:
    """Πρόβλεψη του (τελικού) μοντέλου για μία γραμμή του panel: p_tp, p_sl, p_to, timeout_mean, model_date."""
    if model is None or row is None:
        return None
    x = np.asarray([_f(_get(row, c)) for c in FEATURE_COLUMNS], dtype="float64")
    if not np.isfinite(x).all():
        return None
    p = model.predict_proba(x[None, :])[0]
    if not np.isfinite(p).all():
        return None
    return {"p_tp": float(p[0]), "p_sl": float(p[1]), "p_to": float(p[2]),
            "timeout_mean": float(model.class_timeout_mean),
            "model_date": date_iso(model.trained_until)}


def live_predictions(model, rows: dict) -> dict[str, dict | None]:
    """Όπως το `live_prediction`, για πολλές μετοχές σε μία κλήση του μοντέλου: {ticker: πρόβλεψη ή None}."""
    if model is None or not rows:
        return {t: None for t in rows}
    tickers = list(rows)
    X = np.vstack([feature_vector(rows[t]) for t in tickers])
    P = model.predict_proba(X)
    md = date_iso(model.trained_until)
    out = {}
    for t, p in zip(tickers, P):
        out[t] = ({"p_tp": float(p[0]), "p_sl": float(p[1]), "p_to": float(p[2]),
                   "timeout_mean": float(model.class_timeout_mean), "model_date": md}
                  if np.isfinite(p).all() else None)
    return out


def feature_vector(row) -> np.ndarray:
    return np.asarray([_f(_get(row, c)) for c in FEATURE_COLUMNS], dtype="float64")


def drivers_for(model, row, n: int = N_DRIVERS) -> list[dict]:
    """Οι κυριότεροι παράγοντες (θετικοί = υπέρ στόχου έναντι stop), ταξινομημένοι κατά |συνεισφορά|."""
    if model is None or row is None:
        return []
    try:
        contrib = model.contributions(feature_vector(row))
    except Exception:  # η εξήγηση δεν πρέπει ποτέ να «ρίχνει» τη σελίδα
        return []
    items = [(k, v) for k, v in contrib.items() if math.isfinite(v)]
    items.sort(key=lambda kv: -abs(kv[1]))
    out = []
    for k, v in items[:n]:
        out.append({"feature": k, "label_el": FEATURE_LABELS_EL.get(k, k), "short_el": METRIC_LABELS_EL.get(k, k),
                    "contribution": float(v), "value": _val(_get(row, k)), "fmt": FEATURE_FMT.get(k, "num")})
    return out


# ---------------------------------------------------------------- αξιολόγηση

def nominal_size_eur(adv_eur, cfg) -> float:
    """Ονομαστικό μέγεθος θέσης για την κατάταξη/EV (χωρίς περιορισμό μετρητών), όπως στο backtest."""
    p = cfg.portfolio
    adv = _f(adv_eur)
    adv = adv if math.isfinite(adv) else 0.0
    return float(max(0.0, min(p.position_eur_max, p.capital_eur, p.max_adv_fraction * adv)))


def assess(ticker: str, row: pd.Series | None, pred, cfg, *, as_of=None, model_ready: bool = True,
           row_date=None) -> dict:
    """Αξιολόγηση μιας μετοχής στην τελευταία της συνεδρίαση (χωρίς τους ελέγχους χαρτοφυλακίου).

    `row`: η τελευταία γραμμή του panel για τη μετοχή (ή None)· `pred`: η ζωντανή πρόβλεψη (ή None)·
    `as_of`: η τελευταία ημερομηνία των δεδομένων συνολικά· `row_date`: η ημερομηνία της `row`.
    Αποτέλεσμα: dict με orb, costs, ev, stock_checks, decision_checks, decision, reason_el, candidate κ.ά.
    Το `allocate` συμπληρώνει έπειτα portfolio_checks, size_eur και την τελική απόφαση των υποψηφίων.
    """
    a = {"ticker": ticker, "has_row": row is not None, "as_of": date_iso(as_of), "row_date": date_iso(row_date),
         "current": False, "orb": None, "pred": None, "costs": None, "nominal_eur": None, "ev": None,
         "tp_ret": None, "sl_ret": None, "timeout_mean": None, "stock_checks": [], "decision_checks": [],
         "portfolio_checks": [], "decision": "none", "reason_el": "", "short_el": "", "candidate": False, "size_eur": None,
         "shares": None, "plan": None, "entry_ref": None}
    if row is None:
        a["reason_el"] = "Δεν υπάρχουν δεδομένα τιμών για αυτή τη μετοχή."
        a["short_el"] = "δεν υπάρχουν δεδομένα"
        return a
    a["current"] = a["as_of"] is None or a["row_date"] is None or a["row_date"] == a["as_of"]
    close = _f(_get(row, "close"))
    atr_pct = _f(_get(row, "atr_pct"))
    adv = _f(_get(row, "adv20_eur"))
    vol = _f(_get(row, "vol_20"))
    a["entry_ref"] = _val(close)
    if math.isfinite(atr_pct) and atr_pct > 0:
        a["tp_ret"] = float(cfg.barriers.k_tp) * atr_pct
        a["sl_ret"] = float(cfg.barriers.k_sl) * atr_pct

    nominal = nominal_size_eur(adv, cfg)
    a["nominal_eur"] = nominal
    a["costs"] = quant.round_trip_cost_pct(nominal, adv, close, cfg, vol=vol if math.isfinite(vol) else None)
    a["stock_checks"] = guard.stock_checks(row, cfg)

    orb = orb_dict(pred) if pred is not None else None
    if orb is not None:
        tm = _f(_get(pred, "timeout_mean"))
        a["timeout_mean"] = tm if math.isfinite(tm) else 0.0
        a["pred"] = {k: _val(_get(pred, k)) for k in ("p_tp", "p_sl", "p_to")}
        if a["tp_ret"] is not None:
            ev = quant.expected_value(orb["p_tp"], orb["p_sl"], orb["p_to"], a["tp_ret"], a["sl_ret"],
                                      a["timeout_mean"], a["costs"]["total"])
            a["ev"] = ev
            orb["ev_net"] = ev["ev_net"]
        else:
            orb["ev_net"] = None
        a["orb"] = orb
        a["decision_checks"] = guard.decision_checks(a["ev"] or {}, orb["p_tp"], cfg)

    # η απόφαση (πριν από το χαρτοφυλάκιο)
    if not a["current"]:
        a["decision"] = "none"
        a["reason_el"] = (f"Τα δεδομένα της μετοχής σταματούν στις {date_el(a['row_date'])} "
                          f"(η αγορά έχει δεδομένα έως {date_el(a['as_of'])}) — δεν υπάρχει τρέχον σήμα.")
        a["short_el"] = "τα δεδομένα δεν είναι ενημερωμένα"
        return a
    fail = guard.first_failure(a["stock_checks"])
    if fail is not None:
        a["decision"] = "blocked"
        a["reason_el"] = f"{fail.label_el}: {_lc(fail.detail_el)}."     # ο τίτλος λέει ήδη «μπλοκ»
        a["short_el"] = f"μπλοκ από τον Φύλακα ({fail.label_el.lower()})"
        return a
    if not model_ready:
        a["decision"] = "none"
        a["reason_el"] = ("Δεν υπάρχει ακόμη εκπαιδευμένο μοντέλο: χρειάζεται περισσότερο ιστορικό "
                          f"(τουλάχιστον {fmt_num(cfg.model.min_train_days)} συνεδριάσεις).")
        a["short_el"] = "δεν υπάρχει ακόμη μοντέλο"
        return a
    if orb is None or a["ev"] is None:
        a["decision"] = "none"
        a["reason_el"] = "Δεν υπάρχει πρόβλεψη για την τελευταία συνεδρίαση."
        a["short_el"] = "δεν υπάρχει πρόβλεψη"
        return a
    fail = guard.first_failure(a["decision_checks"])
    if fail is not None:
        a["decision"] = "skip"
        if fail.id == "ev":
            a["reason_el"] = (f"Αναμενόμενη απόδοση μετά τα κόστη {fmt_pct(a['ev']['ev_net'], 2, sign=True)} — "
                              f"κάτω από το ελάχιστο {fmt_pct(cfg.guard.min_ev_net, 2, sign=True)}.")
            a["short_el"] = "το αναμενόμενο καθαρό κέρδος είναι κάτω από το όριο"
        else:
            a["reason_el"] = f"{fail.label_el}: {_lc(fail.detail_el)}."
            a["short_el"] = f"{fail.label_el.lower()} κάτω από το όριο"
        return a
    a["candidate"] = True
    a["decision"] = "take"            # προσωρινά· το allocate αποφασίζει οριστικά
    a["reason_el"] = ""
    return a


def _plan(a: dict, cfg) -> dict | None:
    if a.get("entry_ref") is None or a.get("tp_ret") is None or a.get("orb") is None:
        return None
    e = a["entry_ref"]
    size = a["size_eur"] if a["decision"] == "take" else None
    return {
        "entry_ref": e,
        "tp_price": e * (1.0 + a["tp_ret"]),
        "sl_price": e * (1.0 - a["sl_ret"]),
        "tp_ret": a["tp_ret"],
        "sl_ret": a["sl_ret"],
        "horizon_days": int(cfg.barriers.horizon_days),
        "size_eur": size,
        "shares": (size / e) if size and e else None,
    }


def allocate(assessments: list[dict], sectors: dict[str, str], open_positions: list[dict], cfg,
             cash_eur: float | None = None) -> list[dict]:
    """Έλεγχοι χαρτοφυλακίου και μέγεθος θέσης για τις υποψήφιες της ημέρας (κατά ev_net, φθίνουσα).

    `open_positions`: ήδη ανοιχτές (εικονικές) θέσεις [{"ticker","sector","size_eur"}]· `cash_eur`: διαθέσιμα
    (προεπιλογή: κεφάλαιο − ποσά ανοιχτών θέσεων). Ενημερώνει τα dict επιτόπου και επιστρέφει τις θέσεις
    μετά τις σημερινές λήψεις.
    """
    held = [dict(p) for p in open_positions]
    if cash_eur is None:
        cash_eur = float(cfg.portfolio.capital_eur) - sum(_f(p.get("size_eur")) for p in held
                                                         if _ok(p.get("size_eur")))
    cash = max(float(cash_eur), 0.0)
    cands = [a for a in assessments if a.get("candidate")]
    cands.sort(key=lambda a: (-(a["ev"]["ev_net"]), a["ticker"]))
    for a in cands:
        sector = sectors.get(a["ticker"], "Άλλο")
        checks = guard.portfolio_checks(a["ticker"], sector, held, cfg)
        a["portfolio_checks"] = checks
        fail = guard.first_failure(checks)
        if fail is not None:
            a["decision"] = "skip"
            a["reason_el"] = f"{fail.label_el}: {_lc(fail.detail_el)}."
            a["short_el"] = {"max_positions": "το χαρτοφυλάκιο είναι γεμάτο",
                             "sector": "συμπληρώθηκε το όριο του κλάδου",
                             "not_held": "υπάρχει ήδη θέση στη μετοχή"}.get(fail.id, fail.label_el.lower())
            continue
        size = guard.position_size_eur(cash, _f(a.get("adv20_eur")), cfg)
        if size < guard.MIN_POSITION_EUR:
            a["decision"] = "skip"
            a["reason_el"] = (f"Το διαθέσιμο ποσό θέσης ({eur_short(size)}) είναι κάτω από το ελάχιστο "
                              f"{eur_short(guard.MIN_POSITION_EUR)} — η ελάχιστη προμήθεια θα «έτρωγε» την απόδοση.")
            a["short_el"] = "ανεπαρκή διαθέσιμα"
            continue
        buy = quant.side_cost_pct(size, a.get("adv20_eur"), a.get("entry_ref"), cfg, "buy",
                                  vol=a.get("vol_20"))["total"]
        if size * (1.0 + buy) > cash:
            size = cash / (1.0 + buy)
            if size < guard.MIN_POSITION_EUR:
                a["decision"] = "skip"
                a["reason_el"] = (f"Μετά το κόστος αγοράς τα διαθέσιμα ({eur_short(size)}) δεν φτάνουν για το "
                                  f"ελάχιστο ποσό θέσης {eur_short(guard.MIN_POSITION_EUR)}.")
                a["short_el"] = "ανεπαρκή διαθέσιμα"
                continue
        a["decision"] = "take"
        a["size_eur"] = float(size)
        a["shares"] = float(size / a["entry_ref"]) if a.get("entry_ref") else None
        ev = a["ev"]["ev_net"]
        a["reason_el"] = (f"Αναμενόμενη απόδοση μετά τα κόστη {fmt_pct(ev, 2, sign=True)} (όριο "
                          f"{fmt_pct(cfg.guard.min_ev_net, 2, sign=True)}) και όλοι οι έλεγχοι πέρασαν.")
        cash -= size * (1.0 + buy)
        held.append({"ticker": a["ticker"], "sector": sector, "size_eur": size})
    # έλεγχοι χαρτοφυλακίου και για τις υπόλοιπες (μόνο για εμφάνιση)
    for a in assessments:
        if not a.get("candidate") and a.get("has_row"):
            a["portfolio_checks"] = guard.portfolio_checks(a["ticker"], sectors.get(a["ticker"], "Άλλο"), held, cfg)
    for a in assessments:
        a["plan"] = _plan(a, cfg)
    return held


def prepare_assessment(ticker: str, row, pred, cfg, *, as_of=None, model_ready=True, row_date=None) -> dict:
    """`assess` + τα πεδία της γραμμής που χρειάζεται το `allocate` (τζίρος, μεταβλητότητα)."""
    a = assess(ticker, row, pred, cfg, as_of=as_of, model_ready=model_ready, row_date=row_date)
    a["adv20_eur"] = _val(_get(row, "adv20_eur")) if row is not None else None
    a["vol_20"] = _val(_get(row, "vol_20")) if row is not None else None
    return a


# ---------------------------------------------------------------- οι έξι κάρτες

def _liquidity_level(adv: float, cfg) -> tuple[str, str]:
    if not math.isfinite(adv):
        return "άγνωστη", "warn"
    if adv >= HIGH_LIQUIDITY_EUR:
        return "υψηλή", "ok"
    if adv >= cfg.guard.min_adv_eur:
        return "μέτρια", "ok"
    return "χαμηλή", "warn"


def _scout_card(frame, row, a, stock, cfg, ctx) -> dict:
    if row is None:
        return _agent("scout", "warn", "Δεν υπάρχουν δεδομένα τιμών για τη μετοχή",
                      notes=["Τρέξτε «Ενημέρωση» για λήψη δεδομένων· αν το σύμβολο δεν υπάρχει στο Yahoo Finance, "
                             "η μετοχή παραλείπεται.",
                             "Μπορείτε να δώσετε και δικά σας δεδομένα σε data/csv/<ΣΥΜΒΟΛΟ>.csv "
                             "(date,open,high,low,close,volume) με data.source = \"csv\"."])
    adv = _f(_get(row, "adv20_eur"))
    level, status = _liquidity_level(adv, cfg)
    headline = f"Ρευστότητα {level} · τζίρος 20ημ. {eur_short(adv)}"
    if level == "χαμηλή":
        headline += f" (όριο {eur_short(cfg.guard.min_adv_eur)})"
    hist = _f(_get(row, "history_days"))
    if not a.get("current", True):
        status = "warn"
        headline = f"Δεδομένα έως {date_el(a.get('row_date'))} — όχι ενημερωμένα"
    elif math.isfinite(hist) and hist < cfg.guard.min_history_days:
        status = "warn"
        headline = f"Σύντομο ιστορικό ({fmt_num(hist)} συνεδριάσεις) · τζίρος 20ημ. {eur_short(adv)}"

    chg = _f(_get(row, "ret_1"))
    metrics = [
        _metric("Τελευταίο κλείσιμο", _get(row, "close"), "eur"),
        _metric("Μεταβολή ημέρας", chg, "pct"),
        _metric("Μέσος τζίρος 20ημ.", adv, "eur", "Μέση ημερήσια αξία συναλλαγών (τιμή × όγκος) των τελευταίων 20 "
                                                 "συνεδριάσεων"),
        _metric("Ημέρες ιστορικού", hist, "int"),
    ]
    notes = []
    src = SOURCE_EL.get((ctx or {}).get("source") or "", None)
    if (ctx or {}).get("mode") == "demo":
        notes.append("ΕΠΙΔΕΙΞΗ: οι τιμές είναι συνθετικές — δεν αντιστοιχούν στην πραγματική μετοχή.")
    elif src:
        notes.append(f"Πηγή: {src}· τιμές προσαρμοσμένες για μερίσματα και splits.")
    notes.append(f"Τελευταία συνεδρίαση: {date_el(a.get('row_date'))}.")
    if frame is not None and len(frame):
        tail = frame.tail(60)
        vr = pd.to_numeric(tail.get("vol_ratio"), errors="coerce") if "vol_ratio" in tail.columns else None
        if vr is not None:
            zero = int((vr == 0).sum())
            if zero:
                notes.append(f"{zero} από τις τελευταίες {len(tail)} συνεδριάσεις χωρίς συναλλαγές — "
                             "προσοχή στη ρευστότητα.")
        dates = pd.DatetimeIndex(tail.index)
        if len(dates) > 1:
            expected = len(pd.bdate_range(dates[0], dates[-1]))
            missing = max(expected - len(dates), 0)
            if missing > max(3, len(dates) // 10):
                notes.append(f"Λείπουν ~{missing} εργάσιμες ημέρες στο τελευταίο τρίμηνο (αργίες ή κενά δεδομένων).")
    if level == "υψηλή":
        notes.append("Με υψηλή ρευστότητα το spread είναι μικρό και η εντολή δεν «κινεί» την τιμή.")
    elif level == "χαμηλή":
        notes.append("Με χαμηλή ρευστότητα το spread και η επίδραση στην τιμή μεγαλώνουν σημαντικά το κόστος.")
    return _agent("scout", status, headline, metrics, notes)


def _market_card(row, a, stock, cfg, ctx) -> dict:
    if row is None:
        return _agent("market", "info", "Χωρίς δεδομένα μετοχής δεν γίνεται σύγκριση με αγορά και κλάδο")
    regime = _get(row, "regime")
    regime = regime if isinstance(regime, str) and regime in REGIME_LABELS_EL else "neutral"
    sec_rel = _f(_get(row, "sector_rel_20"))
    rel_mkt = _f(_get(row, "rel_mkt_20"))
    dd = _f(_get(row, "mkt_drawdown_10"))
    parts = []
    peers = (ctx or {}).get("sector_peers")
    if math.isfinite(sec_rel) and not (peers is not None and peers <= 1):
        parts.append(f"{fmt_pct(sec_rel, 1, sign=True)} έναντι κλάδου")
    if math.isfinite(rel_mkt):
        parts.append(f"{fmt_pct(rel_mkt, 1, sign=True)} έναντι αγοράς")
    headline = REGIME_LABELS_EL[regime]
    if parts:
        headline += " · μετοχή " + ", ".join(parts) + " (20ημ.)"
    crash = math.isfinite(dd) and dd <= -cfg.guard.crash_index_drop
    status = "block" if crash else {"bull": "ok", "bear": "warn"}.get(regime, "info")
    if crash:
        headline = f"Απότομη πτώση αγοράς ({fmt_pct(dd, 1)} σε 10 ημ.) — παύση νέων θέσεων"

    ret20 = _f(_get(row, "ret_20"))
    mkt20 = _f(_get(row, "mkt_ret_20"))
    sector_mean = ret20 - sec_rel if math.isfinite(ret20) and math.isfinite(sec_rel) else float("nan")
    sector_vs_mkt = sector_mean - mkt20 if math.isfinite(sector_mean) and math.isfinite(mkt20) else float("nan")
    sector = stock.sector if stock is not None else "Άλλο"
    trend = _f(_get(row, "mkt_trend"))
    alone = peers is not None and peers <= 1
    metrics = [
        _metric("Αγορά 20ημ.", mkt20, "pct", "Απόδοση του δείκτη τις τελευταίες 20 συνεδριάσεις"),
        _metric("Τάση αγοράς", trend, "pct", "Δείκτης έναντι του μέσου όρου 100 ημερών") if alone else
        _metric("Μετοχή vs κλάδος", sec_rel, "pct", f"Απόδοση 20ημ. της μετοχής μείον τον μέσο όρο του κλάδου «{sector}»"),
        _metric("Μετοχή vs αγορά", rel_mkt, "pct", "Απόδοση 20ημ. της μετοχής μείον την απόδοση του δείκτη"),
        _metric("Πτώση από υψηλό 10ημ.", dd, "pct", "Πόσο απέχει ο δείκτης από το υψηλότερο κλείσιμο 10 ημερών"),
    ]
    notes = [
        f"Καθεστώς «{REGIME_LABELS_EL[regime].lower()}»: ανοδικό όταν ο δείκτης είναι πάνω από "
        f"{fmt_pct(REGIME_TREND_BAND, 0)} πάνω από τον μέσο όρο 100 ημερών και ανεβαίνει σε 20 ημέρες· "
        f"πτωτικό το αντίστροφο. Τώρα: {fmt_pct(trend, 1, sign=True)} από τον μέσο όρο 100 ημερών.",
    ]
    if math.isfinite(sector_vs_mkt):
        word = "υπεραποδίδει" if sector_vs_mkt > 0 else "υστερεί"
        notes.append(f"Ο κλάδος «{sector}» {word} της αγοράς κατά {fmt_pct(abs(sector_vs_mkt), 1)} (20ημ.).")
    if peers is not None and peers <= 1:
        notes.append(f"Η μετοχή είναι η μόνη του κλάδου «{sector}» στη λίστα — η σύγκριση με τον κλάδο είναι 0.")
    if (ctx or {}).get("index_ok") is False:
        notes.append((ctx or {}).get("index_note_el") or
                     "Ο Γενικός Δείκτης δεν ήταν διαθέσιμος: η «αγορά» είναι ισοσταθμικός δείκτης των μετοχών της λίστας.")
    return _agent("market", status, headline, metrics, notes)


def _base_rates(lab_frame: pd.DataFrame | None, n: int = BASE_RATE_DAYS) -> dict | None:
    if lab_frame is None or not len(lab_frame) or "label" not in lab_frame.columns:
        return None
    y = pd.to_numeric(lab_frame["label"], errors="coerce").dropna().tail(n)
    if not len(y):
        return None
    return {"n": int(len(y)), "tp": float((y == LABEL_TP).mean()), "sl": float((y == LABEL_SL).mean()),
            "to": float((y == LABEL_TO).mean())}


def _judge_card(row, a, model, lab_frame, cfg) -> dict:
    orb = a.get("orb")
    drivers = drivers_for(model, row) if orb is not None else []
    if row is None or orb is None:
        if row is None:
            headline = "Χωρίς δεδομένα δεν γίνεται πρόβλεψη"
        elif model is None:
            headline = "Δεν υπάρχει ακόμη εκπαιδευμένο μοντέλο"
        else:
            missing = [c for c in FEATURE_COLUMNS if not _ok(_get(row, c))]
            headline = (f"Λείπουν {len(missing)} χαρακτηριστικά — δεν γίνεται πρόβλεψη" if missing
                        else "Δεν υπάρχει πρόβλεψη για την τελευταία συνεδρίαση")
        notes = []
        if row is not None and model is None:
            notes.append(f"Το μοντέλο χρειάζεται τουλάχιστον {fmt_num(cfg.model.min_train_days)} συνεδριάσεις ιστορικού.")
        return _agent("judge", "info" if row is None else "warn", headline, [], notes, drivers=[])

    p_tp, p_sl, p_to = orb["p_tp"], orb["p_sl"], orb["p_to"]
    if orb["dominant"] == "to":
        lean = "Πιθανότερη η λήξη χρόνου χωρίς στόχο ή stop"
    elif p_tp >= 1.1 * p_sl:
        lean = "Γέρνει προς τον στόχο"
    elif p_sl >= 1.1 * p_tp:
        lean = "Γέρνει προς το stop"
    else:
        lean = "Ισορροπία ανάμεσα σε στόχο και stop"
    br = _base_rates(lab_frame)
    # η «βάση»: τι θα έλεγε το μοντέλο για μια μέση περίπτωση — εξηγεί γιατί η πρόβλεψη μπορεί να γέρνει
    # προς το stop ακόμη κι όταν οι περισσότεροι παράγοντες είναι υπέρ του στόχου
    if br and br["sl"] > br["tp"] * 1.05:
        lean += f" · ιστορικά συχνότερα stop ({fmt_pct(br['sl'], 0)} έναντι {fmt_pct(br['tp'], 0)} στόχου)"
    elif br and br["tp"] > br["sl"] * 1.05:
        lean += f" · ιστορικά συχνότερα στόχος ({fmt_pct(br['tp'], 0)} έναντι {fmt_pct(br['sl'], 0)} stop)"
    pos = [d for d in drivers if d["contribution"] > 0]
    neg = [d for d in drivers if d["contribution"] < 0]
    bits = []
    if pos:
        bits.append("υπέρ στόχου: " + ", ".join(SHORT_LABELS_EL.get(d["feature"], d["label_el"]) for d in pos[:2]))
    if neg:
        bits.append("υπέρ stop: " + ", ".join(SHORT_LABELS_EL.get(d["feature"], d["label_el"]) for d in neg[:2]))
    headline = lean + (" · " + " · ".join(bits) if bits else "")
    base_lr = model.base_log_ratio() if model is not None and hasattr(model, "base_log_ratio") and drivers else None

    shown = [d["feature"] for d in drivers[:3]] or ["ret_5", "rsi_14", "dist_sma50"]
    metrics = []
    for f in shown:
        hint = None
        d = next((x for x in drivers if x["feature"] == f), None)
        if d is not None:
            hint = f"Συνεισφορά {fmt_num(d['contribution'], 2)} στον λόγο log(P στόχου / P stop)"
        metrics.append(_metric(METRIC_LABELS_EL.get(f, f), _get(row, f), FEATURE_FMT.get(f, "num"),
                               hint or FEATURE_LABELS_EL.get(f)))
    metrics.append(_metric("Δείγματα εκπαίδευσης", getattr(model, "n_train", None), "int",
                           f"Εκπαίδευση έως {date_el(getattr(model, 'trained_until', None))}"))
    kind = getattr(model, "kind", "logreg")
    notes = []
    if kind == "logreg":
        notes.append("Λογιστική παλινδρόμηση τριών καταστάσεων, walk-forward χωρίς «ματιά στο μέλλον».")
        notes.append("Οι παράγοντες δείχνουν πόσο σπρώχνει κάθε χαρακτηριστικό τον λόγο P(στόχος)/P(stop): "
                     "θετικό = υπέρ του στόχου.")
    else:
        notes.append("Gradient boosting: πιο ευέλικτο μοντέλο, χωρίς ανάλυση παραγόντων ανά πρόβλεψη.")
    if br:
        notes.append(f"Ιστορικά στη μετοχή (τελευταίες {fmt_num(br['n'])} επιλυμένες ημέρες): στόχος "
                     f"{fmt_pct(br['tp'], 0)}, stop {fmt_pct(br['sl'], 0)}, λήξη {fmt_pct(br['to'], 0)}.")
    if base_lr is not None:
        notes.append("Η γραμμή «Βάση» δείχνει πού ξεκινά το μοντέλο για μια μέση περίπτωση (από τις ιστορικές "
                     "συχνότητες)· οι παράγοντες από κάτω τη μετατοπίζουν.")
    notes.append(f"Πρόβλεψη: στόχος {fmt_pct(p_tp, 0)} · stop {fmt_pct(p_sl, 0)} · λήξη {fmt_pct(p_to, 0)}.")
    extra = {"drivers": drivers}
    if base_lr is not None and math.isfinite(base_lr):
        extra["drivers_base"] = {"label_el": "Βάση (μέση περίπτωση)", "contribution": float(base_lr)}
    return _agent("judge", "info", headline, metrics, notes, **extra)


def _quant_card(row, a, cfg) -> dict:
    ev, orb, costs = a.get("ev"), a.get("orb"), a.get("costs")
    if ev is None or orb is None:
        headline = "Δεν υπολογίζεται αναμενόμενη αξία χωρίς πρόβλεψη"
        metrics = []
        if costs is not None:
            metrics.append(_metric("Κόστος μετ' επιστροφής", costs["total"], "pct",
                                   f"Για εντολή {eur_short(a.get('nominal_eur'))}"))
        return _agent("quant", "info", headline, metrics, [])
    be = ev["breakeven_p_tp"]
    headline = f"EV καθαρό {fmt_pct(ev['ev_net'], 2, sign=True)}"
    if math.isfinite(be):
        if be > 1:
            headline += " · κανένα P(στόχος) δεν καλύπτει το κόστος"
        else:
            nd = 0 if round(be * 100) != round(orb["p_tp"] * 100) else 1    # να φαίνεται η διαφορά
            headline += f" · νεκρό σημείο P(στόχος) {fmt_pct(be, nd)} (μοντέλο: {fmt_pct(orb['p_tp'], nd)})"
    ok = ev["ev_net"] >= cfg.guard.min_ev_net
    status = "ok" if ok else "warn"
    breakdown = (f"Προμήθεια {fmt_pct(costs['commission'], 2)} · χρεώσεις {fmt_pct(costs['fees'], 2)} · "
                 f"spread {fmt_pct(costs['spread'], 2)} · επίδραση {fmt_pct(costs['impact'], 2)} · "
                 f"φόρος {fmt_pct(costs['tax'], 2)}")
    metrics = [
        _metric("EV καθαρό", ev["ev_net"], "pct", "Αναμενόμενη απόδοση ανά συναλλαγή μετά από όλα τα κόστη"),
        _metric("EV μικτό", ev["ev_gross"], "pct", "P(στόχος)·στόχος − P(stop)·stop + P(λήξη)·μέση απόδοση λήξης"),
        _metric("Κόστος μετ' επιστροφής", costs["total"], "pct", breakdown),
        _metric("Νεκρό σημείο P(στόχος)", be if math.isfinite(be) and be <= 1 else None, "pct",
                "Η P(στόχος) που μηδενίζει το καθαρό EV (με ίδια P(λήξη))"),
    ]
    notes = [
        f"Στόχος {fmt_pct(a['tp_ret'], 2, sign=True)} · stop {fmt_pct(-a['sl_ret'], 2, sign=True)} "
        f"({fmt_num(cfg.barriers.k_tp, 1)}× και {fmt_num(cfg.barriers.k_sl, 1)}× το ATR/τιμή).",
        f"Κόστη για εντολή {eur_short(a.get('nominal_eur'))} (αγορά + πώληση): {breakdown}.",
        f"Μέση μικτή απόδοση όταν λήγει ο χρόνος (από την εκπαίδευση): {fmt_pct(a.get('timeout_mean'), 2, sign=True)}.",
    ]
    if not ok:
        notes.append(f"Το ελάχιστο καθαρό EV για θέση είναι {fmt_pct(cfg.guard.min_ev_net, 2, sign=True)}.")
    return _agent("quant", status, headline, metrics, notes)


def _guard_card(row, a, cfg) -> dict:
    checks = list(a.get("stock_checks") or []) + list(a.get("decision_checks") or []) + \
        list(a.get("portfolio_checks") or [])
    as_dicts = [c.to_dict() for c in checks]
    if row is None:
        return _agent("guard", "info", "Δεν υπάρχουν δεδομένα για έλεγχο", [], [], checks=[])
    blocking = [c for c in checks if c.blocking]
    n_pass = sum(1 for c in blocking if c.passed)
    stock_fail = guard.first_failure(a.get("stock_checks") or [])
    other_fail = guard.first_failure((a.get("decision_checks") or []) + (a.get("portfolio_checks") or []))
    soft = [c for c in checks if not c.blocking and not c.passed]
    if stock_fail is not None:
        status = "block"
        headline = f"Μπλοκ: {stock_fail.label_el.lower()} — {_lc(stock_fail.detail_el)}"
    elif a.get("decision") == "none":
        status = "info"
        headline = f"Οι έλεγχοι της μετοχής πέρασαν ({n_pass}/{len(blocking)}) — αναμονή πρόβλεψης"
    elif other_fail is not None and other_fail.id == "ev":
        status = "warn"
        ev = (a.get("ev") or {}).get("ev_net")
        headline = (f"Απόρριψη: καθαρό EV {fmt_pct(ev, 2, sign=True)} κάτω από το όριο "
                    f"{fmt_pct(cfg.guard.min_ev_net, 2, sign=True)}")
    elif other_fail is not None:
        status = "warn"
        headline = f"Απόρριψη: {other_fail.label_el.lower()} — {_lc(other_fail.detail_el)}"
    elif a.get("decision") == "skip":
        status = "warn"
        headline = "Απόρριψη: " + (a.get("short_el") or "δεν πληρούνται οι όροι")
    else:
        status = "ok" if not soft else "warn"
        headline = f"Όλοι οι έλεγχοι πέρασαν ({n_pass}/{len(blocking)})"
        if soft:
            headline += f" · προσοχή: {soft[0].label_el.lower()}"
    metrics = [_metric("Έλεγχοι που πέρασαν", f"{n_pass}/{len(blocking)}", "text")]
    pc = {c.id: c for c in a.get("portfolio_checks") or []}
    if "max_positions" in pc:
        metrics.append(_metric("Θέσεις σε χρήση", pc["max_positions"].detail_el.split(" θέσεις")[0], "text"))
    notes = ["Οι έλεγχοι με ✕ είναι δεσμευτικοί: αρκεί ένας για να μη γίνει θέση. Οι προειδοποιήσεις δεν μπλοκάρουν.",
             f"Παύση νέων θέσεων αν ο δείκτης πέσει πάνω από {fmt_pct(cfg.guard.crash_index_drop, 0)} σε 10 ημέρες."]
    return _agent("guard", status, headline, metrics, notes, checks=as_dicts)


def _executor_card(row, a, cfg) -> dict:
    plan = a.get("plan")
    dec = a.get("decision")
    notes = ["Εικονικές συναλλαγές: δεν αποστέλλεται καμία εντολή — οι αποφάσεις καταγράφονται στο ημερολόγιο.",
             "Έξοδος στο πρώτο από: στόχος, stop ή λήξη χρόνου· αν αγγιχτούν και τα δύο όρια την ίδια ημέρα, "
             "μετρά ως stop (συντηρητικά)."]
    if dec == "take" and plan is not None:
        size = plan["size_eur"]
        headline = (f"Αγορά στο άνοιγμα της επόμενης συνεδρίασης · {eur_short(size)}"
                    + (f" (~{fmt_num(plan['shares'], 0)} τεμ.)" if plan.get("shares") else ""))
        status = "ok"
        notes.insert(1, "Η είσοδος γίνεται στο άνοιγμα: η τιμή μπορεί να διαφέρει από το κλείσιμο αναφοράς — "
                        "στόχος και stop υπολογίζονται τότε από την πραγματική τιμή εισόδου.")
    elif dec == "blocked":
        headline = "Καμία θέση — μπλοκ από τον Φύλακα"
        status = "block"
    elif dec == "skip":
        headline = "Καμία θέση — " + (a.get("short_el") or "απόρριψη")
        status = "info"
    else:
        headline = "Κανένα σχέδιο θέσης" + (" — " + a["short_el"] if a.get("short_el") else "")
        status = "info"
    metrics = []
    if plan is not None:
        metrics = [
            _metric("Είσοδος (αναφ.)", plan["entry_ref"], "eur", "Τελευταίο κλείσιμο· η πραγματική είσοδος είναι "
                                                                 "στο επόμενο άνοιγμα"),
            _metric("Στόχος", plan["tp_price"], "eur"),
            _metric("Stop", plan["sl_price"], "eur"),
            _metric("Ποσό", plan["size_eur"], "eur") if plan.get("size_eur") else
            _metric("Ορίζοντας (συνεδριάσεις)", plan["horizon_days"], "int"),
        ]
        notes.append(f"Ορίζοντας: έως {fmt_num(plan['horizon_days'])} συνεδριάσεις (η ημέρα εισόδου μετρά ως 1η).")
    return _agent("executor", status, headline, metrics, notes)


def agent_report(ticker, panel, labels, preds_row_or_None, model, stock, cfg, open_positions=(),
                 assessment: dict | None = None, context: dict | None = None) -> list[dict]:
    """Οι έξι κάρτες πρακτόρων για το `/api/stock/{ticker}`.

    `panel`/`labels`: είτε ολόκληρα (MultiIndex date,ticker) είτε μόνο οι γραμμές της μετοχής.
    `preds_row_or_None`: η ζωντανή πρόβλεψη (dict ή Series με p_tp, p_sl, p_to, timeout_mean).
    `assessment`: έτοιμο αποτέλεσμα `assess`/`allocate` (αλλιώς υπολογίζεται εδώ, με `open_positions`).
    `context`: {"mode", "source", "index_ok", "as_of", "sector_peers"} για τις σημειώσεις.
    """
    frame = ticker_frame(panel, ticker)
    lab = ticker_frame(labels, ticker)
    row = frame.iloc[-1] if frame is not None and len(frame) else None
    row_date = frame.index[-1] if row is not None else None
    ctx = context or {}
    a = assessment
    if a is None:
        a = prepare_assessment(ticker, row, preds_row_or_None, cfg, as_of=ctx.get("as_of", row_date),
                               model_ready=model is not None or preds_row_or_None is not None, row_date=row_date)
        sectors = {ticker: stock.sector if stock is not None else "Άλλο"}
        allocate([a], sectors, [dict(p) for p in open_positions], cfg)
    return [
        _scout_card(frame, row, a, stock, cfg, ctx),
        _market_card(row, a, stock, cfg, ctx),
        _judge_card(row, a, model, lab, cfg),
        _quant_card(row, a, cfg),
        _guard_card(row, a, cfg),
        _executor_card(row, a, cfg),
    ]
