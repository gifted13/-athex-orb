"""«Φύλακας»: έλεγχοι κινδύνου πριν από κάθε (εικονική) θέση και μέγεθος θέσης.

Τρεις ομάδες ελέγχων:
* `stock_checks`      — η ίδια η μετοχή (ρευστότητα, τιμή, μεταβλητότητα, ιστορικό, αγορά, δεδομένα),
* `decision_checks`   — η απόφαση (αναμενόμενη καθαρή απόδοση, πιθανότητα στόχου),
* `portfolio_checks`  — το χαρτοφυλάκιο (μέγιστες θέσεις, όριο κλάδου, όχι διπλή θέση).
Η θέση επιτρέπεται μόνο αν περνούν όλοι οι «μπλοκαριστικοί» έλεγχοι (`passed`).
Το `stock_checks_vec` κάνει τους ίδιους ελέγχους μετοχής σε ολόκληρο panel (για το backtest).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, FEATURE_LABELS_EL

MIN_POSITION_EUR = 300.0        # κάτω από αυτό το ποσό δεν ανοίγουμε θέση (η ελάχιστη προμήθεια «τρώει» την απόδοση)

STOCK_CHECK_IDS = ["liquidity", "price", "volatility", "history", "market", "features", "regime"]
CHECK_LABELS_EL = {
    "liquidity": "Ρευστότητα",
    "price": "Τιμή μετοχής",
    "volatility": "Μεταβλητότητα (ATR)",
    "history": "Ιστορικό",
    "market": "Κατάσταση αγοράς",
    "features": "Πλήρη δεδομένα",
    "regime": "Τάση αγοράς",
    "ev": "Αναμενόμενη καθαρή απόδοση",
    "p_tp": "Πιθανότητα στόχου",
    "max_positions": "Ελεύθερη θέση χαρτοφυλακίου",
    "sector": "Όριο κλάδου",
    "not_held": "Χωρίς διπλή θέση",
}


@dataclass
class Check:
    id: str
    passed: bool
    label_el: str
    detail_el: str
    blocking: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- μορφοποίηση κειμένων (ελληνικά)

def fmt_num(x: float, decimals: int = 0) -> str:
    """1234.5 → «1.234,5» (ελληνική μορφή)."""
    s = f"{x:,.{decimals}f}"
    return s.replace(",", " ").replace(".", ",").replace(" ", ".")


def fmt_eur(x: float) -> str:
    """Ποσό σε € για κείμενα: «1,25 εκ. €», «300 χιλ. €», «4,50 €»."""
    if x is None or not math.isfinite(x):
        return "—"
    ax = abs(x)
    if ax >= 1e6:
        return f"{fmt_num(x / 1e6, 2)} εκ. €"
    if ax >= 1e4:
        return f"{fmt_num(x / 1e3, 0)} χιλ. €"
    return f"{fmt_num(x, 2)} €"


def fmt_pct(x: float, decimals: int = 1, sign: bool = False) -> str:
    """Κλάσμα ως ποσοστό για κείμενα: 0.0345 → «3,5%» (με `sign`: «+3,5%»· αρνητικά με «−»)."""
    if x is None or not math.isfinite(x):
        return "—"
    s = fmt_num(abs(x) * 100.0, decimals) + "%"
    if x < 0:
        return "−" + s
    return ("+" + s) if sign and x > 0 else s


def _f(row, key):
    """Αριθμητική τιμή από γραμμή του panel (NaN αν λείπει)."""
    try:
        v = row[key]
    except (KeyError, IndexError):
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _ok(x) -> bool:
    """True αν το x είναι πεπερασμένος αριθμός."""
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------- έλεγχοι μετοχής

def stock_checks(row: pd.Series, cfg) -> list[Check]:
    """Έλεγχοι της μετοχής σε μια ημερομηνία (γραμμή του panel)."""
    g = cfg.guard
    out: list[Check] = []

    adv = _f(row, "adv20_eur")
    ok = _ok(adv) and adv >= g.min_adv_eur
    out.append(Check("liquidity", ok, CHECK_LABELS_EL["liquidity"],
                     f"Μέσος τζίρος 20 ημερών {fmt_eur(adv)} (ελάχιστο {fmt_eur(g.min_adv_eur)})"))

    price = _f(row, "close")
    ok = _ok(price) and price >= g.min_price_eur
    out.append(Check("price", ok, CHECK_LABELS_EL["price"],
                     f"Τιμή {fmt_eur(price)} (ελάχιστο {fmt_eur(g.min_price_eur)})"))

    atr_pct = _f(row, "atr_pct")
    ok = _ok(atr_pct) and atr_pct <= g.max_atr_pct
    out.append(Check("volatility", ok, CHECK_LABELS_EL["volatility"],
                     f"ATR / τιμή {fmt_pct(atr_pct)} (μέγιστο {fmt_pct(g.max_atr_pct)})"))

    hist = _f(row, "history_days")
    ok = _ok(hist) and hist >= g.min_history_days
    hist_txt = fmt_num(hist) if _ok(hist) else "—"
    out.append(Check("history", ok, CHECK_LABELS_EL["history"],
                     f"{hist_txt} ημέρες διαπραγμάτευσης (ελάχιστο {fmt_num(g.min_history_days)})"))

    dd = _f(row, "mkt_drawdown_10")
    if _ok(dd):
        ok = dd > -g.crash_index_drop
        detail = (f"Ο δείκτης είναι {fmt_pct(dd)} από το υψηλό 10 ημερών "
                  f"(παύση νέων θέσεων κάτω από {fmt_pct(-g.crash_index_drop)})")
    else:
        ok, detail = True, "Δεν υπάρχουν δεδομένα δείκτη — ο έλεγχος παραλείπεται"
    out.append(Check("market", ok, CHECK_LABELS_EL["market"], detail))

    missing = [c for c in FEATURE_COLUMNS if not _ok(_f(row, c))]
    if missing:
        detail = f"Λείπουν {len(missing)} από {len(FEATURE_COLUMNS)} δείκτες (π.χ. «{FEATURE_LABELS_EL[missing[0]]}»)"
    else:
        detail = "Όλοι οι δείκτες του μοντέλου είναι διαθέσιμοι"
    out.append(Check("features", not missing, CHECK_LABELS_EL["features"], detail))

    regime = row.get("regime") if hasattr(row, "get") else None
    bear = isinstance(regime, str) and regime == "bear"
    out.append(Check("regime", not bear, CHECK_LABELS_EL["regime"],
                     "Πτωτική αγορά — αυξημένη προσοχή" if bear else "Η αγορά δεν είναι σε πτωτική τάση",
                     blocking=False))
    return out


def stock_checks_vec(frame: pd.DataFrame, cfg) -> dict[str, np.ndarray]:
    """Οι ίδιοι έλεγχοι με το `stock_checks` για κάθε γραμμή ενός panel: {id: bool array}."""
    g = cfg.guard
    n = len(frame)

    def col(name):
        if name not in frame.columns:
            return np.full(n, np.nan)
        return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)

    with np.errstate(invalid="ignore"):
        adv, price, atr, hist, dd = (col(c) for c in ("adv20_eur", "close", "atr_pct", "history_days",
                                                          "mkt_drawdown_10"))
        feats = np.column_stack([col(c) for c in FEATURE_COLUMNS]) if n else np.empty((0, len(FEATURE_COLUMNS)))
        out = {
            "liquidity": np.isfinite(adv) & (adv >= g.min_adv_eur),
            "price": np.isfinite(price) & (price >= g.min_price_eur),
            "volatility": np.isfinite(atr) & (atr <= g.max_atr_pct),
            "history": np.isfinite(hist) & (hist >= g.min_history_days),
            "market": ~np.isfinite(dd) | (dd > -g.crash_index_drop),
            "features": np.isfinite(feats).all(axis=1),
        }
    if "regime" in frame.columns:
        out["regime"] = (frame["regime"].astype("str") != "bear").to_numpy(dtype=bool)
    else:
        out["regime"] = np.ones(n, dtype=bool)
    return out


def stock_pass_mask(frame: pd.DataFrame, cfg) -> np.ndarray:
    """True όπου περνούν όλοι οι μπλοκαριστικοί έλεγχοι μετοχής."""
    v = stock_checks_vec(frame, cfg)
    mask = np.ones(len(frame), dtype=bool)
    for cid in STOCK_CHECK_IDS:
        if cid != "regime":
            mask &= v[cid]
    return mask


# ---------------------------------------------------------------- απόφαση & χαρτοφυλάκιο

def decision_checks(ev: dict, p_tp: float, cfg) -> list[Check]:
    """Έλεγχοι της απόφασης: ev_net ≥ min_ev_net και p_tp ≥ min_p_tp."""
    g = cfg.guard
    ev_net = float(ev.get("ev_net", float("nan"))) if ev else float("nan")
    p = float(p_tp) if p_tp is not None else float("nan")
    return [
        Check("ev", _ok(ev_net) and ev_net >= g.min_ev_net, CHECK_LABELS_EL["ev"],
              f"Αναμενόμενη απόδοση μετά από κόστη {fmt_pct(ev_net, 2, sign=True)} "
              f"(ελάχιστο {fmt_pct(g.min_ev_net, 2, sign=True)})"),
        Check("p_tp", _ok(p) and p >= g.min_p_tp, CHECK_LABELS_EL["p_tp"],
              f"P(στόχος πρώτα) {fmt_pct(p, 0)} (ελάχιστο {fmt_pct(g.min_p_tp, 0)})"),
    ]


def portfolio_checks(ticker: str, sector: str, open_positions: list[dict], cfg) -> list[Check]:
    """Έλεγχοι χαρτοφυλακίου. `open_positions`: λίστα από dict με τουλάχιστον "ticker" και "sector"."""
    p = cfg.portfolio
    n_open = len(open_positions)
    n_sector = sum(1 for pos in open_positions if pos.get("sector") == sector)
    held = any(pos.get("ticker") == ticker for pos in open_positions)
    return [
        Check("max_positions", n_open < p.max_positions, CHECK_LABELS_EL["max_positions"],
              f"{n_open} από {p.max_positions} θέσεις σε χρήση"),
        Check("sector", n_sector < p.max_per_sector, CHECK_LABELS_EL["sector"],
              f"{n_sector} από {p.max_per_sector} θέσεις στον κλάδο «{sector}»"),
        Check("not_held", not held, CHECK_LABELS_EL["not_held"],
              "Υπάρχει ήδη ανοιχτή θέση στη μετοχή" if held else "Δεν υπάρχει ήδη θέση στη μετοχή"),
    ]


def position_size_eur(cash_eur, adv_eur, cfg) -> float:
    """Μέγεθος θέσης = min(position_eur_max, διαθέσιμα, max_adv_fraction × μέσος τζίρος 20 ημερών)."""
    p = cfg.portfolio
    cash = float(cash_eur) if _ok(cash_eur) else 0.0
    adv = float(adv_eur) if _ok(adv_eur) else 0.0
    return float(max(0.0, min(p.position_eur_max, cash, p.max_adv_fraction * adv)))


def passed(checks) -> bool:
    """True αν περνούν όλοι οι μπλοκαριστικοί έλεγχοι."""
    return all(c.passed for c in checks if c.blocking)


def first_failure(checks) -> Check | None:
    """Ο πρώτος μπλοκαριστικός έλεγχος που απέτυχε (ή None)."""
    for c in checks:
        if c.blocking and not c.passed:
            return c
    return None
