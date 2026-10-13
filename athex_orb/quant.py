"""«Ποσοτικός»: κόστος εκτέλεσης και αναμενόμενη αξία (EV) μιας συναλλαγής.

Όλα τα κόστη επιστρέφονται ως κλάσματα της αξίας της εντολής (0.001 = 0,1%) ανά πλευρά:
* προμήθεια    = max(commission_pct, commission_min_eur / εντολή)
* χρεώσεις     = exchange_fees_pct
* φόρος        = sell_tax_pct, μόνο στην πώληση
* spread       = (spread_bps_base / 1e4) × clip(√(5 εκ. € / μέσος τζίρος), 1, 8)   (μισό spread ανά πλευρά)
* επίδραση     = impact_coef × √(εντολή / μέσος τζίρος) × ημερήσια μεταβλητότητα (0,02 αν είναι άγνωστη)

Οι συναρτήσεις `*_vec` κάνουν τον ίδιο υπολογισμό σε numpy arrays (για το backtest).
"""
from __future__ import annotations

import math

import numpy as np

COST_KEYS = ("commission", "fees", "tax", "spread", "impact", "total")
DEFAULT_DAILY_VOL = 0.02        # ημερήσια μεταβλητότητα όταν δεν είναι γνωστή
SPREAD_REF_ADV_EUR = 5e6        # πάνω από αυτόν τον τζίρο το spread είναι το βασικό
SPREAD_MULT_MIN = 1.0
SPREAD_MULT_MAX = 8.0
ADV_FLOOR_EUR = 1e4             # άγνωστος/μηδενικός τζίρος ⇒ αντιμετωπίζεται ως πολύ χαμηλός (ακριβή εκτέλεση)
SIDES = ("buy", "sell")


def _check_side(side: str) -> str:
    if side not in SIDES:
        raise ValueError(f"Άγνωστη πλευρά εντολής «{side}» (αναμένεται buy ή sell)")
    return side


def _finite_pos(x) -> bool:
    """True αν το x είναι πεπερασμένος θετικός αριθμός."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0


# ---------------------------------------------------------------- κλιμακωτά (ένα ποσό)

def spread_pct(adv_eur, cfg) -> float:
    """Μισό spread (ανά πλευρά) ως κλάσμα, ανάλογα με τη ρευστότητα."""
    adv = float(adv_eur) if _finite_pos(adv_eur) else 0.0
    adv = max(adv, ADV_FLOOR_EUR)
    mult = min(max(math.sqrt(SPREAD_REF_ADV_EUR / adv), SPREAD_MULT_MIN), SPREAD_MULT_MAX)
    return cfg.costs.spread_bps_base / 1e4 * mult


def impact_pct(order_eur, adv_eur, cfg, vol=None) -> float:
    """Επίδραση της εντολής στην τιμή ως κλάσμα: coef × √(εντολή / τζίρος) × ημερήσια μεταβλητότητα."""
    if not _finite_pos(order_eur):
        return 0.0
    adv = max(float(adv_eur) if _finite_pos(adv_eur) else 0.0, ADV_FLOOR_EUR)
    daily_vol = float(vol) if _finite_pos(vol) else DEFAULT_DAILY_VOL
    return cfg.costs.impact_coef * math.sqrt(float(order_eur) / adv) * daily_vol


def side_cost_pct(order_eur, adv_eur, price, cfg, side: str, vol=None) -> dict:
    """Κόστος μιας πλευράς (αγορά ή πώληση) ως κλάσματα της αξίας της εντολής.

    Επιστρέφει {"commission","fees","tax","spread","impact","total"}. Το `price` δεν επηρεάζει
    (προς το παρόν) τον τύπο· κρατιέται για συμβατότητα και για μελλοντικό μοντέλο βήματος τιμής.
    Εντολή ≤ 0 (ή άγνωστη) ⇒ προμήθεια = commission_pct χωρίς ελάχιστο και μηδενική επίδραση.
    """
    _check_side(side)
    c = cfg.costs
    if _finite_pos(order_eur):
        commission = max(c.commission_pct, c.commission_min_eur / float(order_eur))
    else:
        commission = c.commission_pct
    out = {
        "commission": float(commission),
        "fees": float(c.exchange_fees_pct),
        "tax": float(c.sell_tax_pct) if side == "sell" else 0.0,
        "spread": spread_pct(adv_eur, cfg),
        "impact": impact_pct(order_eur, adv_eur, cfg, vol),
    }
    out["total"] = out["commission"] + out["fees"] + out["tax"] + out["spread"] + out["impact"]
    return out


def round_trip_cost_pct(order_eur, adv_eur, price, cfg, vol=None) -> dict:
    """Κόστος αγοράς + πώλησης (ίδιο ποσό και στις δύο πλευρές), με τα ίδια κλειδιά."""
    buy = side_cost_pct(order_eur, adv_eur, price, cfg, "buy", vol)
    sell = side_cost_pct(order_eur, adv_eur, price, cfg, "sell", vol)
    return {k: buy[k] + sell[k] for k in COST_KEYS}


def side_cost_eur(order_eur, adv_eur, price, cfg, side: str, vol=None) -> float:
    """Κόστος μιας πλευράς σε €."""
    if not _finite_pos(order_eur):
        return 0.0
    return float(order_eur) * side_cost_pct(order_eur, adv_eur, price, cfg, side, vol)["total"]


def side_cost_total(order_eur: float, adv_eur: float, cfg, side: str, vol: float | None = None) -> float:
    """Γρήγορη εκδοχή για τον βρόχο του backtest: μόνο το «total» του `side_cost_pct` (ίδιοι τύποι).
    Δέχεται απλούς αριθμούς float (όχι None για ποσό/τζίρο)."""
    if side not in SIDES:
        _check_side(side)
    c = cfg.costs
    adv = adv_eur if math.isfinite(adv_eur) and adv_eur > 0 else 0.0
    adv = max(adv, ADV_FLOOR_EUR)
    mult = min(max(math.sqrt(SPREAD_REF_ADV_EUR / adv), SPREAD_MULT_MIN), SPREAD_MULT_MAX)
    total = c.exchange_fees_pct + c.spread_bps_base / 1e4 * mult
    if side == "sell":
        total += c.sell_tax_pct
    if math.isfinite(order_eur) and order_eur > 0:
        v = vol if vol is not None and math.isfinite(vol) and vol > 0 else DEFAULT_DAILY_VOL
        total += max(c.commission_pct, c.commission_min_eur / order_eur)
        total += c.impact_coef * math.sqrt(order_eur / adv) * v
    else:
        total += c.commission_pct
    return total


# ---------------------------------------------------------------- διανυσματικά

def _as_float(a) -> np.ndarray:
    return np.asarray(a, dtype="float64")


def side_cost_pct_vec(order_eur, adv_eur, cfg, side: str, vol=None) -> dict[str, np.ndarray]:
    """Όπως το `side_cost_pct`, για numpy arrays (ίδιο σχήμα με το broadcast των εισόδων)."""
    _check_side(side)
    c = cfg.costs
    order = _as_float(order_eur)
    adv = _as_float(adv_eur)
    v = DEFAULT_DAILY_VOL if vol is None else _as_float(vol)
    order, adv, v = np.broadcast_arrays(order, adv, v)

    order_ok = np.isfinite(order) & (order > 0)
    adv_eff = np.where(np.isfinite(adv) & (adv > 0), adv, 0.0)
    adv_eff = np.maximum(adv_eff, ADV_FLOOR_EUR)
    v_eff = np.where(np.isfinite(v) & (v > 0), v, DEFAULT_DAILY_VOL)
    safe_order = np.where(order_ok, order, 1.0)

    commission = np.where(order_ok, np.maximum(c.commission_pct, c.commission_min_eur / safe_order),
                          c.commission_pct)
    fees = np.full(order.shape, float(c.exchange_fees_pct))
    tax = np.full(order.shape, float(c.sell_tax_pct) if side == "sell" else 0.0)
    mult = np.clip(np.sqrt(SPREAD_REF_ADV_EUR / adv_eff), SPREAD_MULT_MIN, SPREAD_MULT_MAX)
    spread = c.spread_bps_base / 1e4 * mult
    impact = np.where(order_ok, c.impact_coef * np.sqrt(safe_order / adv_eff) * v_eff, 0.0)
    total = commission + fees + tax + spread + impact
    return {"commission": commission, "fees": fees, "tax": tax, "spread": spread, "impact": impact,
            "total": total}


def round_trip_cost_pct_vec(order_eur, adv_eur, cfg, vol=None) -> dict[str, np.ndarray]:
    buy = side_cost_pct_vec(order_eur, adv_eur, cfg, "buy", vol)
    sell = side_cost_pct_vec(order_eur, adv_eur, cfg, "sell", vol)
    return {k: buy[k] + sell[k] for k in COST_KEYS}


# ---------------------------------------------------------------- αναμενόμενη αξία

def expected_value_vec(p_tp, p_sl, p_to, tp_ret, sl_ret, timeout_mean, cost_rt) -> dict[str, np.ndarray]:
    """Διανυσματική εκδοχή του `expected_value` (ίδιοι τύποι)."""
    p_tp, p_sl, p_to, tp_ret, sl_ret, tm, cost = (
        _as_float(x) for x in (p_tp, p_sl, p_to, tp_ret, sl_ret, timeout_mean, cost_rt))
    ev_gross = p_tp * tp_ret - p_sl * sl_ret + p_to * tm
    ev_net = ev_gross - cost
    denom = tp_ret + sl_ret
    with np.errstate(divide="ignore", invalid="ignore"):
        breakeven = ((1.0 - p_to) * sl_ret - p_to * tm + cost) / denom
    breakeven = np.where(np.isfinite(breakeven), breakeven, np.nan)
    return {"ev_gross": ev_gross, "ev_net": ev_net, "breakeven_p_tp": breakeven}


def expected_value(p_tp, p_sl, p_to, tp_ret, sl_ret, timeout_mean, cost_rt) -> dict:
    """Αναμενόμενη απόδοση μιας συναλλαγής από τις τρεις πιθανότητες της «Σφαίρας».

    * ev_gross = p_tp × tp_ret − p_sl × sl_ret + p_to × timeout_mean
    * ev_net   = ev_gross − cost_rt   (κόστος μετ' επιστροφής, κλάσμα· δεκτό και το dict του round_trip_cost_pct)
    * breakeven_p_tp = η P(στόχος) που μηδενίζει το ev_net αν η P(λήξη χρόνου) μείνει ίδια
      (p_sl = 1 − p_to − p_tp). Δεν περικόπτεται στο [0, 1]: τιμή > 1 σημαίνει ότι καμία πιθανότητα
      δεν αρκεί για να καλυφθεί το κόστος. NaN αν tp_ret + sl_ret = 0.
    """
    if isinstance(cost_rt, dict):
        cost_rt = cost_rt["total"]
    tm = 0.0 if timeout_mean is None else timeout_mean
    r = expected_value_vec(p_tp, p_sl, p_to, tp_ret, sl_ret, tm, cost_rt)
    if all(np.ndim(x) == 0 for x in (p_tp, p_sl, p_to, tp_ret, sl_ret, tm, cost_rt)):
        return {k: float(v) for k, v in r.items()}
    return r
