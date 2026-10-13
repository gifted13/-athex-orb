"""Συνθετικά δεδομένα OHLCV για τη λειτουργία επίδειξης και για τα tests.

Μοντέλο: συσχετισμένη γεωμετρική κίνηση Brown με
* παράγοντα αγοράς (με περιόδους υψηλής/χαμηλής μεταβλητότητας),
* παράγοντα κλάδου,
* ιδιοσυγκρασιακό θόρυβο με «παχιές ουρές» (t‑κατανομή),
* χάσματα (gaps) στο άνοιγμα και σποραδικά άλματα,
* όγκο log‑normal (με σποραδικές εκτινάξεις) ώστε ο ημερήσιος τζίρος να κυμαίνεται ~50 χιλ. € – 30 εκ. €
  ανάμεσα στις μετοχές (οι κλάδοι με μεγάλες εταιρείες, π.χ. τράπεζες, είναι κατά κανόνα πιο ρευστοί).

Κάθε σειρά (αγορά, κλάδος, μετοχή) έχει δική της γεννήτρια τυχαίων αριθμών που εξαρτάται μόνο από
τον σπόρο (seed) και το όνομά της. Έτσι:
* ίδιος σπόρος ⇒ ίδια δεδομένα,
* μια μετοχή παράγεται ίδια είτε μόνη της είτε μαζί με άλλες (χρήσιμο όταν ο χρήστης προσθέτει μετοχή σε demo),
* αν μεγαλώσει το `end`, το προηγούμενο ιστορικό μένει αμετάβλητο.

Φυτεμένο σήμα (`planted_signal > 0`, μόνο για tests): **επαναφορά μετά από κίνηση 5 ημερών**.
Σε κάθε ημέρα τ προστίθεται στην ενδοημερήσια κίνηση (μετά το άνοιγμα, άρα εκμεταλλεύσιμη από
είσοδο στο άνοιγμα) τάση
    −planted_signal × PLANTED_DRIFT × σ × tanh(ret_5[τ−1] / (σ·√5)),
όπου ret_5[τ−1] η (λογαριθμική) απόδοση 5 ημερών μέχρι το κλείσιμο της προηγούμενης ημέρας. Μετά
από πτώση ακολουθεί άνοδος και αντίστροφα. Το σήμα το φέρει το χαρακτηριστικό **ret_5** (και τα
συσχετισμένα rsi_14, dist_sma20)· ένα λογιστικό μοντέλο στα FEATURE_COLUMNS το βρίσκει εύκολα.
Με `planted_signal = 0` δεν προστίθεται τίποτα ⇒ καθαρός τυχαίος περίπατος (με μικρή σταθερή θετική
τάση αγοράς), χωρίς εκμεταλλεύσιμο πλεονέκτημα. Οι υπόλοιπες τυχαίες σειρές είναι ίδιες σε κάθε περίπτωση.
"""
from __future__ import annotations

import math
import zlib

import numpy as np
import pandas as pd

from .universe import Stock

# Παράμετροι αγοράς / κλάδων (ημερήσιες, σε λογαριθμικές αποδόσεις)
MARKET_DRIFT = 0.00022          # ≈ 5,5% ετησίως
MARKET_VOL = 0.011
SECTOR_VOL = 0.007
VOL_REGIME_PHI = 0.98           # εμμονή περιόδων μεταβλητότητας
VOL_REGIME_SHOCK = 0.08

# Ρευστότητα: μέσος ημερήσιος τζίρος ανά μετοχή στο [ADV_MIN_EUR, ADV_MAX_EUR]
ADV_MIN_EUR = 5e4
ADV_MAX_EUR = 3e7
SECTOR_LIQUIDITY = {            # «προτίμηση» ρευστότητας ανά κλάδο (0 = πολύ χαμηλή, 1 = πολύ υψηλή)
    "Τράπεζες": 0.80,
    "Τηλεπικοινωνίες": 0.75,
    "Ψυχαγωγία": 0.75,
    "Ενέργεια": 0.65,
    "Βιομηχανία": 0.50,
    "Λιανική": 0.50,
    "Κατασκευές": 0.45,
    "Υποδομές": 0.40,
    "Μεταφορές": 0.40,
    "Ακίνητα": 0.35,
    "Καταναλωτικά": 0.30,
}
DEFAULT_LIQUIDITY = 0.25
LIQUIDITY_SPREAD = 0.70         # τυχαία απόκλιση ανά μετοχή γύρω από την προτίμηση του κλάδου

# Χάσματα / άλματα
GAP_SHARE = 0.25                # μέρος της ημερήσιας κίνησης που γίνεται «εκτός συνεδρίασης»
JUMP_PROB = 0.004
JUMP_SCALE = 4.0                # σε μονάδες ημερήσιας μεταβλητότητας

# Σποραδικές εκτινάξεις όγκου (ειδήσεις κ.λπ.)
SPIKE_PROB = 0.04
SPIKE_VOLUME_MULT = (2.5, 5.0)

# Φυτεμένο σήμα (βλ. docstring)
PLANTED_LOOKBACK = 5
PLANTED_DRIFT = 0.50            # μέγιστη ημερήσια τάση σε μονάδες σ της μετοχής, για planted_signal = 1

PRICE_DECIMALS = 4
MIN_PRICE = 0.01


def default_end() -> pd.Timestamp:
    """Η τελευταία εργάσιμη ημέρα πριν από σήμερα."""
    return (pd.Timestamp.today().normalize() - pd.offsets.BDay(1)).normalize()


def _key(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))


def _rng(seed: int, *keys: str) -> np.random.Generator:
    """Ανεξάρτητη γεννήτρια για κάθε (σπόρο, όνομα σειράς)."""
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, *(_key(k) for k in keys)])


def _fat_tails(rng: np.random.Generator, n: int, dof: float = 5.0) -> np.ndarray:
    """Θόρυβος t‑κατανομής με μοναδιαία διασπορά."""
    return rng.standard_t(dof, n) / np.sqrt(dof / (dof - 2.0))


def _ar1(eps: np.ndarray, phi: float) -> np.ndarray:
    out = np.empty_like(eps)
    x = 0.0
    for i, e in enumerate(eps):
        x = phi * x + e
        out[i] = x
    return out


def _plant_reversal(log_p0: float, gap: np.ndarray, intraday: np.ndarray, strength: float,
                    sigma: float, lookback: int = PLANTED_LOOKBACK) -> np.ndarray:
    """Προσθέτει την τάση επαναφοράς (βλ. docstring) — διαδοχικά, γιατί εξαρτάται από τις ίδιες τις τιμές."""
    n = len(gap)
    gap_l, out = gap.tolist(), intraday.tolist()
    lc = [0.0] * n
    scale = sigma * math.sqrt(lookback)
    prev = log_p0
    for i in range(n):
        if i - 1 - lookback >= 0:
            out[i] -= strength * math.tanh((lc[i - 1] - lc[i - 1 - lookback]) / scale)
        prev = prev + gap_l[i] + out[i]
        lc[i] = prev
    return np.asarray(out, dtype="float64")


def _ohlc_from_returns(log_p0: float, gap: np.ndarray, intraday: np.ndarray,
                       up: np.ndarray, down: np.ndarray) -> tuple[np.ndarray, ...]:
    """Κατασκευή OHLC από χάσμα (κλείσιμο→άνοιγμα) και ενδοημερήσια κίνηση (άνοιγμα→κλείσιμο)."""
    log_close = log_p0 + np.cumsum(gap + intraday)
    prev_close = np.concatenate(([log_p0], log_close[:-1]))
    log_open = prev_close + gap
    close = np.maximum(np.exp(log_close), MIN_PRICE)
    open_ = np.maximum(np.exp(log_open), MIN_PRICE)
    high = np.maximum(open_, close) * np.exp(up)
    low = np.minimum(open_, close) * np.exp(-down)
    rnd = lambda a: np.round(a, PRICE_DECIMALS)  # noqa: E731 — η στρογγυλοποίηση είναι μονότονη ⇒ κρατά τη συνέπεια OHLC
    return rnd(open_), rnd(high), np.maximum(rnd(low), 10.0 ** -PRICE_DECIMALS), rnd(close)


def _frame(dates: pd.DatetimeIndex, o, h, l, c, v) -> pd.DataFrame:
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=dates, dtype="float64")
    df.index.name = "date"
    return df


def generate_universe(stocks: list[Stock], start="2017-01-02", end=None, seed: int = 7,
                      planted_signal: float = 0.0) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Παράγει `(prices, index_df)` σε εργάσιμες ημέρες για τις δοσμένες μετοχές.

    `planted_signal = 0` ⇒ καθαρός τυχαίος περίπατος· `> 0` ⇒ μαθήσιμο πλεονέκτημα επαναφοράς
    που φέρει το ret_5 (βλ. docstring του module). Τιμές ~0,5–1 δίνουν ισχυρό σήμα.
    """
    end_ts = default_end() if end is None else pd.Timestamp(end).normalize()
    dates = pd.bdate_range(pd.Timestamp(start).normalize(), end_ts, name="date").as_unit("us")
    n = len(dates)
    if n == 0:
        raise ValueError("Κενό διάστημα ημερομηνιών για τα συνθετικά δεδομένα")

    # --- Παράγοντας αγοράς (και καθεστώς μεταβλητότητας) ---
    vol_mult = np.exp(_ar1(VOL_REGIME_SHOCK * _rng(seed, "market", "vol").standard_normal(n), VOL_REGIME_PHI))
    sigma_m = MARKET_VOL * vol_mult
    mkt = MARKET_DRIFT + sigma_m * _fat_tails(_rng(seed, "market", "z"), n)

    # --- Δείκτης αγοράς ---
    m_gap = 0.2 * mkt + 0.1 * sigma_m * _rng(seed, "market", "gap").standard_normal(n)
    io, ih, il, ic = _ohlc_from_returns(
        np.log(1000.0), m_gap, mkt - m_gap,
        np.abs(_rng(seed, "market", "hi").standard_normal(n)) * 0.4 * sigma_m,
        np.abs(_rng(seed, "market", "lo").standard_normal(n)) * 0.4 * sigma_m,
    )
    iv = np.round(np.exp(18.0 + 0.3 * _rng(seed, "market", "volume").standard_normal(n)))
    index_df = _frame(dates, io, ih, il, ic, iv)

    # --- Παράγοντες κλάδων ---
    sector_cache: dict[str, np.ndarray] = {}

    def sector_factor(name: str) -> np.ndarray:
        if name not in sector_cache:
            sector_cache[name] = SECTOR_VOL * vol_mult * _fat_tails(_rng(seed, "sector", name), n)
        return sector_cache[name]

    prices: dict[str, pd.DataFrame] = {}
    for stock in stocks:
        t = stock.ticker
        if t in prices:
            continue
        par = _rng(seed, "stock", t, "params")
        u_liq = par.uniform()
        beta = par.uniform(0.6, 1.4)
        gamma = par.uniform(0.5, 1.2)
        sig_i = par.uniform(0.010, 0.022)
        log_p0 = par.uniform(np.log(1.5), np.log(40.0))

        sig_tot_base = np.sqrt((beta * MARKET_VOL) ** 2 + (gamma * SECTOR_VOL) ** 2 + sig_i ** 2)
        sig_tot = np.sqrt((beta * sigma_m) ** 2 + (gamma * SECTOR_VOL * vol_mult) ** 2 + sig_i ** 2 * vol_mult)

        idio = sig_i * np.sqrt(vol_mult) * _fat_tails(_rng(seed, "stock", t, "idio"), n)
        ret = beta * mkt + gamma * sector_factor(stock.sector) + idio

        # χάσμα ανοίγματος + σποραδικά άλματα (μηδενικής μέσης τιμής)
        gap = GAP_SHARE * ret + 0.25 * sig_tot * _rng(seed, "stock", t, "gap").standard_normal(n)
        intraday = ret - gap
        jr = _rng(seed, "stock", t, "jump")
        jump_mask = jr.uniform(size=n) < JUMP_PROB
        jumps = np.where(jump_mask, JUMP_SCALE * sig_tot * _rng(seed, "stock", t, "jump_size").standard_normal(n), 0.0)
        gap = gap + jumps

        if planted_signal > 0:
            intraday = _plant_reversal(log_p0, gap, intraday, planted_signal * PLANTED_DRIFT * sig_tot_base,
                                       sig_tot_base)

        o, h, l, c = _ohlc_from_returns(
            log_p0, gap, intraday,
            np.abs(_rng(seed, "stock", t, "hi").standard_normal(n)) * 0.5 * sig_tot,
            np.abs(_rng(seed, "stock", t, "lo").standard_normal(n)) * 0.5 * sig_tot,
        )

        # όγκος: επίπεδο ρευστότητας × καθεστώς όγκου × θόρυβος × δραστηριότητα × γεγονότα
        prior = SECTOR_LIQUIDITY.get(stock.sector, DEFAULT_LIQUIDITY)
        liq = float(np.clip(prior + LIQUIDITY_SPREAD * (u_liq - 0.5), 0.0, 1.0))
        log_adv = np.log(ADV_MIN_EUR) + liq * np.log(ADV_MAX_EUR / ADV_MIN_EUR)
        regime = _ar1(0.12 * _rng(seed, "stock", t, "vol_regime").standard_normal(n), 0.9)
        noise = 0.35 * _rng(seed, "stock", t, "vol_noise").standard_normal(n)
        spikes = _rng(seed, "stock", t, "spikes").uniform(size=n) < SPIKE_PROB
        spike = np.where(spikes, _rng(seed, "stock", t, "spike_size").uniform(*SPIKE_VOLUME_MULT, size=n), 1.0)
        activity = 0.6 + 0.4 * np.abs(ret + jumps) / sig_tot
        traded_value = np.exp(log_adv + regime + noise - 0.1) * activity * spike
        volume = np.maximum(np.round(traded_value / c), 1.0)

        prices[t] = _frame(dates, o, h, l, c, volume)

    return prices, index_df
