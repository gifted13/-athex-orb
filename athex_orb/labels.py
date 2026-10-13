"""Οι τρεις καταστάσεις της «Σφαίρας» (triple barrier) και ο κοινός προσομοιωτής εξόδου.

Κανόνες (ίδιοι για ετικέτες, backtest και ημερολόγιο):
* Απόφαση στο κλείσιμο της ημέρας t· είσοδος στο άνοιγμα της t+1 (`entry_idx`).
* Στόχος = είσοδος × (1 + k_tp × ATR%_t), stop = είσοδος × (1 − k_sl × ATR%_t).
* Εξετάζονται οι ημέρες entry_idx … entry_idx + horizon − 1 (η ημέρα εισόδου μετρά ως 1η).
  Για ημέρες μετά την είσοδο, αν το άνοιγμα ξεπερνά ήδη ένα όριο (χάσμα) η έξοδος γίνεται στο άνοιγμα.
  Αν μέσα στην ίδια ημέρα αγγίζονται και τα δύο όρια ⇒ θεωρούμε stop (συντηρητικά).
* Αν δεν αγγιχτεί όριο ⇒ λήξη χρόνου στο κλείσιμο της τελευταίας ημέρας.
* Αν τα δεδομένα τελειώνουν νωρίτερα ⇒ ανεπίλυτο (−1, −1, NaN).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_TP, LABEL_SL, LABEL_TO = 0, 1, 2
LABEL_NAMES_EL = {0: "Στόχος πρώτα", 1: "Stop πρώτα", 2: "Λήξη χρόνου"}
UNRESOLVED = (-1, -1, float("nan"))

LABEL_COLUMNS = ["label", "entry_date", "entry_price", "tp_ret", "sl_ret", "tp_price", "sl_price",
                 "exit_date", "exit_price", "gross_ret", "resolved_date", "bars_held"]


def barrier_prices(entry_price, atr_pct, k_tp: float, k_sl: float):
    """(tp_price, sl_price) για τιμή εισόδου και ATR% της ημέρας απόφασης (δέχεται και numpy arrays)."""
    return entry_price * (1.0 + k_tp * atr_pct), entry_price * (1.0 - k_sl * atr_pct)


def simulate_exit(o, h, l, c, entry_idx: int, entry_price: float, tp_price: float, sl_price: float,
                  horizon: int) -> tuple[int, int, float]:
    """Προσομοίωση μίας θέσης. Επιστρέφει (label, exit_idx, exit_price) ή (−1, −1, NaN) αν δεν επιλύθηκε."""
    n = len(o)
    if entry_idx < 0 or entry_idx >= n or horizon < 1:
        return UNRESOLVED
    last = entry_idx + horizon - 1
    for j in range(entry_idx, min(last, n - 1) + 1):
        if j > entry_idx:
            if o[j] >= tp_price:
                return LABEL_TP, j, float(o[j])
            if o[j] <= sl_price:
                return LABEL_SL, j, float(o[j])
        if l[j] <= sl_price:                 # και αν ταυτόχρονα h ≥ στόχου ⇒ stop (συντηρητικά)
            return LABEL_SL, j, float(sl_price)
        if h[j] >= tp_price:
            return LABEL_TP, j, float(tp_price)
    if last <= n - 1:
        return LABEL_TO, last, float(c[last])
    return UNRESOLVED


def _forward(a: np.ndarray, horizon: int) -> np.ndarray:
    """Πίνακας (n, horizon): η γραμμή t περιέχει a[t+1 … t+horizon] (NaN πέρα από το τέλος)."""
    padded = np.concatenate([a.astype("float64"), np.full(horizon + 1, np.nan)])
    return np.lib.stride_tricks.sliding_window_view(padded[1:], horizon)[: len(a)]


def simulate_all(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, atr_pct: np.ndarray,
                 k_tp: float, k_sl: float, horizon: int) -> dict[str, np.ndarray]:
    """Διανυσματική εκδοχή του simulate_exit για όλες τις ημέρες απόφασης t = 0 … n−1 μίας μετοχής.

    Αποτέλεσμα ίδιο με `simulate_exit(o, h, l, c, t+1, o[t+1], tp, sl, horizon)` για κάθε t.
    Γραμμές χωρίς έγκυρο ATR% (NaN ή ≤ 0) ή χωρίς ημέρα εισόδου μένουν ανεπίλυτες.
    """
    n = len(o)
    entry_price = np.full(n, np.nan)
    entry_price[:-1] = o[1:]
    valid = np.isfinite(atr_pct) & (atr_pct > 0) & np.isfinite(entry_price)
    tp, sl = barrier_prices(entry_price, np.where(valid, atr_pct, np.nan), k_tp, k_sl)

    O, H, L, C = (_forward(x, horizon) for x in (o, h, l, c))
    tp2, sl2 = tp[:, None], sl[:, None]
    later = np.arange(horizon)[None, :] > 0          # μόνο μετά την ημέρα εισόδου ελέγχεται το χάσμα
    with np.errstate(invalid="ignore"):
        gap_tp = later & (O >= tp2)
        gap_sl = later & (O <= sl2) & ~gap_tp
        bar_sl = L <= sl2
        bar_tp = H >= tp2
    hit = gap_tp | gap_sl | bar_sl | bar_tp
    any_hit = hit.any(axis=1) & valid
    k = np.argmax(hit, axis=1)
    rows = np.arange(n)

    label = np.full(n, np.nan)
    exit_off = np.full(n, -1, dtype="int64")
    exit_price = np.full(n, np.nan)

    g_tp, g_sl, b_sl = gap_tp[rows, k], gap_sl[rows, k], bar_sl[rows, k]
    hit_label = np.where(g_tp, LABEL_TP, np.where(g_sl | b_sl, LABEL_SL, LABEL_TP))
    hit_price = np.where(g_tp | g_sl, O[rows, k], np.where(b_sl, sl, tp))
    label[any_hit] = hit_label[any_hit]
    exit_off[any_hit] = k[any_hit]
    exit_price[any_hit] = hit_price[any_hit]

    # λήξη χρόνου: η τελευταία ημέρα (t + horizon) υπάρχει στα δεδομένα
    timeout = valid & ~any_hit & (rows + horizon <= n - 1)
    label[timeout] = LABEL_TO
    exit_off[timeout] = horizon - 1
    exit_price[timeout] = C[timeout, horizon - 1]

    resolved = any_hit | timeout
    exit_idx = np.where(resolved, rows + 1 + exit_off, -1)
    return {
        "label": label, "entry_price": entry_price, "tp_price": tp, "sl_price": sl,
        "exit_idx": exit_idx, "exit_price": exit_price, "valid": valid, "resolved": resolved,
    }


def triple_barrier(prices: dict[str, pd.DataFrame], panel: pd.DataFrame, cfg) -> pd.DataFrame:
    """Ετικέτες triple barrier, ευθυγραμμισμένες με το index του panel.

    Στήλες: label (float, NaN αν ανεπίλυτο), entry_date, entry_price, tp_ret, sl_ret, tp_price, sl_price,
    exit_date, exit_price, gross_ret, resolved_date (= exit_date), bars_held.
    Τα tp_ret/sl_ret (= k × ATR%) υπάρχουν σε κάθε γραμμή με έγκυρο ATR%, ακόμη και χωρίς ημέρα εισόδου·
    τα entry_*/tp_price/sl_price μόνο όπου υπάρχει ημέρα εισόδου· τα exit_*/label μόνο αν επιλύθηκε.
    """
    k_tp, k_sl = float(cfg.barriers.k_tp), float(cfg.barriers.k_sl)
    horizon = int(cfg.barriers.horizon_days)
    m = len(panel)
    nat = np.datetime64("NaT", "us")
    out = {
        "label": np.full(m, np.nan),
        "entry_date": np.full(m, nat, dtype="datetime64[us]"),
        "entry_price": np.full(m, np.nan),
        "tp_ret": np.full(m, np.nan),
        "sl_ret": np.full(m, np.nan),
        "tp_price": np.full(m, np.nan),
        "sl_price": np.full(m, np.nan),
        "exit_date": np.full(m, nat, dtype="datetime64[us]"),
        "exit_price": np.full(m, np.nan),
        "bars_held": np.full(m, np.nan),
    }
    if m:
        atr_all = panel["atr_pct"].to_numpy(dtype="float64")
        dates_all = panel.index.get_level_values("date")
        for ticker, pos in panel.groupby(level="ticker", sort=False).indices.items():
            df = prices.get(ticker)
            if df is None or not len(df):
                continue
            idx = df.index
            dates_us = idx.values.astype("datetime64[us]")
            # θέση κάθε γραμμής του panel μέσα στις τιμές της μετοχής
            loc = idx.get_indexer(dates_all[pos])
            ok = loc >= 0
            pos, loc = pos[ok], loc[ok]
            atr = np.full(len(idx), np.nan)
            atr[loc] = atr_all[pos]
            r = simulate_all(df["open"].to_numpy("float64"), df["high"].to_numpy("float64"),
                             df["low"].to_numpy("float64"), df["close"].to_numpy("float64"),
                             atr, k_tp, k_sl, horizon)
            valid = r["valid"][loc]
            vp, vl = pos[valid], loc[valid]
            out["entry_date"][vp] = dates_us[vl + 1]
            out["entry_price"][vp] = r["entry_price"][vl]
            out["tp_price"][vp] = r["tp_price"][vl]
            out["sl_price"][vp] = r["sl_price"][vl]
            res = r["resolved"][loc]
            rp, rl = pos[res], loc[res]
            ex = r["exit_idx"][rl]
            out["label"][rp] = r["label"][rl]
            out["exit_date"][rp] = dates_us[ex]
            out["exit_price"][rp] = r["exit_price"][rl]
            out["bars_held"][rp] = ex - rl            # = exit_idx − entry_idx + 1, με entry_idx = t + 1

        # τα ποσοστά των ορίων δεν εξαρτώνται από την είσοδο ⇒ ορίζονται και για την τελευταία ημέρα (ζωντανό σήμα)
        atr_ok = np.isfinite(atr_all) & (atr_all > 0)
        out["tp_ret"][atr_ok] = k_tp * atr_all[atr_ok]
        out["sl_ret"][atr_ok] = k_sl * atr_all[atr_ok]

    res_df = pd.DataFrame(out, index=panel.index)
    res_df["gross_ret"] = res_df["exit_price"] / res_df["entry_price"] - 1.0
    res_df["resolved_date"] = res_df["exit_date"]
    return res_df[LABEL_COLUMNS]
