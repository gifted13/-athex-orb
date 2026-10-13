"""«Κριτής»: χαρακτηριστικά ανά μετοχή και ημερομηνία (panel).

Κάθε χαρακτηριστικό στην ημερομηνία t υπολογίζεται με κυλιόμενα παράθυρα που τελειώνουν στην t,
δηλαδή μόνο με δεδομένα ≤ t (καμία «ματιά στο μέλλον»).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import market
from .universe import Stock

FEATURE_COLUMNS: list[str] = [
    "ret_1", "ret_5", "ret_20", "ret_60",
    "dist_sma20", "dist_sma50", "dist_sma200",
    "rsi_14", "atr_pct", "vol_20", "vol_ratio", "value_z", "gap_1", "range_pos_252",
    "rel_mkt_20", "sector_rel_20",
    "mkt_ret_20", "mkt_vol_20", "mkt_trend",
]

FEATURE_LABELS_EL: dict[str, str] = {
    "ret_1": "Απόδοση 1 ημέρας",
    "ret_5": "Απόδοση 5 ημερών",
    "ret_20": "Απόδοση 20 ημερών",
    "ret_60": "Απόδοση 60 ημερών",
    "dist_sma20": "Απόσταση από μέσο όρο 20 ημερών",
    "dist_sma50": "Απόσταση από μέσο όρο 50 ημερών",
    "dist_sma200": "Απόσταση από μέσο όρο 200 ημερών",
    "rsi_14": "Δείκτης σχετικής ισχύος (RSI 14)",
    "atr_pct": "Εύρος κίνησης (ATR / τιμή)",
    "vol_20": "Μεταβλητότητα 20 ημερών",
    "vol_ratio": "Όγκος έναντι μέσου 20 ημερών",
    "value_z": "Τζίρος έναντι του συνηθισμένου (z‑score 60 ημερών)",
    "gap_1": "Χάσμα ανοίγματος (άνοιγμα / χθεσινό κλείσιμο)",
    "range_pos_252": "Θέση στο εύρος 52 εβδομάδων",
    "rel_mkt_20": "Απόδοση έναντι αγοράς (20 ημέρες)",
    "sector_rel_20": "Απόδοση έναντι κλάδου (20 ημέρες)",
    "mkt_ret_20": "Απόδοση αγοράς 20 ημερών",
    "mkt_vol_20": "Μεταβλητότητα αγοράς 20 ημερών",
    "mkt_trend": "Τάση αγοράς (έναντι μέσου όρου 100 ημερών)",
}

# Υπόδειξη μορφοποίησης για το UI: "pct" (κλάσμα → %), "num" (απλός αριθμός)
FEATURE_FMT: dict[str, str] = {c: "pct" for c in FEATURE_COLUMNS}
FEATURE_FMT.update({"rsi_14": "num", "vol_ratio": "num", "value_z": "num", "range_pos_252": "pct"})

# Στήλες του panel που δεν είναι είσοδοι του μοντέλου
EXTRA_COLUMNS: list[str] = ["close", "atr", "adv20_eur", "history_days", "sector", "mkt_drawdown_10", "regime"]

OTHER_SECTOR = market.OTHER_SECTOR
MKT_FFILL_LIMIT = 5             # μέγιστες συνεδριάσεις μεταφοράς της τελευταίας τιμής του δείκτη


def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    """ATR κατά Wilder (εκθετική εξομάλυνση με α = 1/n). Η πρώτη τιμή ορίζεται μετά από n ημέρες."""
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1, skipna=True)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI κατά Wilder, στην κλίμακα 0–100."""
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    loss = (-delta).clip(lower=0.0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 - 100.0 / (1.0 + gain / loss)
    out = out.where(loss > 0, np.where(gain > 0, 100.0, 50.0))
    return out.where(gain.notna() & loss.notna())


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    return out.replace([np.inf, -np.inf], np.nan)


def stock_features(df: pd.DataFrame, mkt: pd.DataFrame, atr_days: int) -> pd.DataFrame:
    """Χαρακτηριστικά μίας μετοχής (index = ημερομηνίες της μετοχής)."""
    o, h, l, c, v = (df[k].astype("float64") for k in ("open", "high", "low", "close", "volume"))
    out = pd.DataFrame(index=df.index)
    ret_1 = c.pct_change(fill_method=None)
    out["ret_1"] = ret_1
    for n in (5, 20, 60):
        out[f"ret_{n}"] = c / c.shift(n) - 1.0
    for n in (20, 50, 200):
        out[f"dist_sma{n}"] = c / c.rolling(n).mean() - 1.0
    out["rsi_14"] = rsi(c, 14)

    atr = wilder_atr(h, l, c, atr_days)
    out["atr_pct"] = _safe_div(atr, c)
    out["vol_20"] = ret_1.rolling(20).std()
    out["vol_ratio"] = _safe_div(v, v.rolling(20).mean())

    log_value = np.log1p((c * v).clip(lower=0.0))
    out["value_z"] = _safe_div(log_value - log_value.rolling(60).mean(), log_value.rolling(60).std())
    out["gap_1"] = o / c.shift(1) - 1.0
    lo252, hi252 = c.rolling(252).min(), c.rolling(252).max()
    span = hi252 - lo252
    out["range_pos_252"] = ((c - lo252) / span).where(span > 0, 0.5).where(span.notna())

    # πλαίσιο αγοράς: η πιο πρόσφατη τιμή με ημερομηνία ≤ t — το πολύ MKT_FFILL_LIMIT συνεδριάσεις πίσω,
    # ώστε ένας «παγωμένος» δείκτης να δίνει κενές τιμές (καμία πρόβλεψη) αντί για μήνες ίδιων τιμών
    m = (mkt.reindex(df.index, method="ffill", limit=MKT_FFILL_LIMIT) if len(mkt) else mkt.reindex(df.index))
    out["mkt_ret_20"] = m["mkt_ret_20"]
    out["mkt_vol_20"] = m["mkt_vol_20"]
    out["mkt_trend"] = m["mkt_trend"]
    out["rel_mkt_20"] = out["ret_20"] - out["mkt_ret_20"]

    # μη‑μοντελικές στήλες
    out["close"] = c
    out["atr"] = atr
    out["adv20_eur"] = (c * v).rolling(20).mean()
    out["history_days"] = np.arange(1, len(df) + 1, dtype="int64")
    out["mkt_drawdown_10"] = m["mkt_drawdown_10"]
    out["regime"] = m["regime"].astype("str").where(m["regime"].notna(), "neutral")
    return out


def build_panel(prices: dict[str, pd.DataFrame], index_df: pd.DataFrame | None,
                stocks: list[Stock], cfg) -> pd.DataFrame:
    """Panel με MultiIndex (date, ticker), όλα τα FEATURE_COLUMNS και τις στήλες
    close, atr, atr_pct, adv20_eur, history_days, sector, mkt_drawdown_10, regime.

    Γραμμές με NaN σε κάποιο χαρακτηριστικό κρατιούνται (τις χρειάζεται το UI)· το μοντέλο τις αγνοεί.
    """
    prices = {t: df for t, df in prices.items() if df is not None and len(df)}
    if index_df is None or not len(index_df):
        index_df = market.equal_weight_index(prices)
    mkt = market.market_features(index_df)
    sector_of = {s.ticker: s.sector for s in stocks}
    sec_wide = market.sector_strength(prices, stocks)["sector_rel_20"].unstack("ticker")

    frames = []
    for t in sorted(prices):
        f = stock_features(prices[t], mkt, cfg.barriers.atr_days)
        f["sector_rel_20"] = sec_wide[t].reindex(f.index) if t in sec_wide.columns else np.nan
        f["sector"] = sector_of.get(t, OTHER_SECTOR)
        f["ticker"] = t
        frames.append(f)

    cols = FEATURE_COLUMNS + [c for c in EXTRA_COLUMNS if c not in FEATURE_COLUMNS]
    if not frames:
        idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex([]), pd.Index([], dtype="str")], names=["date", "ticker"])
        return pd.DataFrame(index=idx, columns=cols)

    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.set_index("ticker", append=True).sort_index()
    panel = panel[cols]
    num = [c for c in cols if c not in ("sector", "regime", "history_days")]
    panel[num] = panel[num].replace([np.inf, -np.inf], np.nan)
    return panel
