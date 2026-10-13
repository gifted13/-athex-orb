"""«Χαρτογράφος»: πλαίσιο αγοράς (καθεστώς, τάση, πτώσεις) και δύναμη κλάδων.

Όλοι οι υπολογισμοί είναι αιτιακοί: η τιμή στην ημερομηνία t χρησιμοποιεί μόνο δεδομένα ≤ t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .universe import Stock

REGIME_LABELS_EL = {
    "bull": "Ανοδική αγορά",
    "bear": "Πτωτική αγορά",
    "neutral": "Ουδέτερη αγορά",
}
REGIME_TREND_BAND = 0.02        # |κλείσιμο/ΚΜΟ100 − 1| πάνω από αυτό ⇒ τάση
OTHER_SECTOR = "Άλλο"


def equal_weight_index(prices: dict[str, pd.DataFrame], base: float = 1000.0) -> pd.DataFrame:
    """Ισοσταθμικός δείκτης (open/high/low/close/volume) όταν λείπει ο επίσημος δείκτης.

    Κάθε ημέρα: μέσος όρος των ημερήσιων αποδόσεων των διαθέσιμων μετοχών. Τα open/high/low
    προκύπτουν από τους μέσους λόγους open/high/low προς το προηγούμενο κλείσιμο. Ο «όγκος» είναι
    ο συνολικός τζίρος σε €.
    """
    cols = ["open", "high", "low", "close", "volume"]
    frames = {t: df for t, df in prices.items() if df is not None and len(df)}
    if not frames:
        empty = pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="date"), dtype="float64")
        return empty

    rel = {k: {} for k in ("open", "high", "low", "close")}
    value = {}
    for t, df in frames.items():
        prev = df["close"].shift(1)
        for k in rel:
            r = (df[k] / prev).clip(lower=0.5, upper=2.0)   # προστασία από λανθασμένα δεδομένα
            rel[k][t] = r
        value[t] = df["close"] * df["volume"]

    mean_rel = {k: pd.DataFrame(v).sort_index().mean(axis=1) for k, v in rel.items()}
    close_ratio = mean_rel["close"].fillna(1.0)
    close = base * close_ratio.cumprod()
    prev_close = close.shift(1).fillna(base)
    out = pd.DataFrame({
        "open": prev_close * mean_rel["open"].fillna(1.0),
        "high": prev_close * mean_rel["high"].fillna(1.0),
        "low": prev_close * mean_rel["low"].fillna(1.0),
        "close": close,
        "volume": pd.DataFrame(value).sort_index().sum(axis=1, min_count=1).fillna(0.0),
    })
    out["high"] = out[["high", "open", "close"]].max(axis=1)
    out["low"] = out[["low", "open", "close"]].min(axis=1)
    out.index.name = "date"
    return out.astype("float64")


def market_features(index_df: pd.DataFrame) -> pd.DataFrame:
    """Χαρακτηριστικά αγοράς ανά ημερομηνία.

    Στήλες: mkt_ret_5, mkt_ret_20, mkt_vol_20, mkt_trend (κλείσιμο/ΚΜΟ100 − 1),
    mkt_drawdown_10 (κλείσιμο / μέγιστο κλείσιμο 10 ημερών − 1), regime, regime_el.
    """
    close = index_df["close"].astype("float64")
    ret = close.pct_change(fill_method=None)
    out = pd.DataFrame(index=index_df.index)
    out["mkt_ret_5"] = close / close.shift(5) - 1.0
    out["mkt_ret_20"] = close / close.shift(20) - 1.0
    out["mkt_vol_20"] = ret.rolling(20).std()
    out["mkt_trend"] = close / close.rolling(100).mean() - 1.0
    out["mkt_drawdown_10"] = close / close.rolling(10, min_periods=1).max() - 1.0

    trend = out["mkt_trend"].to_numpy()
    r20 = out["mkt_ret_20"].to_numpy()
    with np.errstate(invalid="ignore"):
        bull = (trend > REGIME_TREND_BAND) & (r20 > 0)
        bear = (trend < -REGIME_TREND_BAND) & (r20 < 0)
    regime = np.where(bull, "bull", np.where(bear, "bear", "neutral"))
    out["regime"] = pd.Series(regime, index=out.index, dtype="str")
    out["regime_el"] = out["regime"].map(REGIME_LABELS_EL)
    out.index.name = "date"
    return out


def sector_strength(prices: dict[str, pd.DataFrame], stocks: list[Stock]) -> pd.DataFrame:
    """`sector_rel_20` = απόδοση 20 ημερών της μετοχής − μέση απόδοση 20 ημερών του κλάδου της.

    Επιστρέφει DataFrame με MultiIndex (date, ticker). Αν ο κλάδος έχει μία μόνο μετοχή, η τιμή είναι 0.
    Μετοχές που δεν υπάρχουν στο `stocks` θεωρούνται κλάδου «Άλλο».
    """
    sector_of = {s.ticker: s.sector for s in stocks}
    ret20 = {}
    for t, df in prices.items():
        if df is None or not len(df):
            continue
        c = df["close"]
        ret20[t] = c / c.shift(20) - 1.0          # στις δικές της ημέρες διαπραγμάτευσης
    if not ret20:
        idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex([]), pd.Index([], dtype="str")], names=["date", "ticker"])
        return pd.DataFrame({"sector_rel_20": pd.Series([], dtype="float64")}, index=idx)

    wide = pd.DataFrame(ret20).sort_index()
    sectors = pd.Series({t: sector_of.get(t, OTHER_SECTOR) for t in wide.columns})
    rel = pd.DataFrame(index=wide.index, columns=wide.columns, dtype="float64")
    for _, cols in sectors.groupby(sectors).groups.items():
        cols = list(cols)
        block = wide[cols]
        rel[cols] = block.sub(block.mean(axis=1), axis=0)

    long = rel.stack(future_stack=True).dropna()
    long.index = long.index.set_names(["date", "ticker"])
    out = long.to_frame("sector_rel_20").sort_index()
    return out
