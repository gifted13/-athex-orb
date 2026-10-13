"""Χαρακτηριστικά (panel): πληρότητα, ορθότητα και — κυρίως — καμία «ματιά στο μέλλον»."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from athex_orb import features, labels, market, synthetic
from athex_orb.features import FEATURE_COLUMNS, FEATURE_LABELS_EL

EXTRA = ["close", "atr", "atr_pct", "adv20_eur", "history_days", "sector", "mkt_drawdown_10", "regime"]
TEXT_COLUMNS = ["sector", "regime"]


def _truncate(prices, index_df, T):
    p = {t: df.loc[:T] for t, df in prices.items()}
    return p, (index_df.loc[:T] if index_df is not None else None)


def _assert_same_rows(full: pd.DataFrame, part: pd.DataFrame, T):
    upto = full[full.index.get_level_values("date") <= T]
    assert upto.index.equals(part.index)
    num = [c for c in part.columns if c not in TEXT_COLUMNS]
    a = upto[num].to_numpy(dtype="float64")
    b = part[num].to_numpy(dtype="float64")
    bad = ~np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)
    if bad.any():
        cols = sorted({num[j] for j in np.nonzero(bad)[1]})
        pytest.fail(f"Ματιά στο μέλλον στις στήλες: {cols}")
    for col in TEXT_COLUMNS:
        assert (upto[col].to_numpy() == part[col].to_numpy()).all()


# ---------------------------------------------------------------- σχήμα

def test_columns_and_index(panel, synth):
    prices, _ = synth
    for col in FEATURE_COLUMNS + EXTRA:
        assert col in panel.columns, col
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS)) == 19
    assert panel.index.names == ["date", "ticker"]
    assert panel.index.is_monotonic_increasing and panel.index.is_unique
    assert len(panel) == sum(len(df) for df in prices.values())
    assert set(panel.index.get_level_values("ticker")) == set(prices)
    for col in FEATURE_COLUMNS:
        assert panel[col].dtype == np.float64, col
    assert set(panel["regime"].unique()) <= {"bull", "bear", "neutral"}


def test_feature_labels_are_greek():
    assert set(FEATURE_LABELS_EL) == set(FEATURE_COLUMNS)
    for label in FEATURE_LABELS_EL.values():
        assert any("Ͱ" <= ch <= "Ͽ" for ch in label), label


def test_features_mostly_available_after_warmup(panel):
    late = panel[panel["history_days"] > 260]
    assert late[FEATURE_COLUMNS].notna().all(axis=1).mean() > 0.99
    assert not np.isinf(panel[FEATURE_COLUMNS].to_numpy()).any()


# ---------------------------------------------------------------- καμία ματιά στο μέλλον

@pytest.mark.parametrize("T", ["2020-03-16", "2021-06-30", "2022-11-15"])
@pytest.mark.parametrize("with_index", [True, False])
def test_no_lookahead_truncation(synth, small_universe, session_cfg, T, with_index):
    prices, index_df = synth
    index_df = index_df if with_index else None
    T = pd.Timestamp(T)
    full = features.build_panel(prices, index_df, small_universe, session_cfg)
    part = features.build_panel(*_truncate(prices, index_df, T), small_universe, session_cfg)
    _assert_same_rows(full, part, T)


def test_no_lookahead_poisoned_future(synth, small_universe, session_cfg):
    """Αν «χαλάσουμε» εντελώς το μέλλον, το παρελθόν πρέπει να μείνει ίδιο."""
    prices, index_df = synth
    T = pd.Timestamp("2021-09-01")
    rng = np.random.default_rng(0)
    poisoned = {}
    for t, df in prices.items():
        d = df.copy()
        fut = d.index > T
        k = rng.uniform(0.2, 5.0)
        d.loc[fut, ["open", "high", "low", "close"]] *= k
        d.loc[fut, "volume"] *= 50.0
        poisoned[t] = d
    pidx = index_df.copy()
    pidx.loc[pidx.index > T, ["open", "high", "low", "close"]] *= 0.3
    full = features.build_panel(prices, index_df, small_universe, session_cfg)
    bad = features.build_panel(poisoned, pidx, small_universe, session_cfg)
    _assert_same_rows(full, bad[bad.index.get_level_values("date") <= T], T)


# ---------------------------------------------------------------- ορθότητα

def _wilder_loop(h, l, c, n):
    tr = np.empty(len(c))
    tr[0] = h[0] - l[0]
    for i in range(1, len(c)):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = np.full(len(c), np.nan)
    avg = tr[0]
    for i in range(1, len(c)):
        avg = avg + (tr[i] - avg) / n
        if i >= n - 1:
            atr[i] = avg
    return atr


def test_atr_matches_wilder_loop(panel, synth, session_cfg):
    prices, _ = synth
    n = session_cfg.barriers.atr_days
    for ticker in ("ETE.AT", "BELA.AT"):
        df = prices[ticker]
        sub = panel.xs(ticker, level="ticker")
        expected = _wilder_loop(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(), n)
        np.testing.assert_allclose(sub["atr"].to_numpy(), expected, rtol=1e-10, equal_nan=True)
        np.testing.assert_allclose(sub["atr_pct"].to_numpy(), expected / df["close"].to_numpy(),
                                   rtol=1e-10, equal_nan=True)


def test_atr_and_adv_sanity(panel, synth):
    prices, _ = synth
    atr = panel["atr"].dropna()
    assert (atr > 0).all()
    pct = panel["atr_pct"].dropna()
    assert pct.between(0.002, 0.25).all()
    assert 0.01 < pct.median() < 0.08
    for ticker, df in prices.items():
        sub = panel.xs(ticker, level="ticker")
        adv = (df["close"] * df["volume"]).rolling(20).mean()
        np.testing.assert_allclose(sub["adv20_eur"].to_numpy(), adv.to_numpy(), rtol=1e-12, equal_nan=True)
        assert sub["adv20_eur"].iloc[:19].isna().all() and sub["adv20_eur"].iloc[19:].notna().all()
        assert (sub["history_days"].to_numpy() == np.arange(1, len(df) + 1)).all()
        assert (sub["close"].to_numpy() == df["close"].to_numpy()).all()


def test_simple_features_by_definition(panel, synth):
    prices, _ = synth
    df = prices["OPAP.AT"]
    sub = panel.xs("OPAP.AT", level="ticker")
    c = df["close"]
    np.testing.assert_allclose(sub["ret_5"], c / c.shift(5) - 1, equal_nan=True)
    np.testing.assert_allclose(sub["dist_sma50"], c / c.rolling(50).mean() - 1, equal_nan=True)
    np.testing.assert_allclose(sub["gap_1"], df["open"] / c.shift(1) - 1, equal_nan=True)
    np.testing.assert_allclose(sub["vol_ratio"], df["volume"] / df["volume"].rolling(20).mean(), equal_nan=True)
    assert sub["rsi_14"].dropna().between(0, 100).all()
    assert sub["range_pos_252"].dropna().between(0, 1).all()
    assert sub["range_pos_252"].iloc[:251].isna().all()
    np.testing.assert_allclose(sub["rel_mkt_20"], sub["ret_20"] - sub["mkt_ret_20"], equal_nan=True)


def test_sector_relative_strength_sums_to_zero(panel):
    banks = panel[panel["sector"] == "Τράπεζες"]["sector_rel_20"].dropna()
    by_date = banks.groupby(level="date")
    sums = by_date.sum()[by_date.count() == 3]
    assert len(sums) > 500
    assert np.allclose(sums.to_numpy(), 0.0, atol=1e-12)
    # μοναδική μετοχή κλάδου ⇒ 0
    solo = panel.xs("OPAP.AT", level="ticker")["sector_rel_20"].dropna()
    assert (solo == 0).all()


def test_market_features_and_regime(synth):
    _, index_df = synth
    mf = market.market_features(index_df)
    for col in ("mkt_ret_5", "mkt_ret_20", "mkt_vol_20", "mkt_trend", "mkt_drawdown_10", "regime", "regime_el"):
        assert col in mf.columns
    assert (mf["mkt_drawdown_10"] <= 1e-12).all()
    assert set(mf["regime"]) <= {"bull", "bear", "neutral"}
    assert mf["regime_el"].map(lambda s: isinstance(s, str) and len(s) > 0).all()
    bull = mf[mf["regime"] == "bull"]
    assert (bull["mkt_trend"] > 0).all() and (bull["mkt_ret_20"] > 0).all()


def test_equal_weight_index_shape(synth):
    prices, _ = synth
    ew = market.equal_weight_index(prices)
    assert list(ew.columns) == ["open", "high", "low", "close", "volume"]
    assert ew.index.is_monotonic_increasing and ew.index.is_unique
    assert (ew["high"] >= ew[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (ew["low"] <= ew[["open", "close"]].min(axis=1) + 1e-9).all()
    rets = pd.DataFrame({t: df["close"].pct_change() for t, df in prices.items()}).mean(axis=1)
    np.testing.assert_allclose(ew["close"].pct_change().iloc[1:], rets.iloc[1:], rtol=1e-9)


def test_panel_without_index_uses_equal_weight(synth, small_universe, session_cfg):
    prices, _ = synth
    p = features.build_panel(prices, None, small_universe, session_cfg)
    assert p["mkt_ret_20"].notna().sum() > 0.9 * len(p)


def test_unknown_ticker_gets_other_sector(synth, small_universe, session_cfg):
    prices, index_df = synth
    p = features.build_panel({"ZZZ.AT": prices["ETE.AT"]}, index_df, small_universe, session_cfg)
    assert (p["sector"] == "Άλλο").all()


def test_full_universe_speed(full_universe, session_cfg):
    """31 μετοχές × 8 χρόνια: panel + ετικέτες σε λίγα δευτερόλεπτα."""
    prices, index_df = synthetic.generate_universe(full_universe, start="2017-01-02", end="2024-12-31", seed=3)
    t0 = time.perf_counter()
    panel = features.build_panel(prices, index_df, full_universe, session_cfg)
    lab = labels.triple_barrier(prices, panel, session_cfg)
    elapsed = time.perf_counter() - t0
    assert len(panel) > 60_000 and len(lab) == len(panel)
    assert elapsed < 10.0, f"πολύ αργό: {elapsed:.1f}s"


def test_stale_market_index_is_not_carried_forward(synth, small_universe, session_cfg):
    """Ένας δείκτης που σταμάτησε δεν «παγώνει» τα χαρακτηριστικά αγοράς: μετά από λίγες συνεδριάσεις γίνονται
    κενά (καμία πρόβλεψη) αντί για μήνες ίδιων τιμών."""
    prices, index_df = synth
    cut = index_df.index[-60]
    panel = features.build_panel(prices, index_df[index_df.index <= cut], small_universe, session_cfg)
    t = small_universe[0].ticker
    f = panel.xs(t, level="ticker")
    after = f[f.index > cut]
    assert after["mkt_ret_20"].iloc[:features.MKT_FFILL_LIMIT].notna().all()
    assert after["mkt_ret_20"].iloc[features.MKT_FFILL_LIMIT:].isna().all()
