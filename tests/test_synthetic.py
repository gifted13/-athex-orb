"""Συνθετικά δεδομένα: συνέπεια OHLC, ντετερμινισμός, ρευστότητα και φυτεμένο σήμα."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from athex_orb import features, labels, synthetic
from athex_orb.features import FEATURE_COLUMNS

SYNTH_START, SYNTH_END, SYNTH_SEED = "2019-01-02", "2022-12-30", 11     # όπως στο conftest


def test_conftest_range_matches(synth_range):
    assert synth_range == (SYNTH_START, SYNTH_END, SYNTH_SEED)


def _check_frame(df: pd.DataFrame):
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert (df.dtypes == np.float64).all()
    assert isinstance(df.index, pd.DatetimeIndex) and df.index.name == "date"
    assert df.index.tz is None and df.index.is_monotonic_increasing and df.index.is_unique
    assert (df.index == df.index.normalize()).all()
    assert (df.index.dayofweek < 5).all()
    assert np.isfinite(df.to_numpy()).all()
    assert (df[["open", "high", "low", "close"]] > 0).all().all()
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert (df["volume"] >= 1).all()


def test_ohlc_consistency(synth):
    prices, index_df = synth
    assert len(prices) == 8
    for df in prices.values():
        _check_frame(df)
        assert df.index.equals(index_df.index)
    _check_frame(index_df)
    assert index_df.index[0] == pd.Timestamp(SYNTH_START) and index_df.index[-1] == pd.Timestamp(SYNTH_END)


def test_occasional_gaps(synth):
    prices, _ = synth
    gaps = pd.concat([(df["open"] / df["close"].shift(1) - 1).abs() for df in prices.values()])
    assert (gaps > 0.03).mean() > 0.001                      # υπάρχουν χάσματα
    assert gaps.median() < 0.01                               # αλλά δεν είναι ο κανόνας


def test_deterministic_by_seed(small_universe):
    a, ia = synthetic.generate_universe(small_universe[:3], start="2020-01-01", end="2021-01-01", seed=5)
    b, ib = synthetic.generate_universe(small_universe[:3], start="2020-01-01", end="2021-01-01", seed=5)
    c, _ = synthetic.generate_universe(small_universe[:3], start="2020-01-01", end="2021-01-01", seed=6)
    for t in a:
        pd.testing.assert_frame_equal(a[t], b[t])
        assert not np.allclose(a[t]["close"].to_numpy(), c[t]["close"].to_numpy())
    pd.testing.assert_frame_equal(ia, ib)


def test_stock_independent_of_universe_and_end(synth, small_universe):
    prices, _ = synth
    stock = small_universe[4]
    alone, _ = synthetic.generate_universe([stock], start=SYNTH_START, end=SYNTH_END, seed=SYNTH_SEED)
    pd.testing.assert_frame_equal(alone[stock.ticker], prices[stock.ticker])
    longer, _ = synthetic.generate_universe([stock], start=SYNTH_START, end="2023-06-30", seed=SYNTH_SEED)
    pd.testing.assert_frame_equal(longer[stock.ticker].loc[:SYNTH_END], prices[stock.ticker])


def test_default_end_is_previous_business_day():
    end = synthetic.default_end()
    today = pd.Timestamp.today().normalize()
    assert end < today and end.dayofweek < 5
    # καμία εργάσιμη ανάμεσα στο end και στο σήμερα
    assert len(pd.bdate_range(end + pd.Timedelta(days=1), today - pd.Timedelta(days=1))) == 0


def test_traded_value_range(full_universe):
    prices, _ = synthetic.generate_universe(full_universe, start="2020-01-01", end="2023-12-29", seed=7)
    med = pd.Series({t: (df["close"] * df["volume"]).median() for t, df in prices.items()})
    assert med.between(3e4, 5e7).all(), med.sort_values()
    assert med.min() < 4e5                 # υπάρχουν μη ρευστές μετοχές…
    assert med.max() > 1e7                 # …και πολύ ρευστές
    assert med.max() / med.min() > 50
    # οι τράπεζες είναι κατά κανόνα πιο ρευστές από τις μικρές εταιρείες
    banks = [s.ticker for s in full_universe if s.sector == "Τράπεζες"]
    assert med[banks].median() > med.median()


def _tp_auc(prices, index_df, stocks, cfg):
    panel = features.build_panel(prices, index_df, stocks, cfg)
    lab = labels.triple_barrier(prices, panel, cfg)
    df = panel.join(lab).dropna(subset=FEATURE_COLUMNS + ["label"])
    dates = df.index.get_level_values("date")
    cut = dates.unique()[int(0.6 * dates.nunique())]
    train, test = df[dates < cut], df[dates >= cut]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(train[FEATURE_COLUMNS], train["label"] == labels.LABEL_TP)
    auc = roc_auc_score(test["label"] == labels.LABEL_TP, model.predict_proba(test[FEATURE_COLUMNS])[:, 1])
    coefs = dict(zip(FEATURE_COLUMNS, model[-1].coef_[0]))
    return auc, coefs


def test_planted_signal_is_learnable(synth, synth_planted, small_universe, session_cfg):
    auc_none, _ = _tp_auc(*synth, small_universe, session_cfg)
    auc_planted, coefs = _tp_auc(*synth_planted, small_universe, session_cfg)
    assert auc_planted > 0.6, auc_planted
    assert auc_planted > auc_none + 0.05, (auc_none, auc_planted)
    # σήμα επαναφοράς: μετά από πτώση 5 ημερών ο στόχος είναι πιθανότερος
    assert coefs["ret_5"] < 0 or coefs["rsi_14"] < 0 or coefs["dist_sma20"] < 0


def test_planted_signal_zero_changes_nothing(small_universe):
    a, _ = synthetic.generate_universe(small_universe[:2], start="2020-01-01", end="2020-12-31", seed=1)
    b, _ = synthetic.generate_universe(small_universe[:2], start="2020-01-01", end="2020-12-31", seed=1,
                                       planted_signal=0.0)
    c, _ = synthetic.generate_universe(small_universe[:2], start="2020-01-01", end="2020-12-31", seed=1,
                                       planted_signal=1.0)
    t = small_universe[0].ticker
    pd.testing.assert_frame_equal(a[t], b[t])
    assert not np.allclose(a[t]["close"], c[t]["close"])


def test_empty_range_raises(small_universe):
    with pytest.raises(ValueError):
        synthetic.generate_universe(small_universe[:1], start="2020-01-04", end="2020-01-05")
