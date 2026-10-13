"""Ετικέτες triple barrier και κοινός προσομοιωτής εξόδου."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from athex_orb import labels
from athex_orb.labels import LABEL_SL, LABEL_TO, LABEL_TP, simulate_exit

TP, SL = 105.0, 95.0     # όρια για είσοδο στο 100


def arrays(bars):
    """bars: λίστα από (open, high, low, close) → numpy arrays o, h, l, c."""
    a = np.asarray(bars, dtype=float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


QUIET = (100.0, 101.0, 99.0, 100.0)         # ημέρα χωρίς άγγιγμα ορίων


def run(bars, horizon=5, entry_idx=1, tp=TP, sl=SL):
    o, h, l, c = arrays(bars)
    return simulate_exit(o, h, l, c, entry_idx, o[entry_idx], tp, sl, horizon)


def test_tp_hit_intrabar():
    bars = [QUIET, QUIET, (101, 106, 100, 104), QUIET, QUIET, QUIET, QUIET]
    assert run(bars) == (LABEL_TP, 2, 105.0)


def test_tp_touch_exactly_counts():
    bars = [QUIET, QUIET, (101, 105, 100, 104), QUIET, QUIET, QUIET, QUIET]
    assert run(bars) == (LABEL_TP, 2, 105.0)


def test_sl_hit_intrabar():
    bars = [QUIET, QUIET, QUIET, (99, 100, 94, 96), QUIET, QUIET, QUIET]
    assert run(bars) == (LABEL_SL, 3, 95.0)


def test_both_in_same_bar_is_sl():
    bars = [QUIET, QUIET, (100, 106, 94, 100), QUIET, QUIET, QUIET, QUIET]
    assert run(bars) == (LABEL_SL, 2, 95.0)


def test_gap_through_tp_fills_at_open():
    # το άνοιγμα είναι ήδη πάνω από τον στόχο — ακόμη κι αν μετά πέφτει κάτω από το stop
    bars = [QUIET, QUIET, (107, 108, 94, 100), QUIET, QUIET, QUIET, QUIET]
    assert run(bars) == (LABEL_TP, 2, 107.0)


def test_gap_through_sl_fills_at_open():
    bars = [QUIET, QUIET, QUIET, (93, 106, 92, 104), QUIET, QUIET, QUIET]
    assert run(bars) == (LABEL_SL, 3, 93.0)


def test_gap_exactly_at_barrier():
    bars = [QUIET, QUIET, (105, 106, 104, 105), QUIET, QUIET, QUIET]
    assert run(bars) == (LABEL_TP, 2, 105.0)


def test_timeout_exits_at_close_of_last_bar():
    bars = [QUIET, QUIET, (100, 103, 97, 101), (101, 104, 98, 102), (102, 104, 96, 97),
            (97, 99, 96, 98.5), (98, 120, 80, 110)]
    # horizon 5 ⇒ ημέρες 1..5· η ημέρα 6 (με τεράστιο εύρος) δεν εξετάζεται
    assert run(bars, horizon=5) == (LABEL_TO, 5, 98.5)


def test_entry_day_counts_as_bar_one():
    # horizon = 1 ⇒ εξετάζεται μόνο η ημέρα εισόδου
    bars = [QUIET, (100, 103, 97, 102), (102, 110, 101, 109)]
    assert run(bars, horizon=1) == (LABEL_TO, 1, 102.0)
    # άγγιγμα στόχου μέσα στην ημέρα εισόδου
    bars = [QUIET, (100, 106, 99, 104), QUIET]
    assert run(bars, horizon=1) == (LABEL_TP, 1, 105.0)
    # άγγιγμα stop μέσα στην ημέρα εισόδου
    bars = [QUIET, (100, 101, 94, 96), QUIET]
    assert run(bars, horizon=3) == (LABEL_SL, 1, 95.0)


def test_entry_bar_open_is_not_a_gap_exit():
    # η είσοδος γίνεται στο άνοιγμα της ημέρας εισόδου· ο έλεγχος χάσματος ισχύει μόνο για τις επόμενες
    o, h, l, c = arrays([QUIET, (100, 101, 99, 100), QUIET, QUIET])
    assert simulate_exit(o, h, l, c, 1, 100.0, 100.0 * 1.2, 100.5, 2) == (LABEL_SL, 1, 100.5)


def test_hit_on_bar_after_horizon_is_timeout():
    bars = [QUIET, QUIET, QUIET, (100, 110, 99, 109)]
    assert run(bars, horizon=2) == (LABEL_TO, 2, 100.0)


def test_unresolved_when_data_ends():
    bars = [QUIET, QUIET, QUIET, QUIET]          # χρειάζονται ημέρες 1..5, υπάρχουν 1..3
    lab, idx, px = run(bars, horizon=5)
    assert (lab, idx) == (-1, -1) and math.isnan(px)


def test_resolved_before_data_end_even_if_horizon_longer():
    bars = [QUIET, QUIET, (101, 106, 100, 104)]
    assert run(bars, horizon=10) == (LABEL_TP, 2, 105.0)


def test_unresolved_when_no_entry_bar():
    o, h, l, c = arrays([QUIET, QUIET])
    lab, idx, px = simulate_exit(o, h, l, c, 2, 100.0, TP, SL, 3)
    assert (lab, idx) == (-1, -1) and math.isnan(px)


# ---------------------------------------------------------------- triple_barrier

def _hand_built():
    bars = [
        QUIET,                       # 0: απόφαση → είσοδος 1
        (100, 101, 99, 100),         # 1
        (100, 112, 99, 111),         # 2: στόχος (για απόφαση στο 0: tp=110)
        (111, 112, 110, 111),        # 3
        (111, 112, 110, 111),        # 4
        (111, 112, 110, 111),        # 5
        (111, 112, 110, 111),        # 6
    ]
    dates = pd.bdate_range("2024-01-01", periods=len(bars), name="date")
    o, h, l, c = arrays(bars)
    prices = {"TEST.AT": pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                                       "volume": np.full(len(bars), 1000.0)}, index=dates)}
    idx = pd.MultiIndex.from_arrays([dates, ["TEST.AT"] * len(bars)], names=["date", "ticker"])
    atr_pct = np.full(len(bars), 0.05)
    atr_pct[3] = np.nan
    panel = pd.DataFrame({"atr_pct": atr_pct}, index=idx)
    return prices, panel, dates


def test_triple_barrier_hand_built(cfg):
    cfg.barriers.k_tp, cfg.barriers.k_sl, cfg.barriers.horizon_days = 2.0, 1.0, 3
    prices, panel, dates = _hand_built()
    out = labels.triple_barrier(prices, panel, cfg)

    assert list(out.columns) == labels.LABEL_COLUMNS
    assert out.index.equals(panel.index)

    r0 = out.iloc[0]
    assert r0.label == LABEL_TP
    assert r0.entry_date == dates[1] and r0.entry_price == 100.0
    assert r0.tp_ret == pytest.approx(0.10) and r0.sl_ret == pytest.approx(0.05)
    assert r0.tp_price == pytest.approx(110.0) and r0.sl_price == pytest.approx(95.0)
    assert r0.exit_date == dates[2] and r0.exit_price == pytest.approx(110.0)
    assert r0.gross_ret == pytest.approx(0.10)
    assert r0.resolved_date == r0.exit_date
    assert r0.bars_held == 2

    # απόφαση στο 2: είσοδος στο 111 (ημ. 3), tp 122.1, sl 105.45 → λήξη χρόνου στο κλείσιμο της ημ. 5
    r2 = out.iloc[2]
    assert r2.label == LABEL_TO and r2.exit_date == dates[5] and r2.exit_price == 111.0 and r2.bars_held == 3

    # NaN ATR ⇒ καμία ετικέτα, κανένα όριο
    r3 = out.iloc[3]
    assert math.isnan(r3.label) and pd.isna(r3.entry_date) and math.isnan(r3.tp_ret)

    # απόφαση στο 4: είσοδος 5, χρειάζονται ημέρες 5..7 — λείπει η 7 ⇒ ανεπίλυτο, αλλά με είσοδο
    r4 = out.iloc[4]
    assert math.isnan(r4.label) and r4.entry_date == dates[5] and pd.isna(r4.resolved_date)
    assert math.isnan(r4.bars_held) and math.isnan(r4.gross_ret)

    # τελευταία ημέρα: δεν υπάρχει ημέρα εισόδου
    last = out.iloc[-1]
    assert math.isnan(last.label) and pd.isna(last.entry_date) and math.isnan(last.entry_price)
    assert math.isnan(last.tp_price)
    # …αλλά τα ποσοστά ορίων υπάρχουν (χρειάζονται για το σημερινό σήμα)
    assert last.tp_ret == pytest.approx(0.10) and last.sl_ret == pytest.approx(0.05)


def test_triple_barrier_matches_simulate_exit(synth, panel, labels_df, session_cfg):
    """Η διανυσματική εκδοχή δίνει ακριβώς ό,τι και ο βρόχος του simulate_exit (που χρησιμοποιεί το executor)."""
    prices, _ = synth
    b = session_cfg.barriers
    checked = 0
    for ticker, df in prices.items():
        o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
        sub = labels_df.xs(ticker, level="ticker")
        atr = panel.xs(ticker, level="ticker")["atr_pct"].reindex(df.index).to_numpy()
        for t in range(len(df) - 1):
            if not (np.isfinite(atr[t]) and atr[t] > 0):
                continue
            tp, sl = labels.barrier_prices(o[t + 1], atr[t], b.k_tp, b.k_sl)
            lab, ex, px = simulate_exit(o, h, l, c, t + 1, o[t + 1], tp, sl, b.horizon_days)
            row = sub.iloc[t]
            if lab == -1:
                assert math.isnan(row.label)
                continue
            assert row.label == lab
            assert row.exit_date == df.index[ex]
            assert row.exit_price == pytest.approx(px, rel=1e-12)
            assert row.bars_held == ex - (t + 1) + 1
            checked += 1
    assert checked > 5000


def test_labels_distribution_sane(labels_df):
    resolved = labels_df["label"].dropna()
    share = resolved.value_counts(normalize=True)
    assert set(share.index) == {0.0, 1.0, 2.0}
    # με k_tp=2, k_sl=1 σε τυχαίο περίπατο το stop έρχεται πρώτο συχνότερα από τον στόχο
    assert share[1.0] > share[0.0] > 0.1
    assert labels_df["bars_held"].dropna().between(1, 10).all()
    assert (labels_df["resolved_date"].dropna() > labels_df.index.get_level_values("date")[
        labels_df["resolved_date"].notna().to_numpy()]).all()
