"""«Ποσοτικός»: κόστη εκτέλεσης και αναμενόμενη αξία."""
from __future__ import annotations

import math

import numpy as np
import pytest

from athex_orb import quant


@pytest.fixture
def qcfg(cfg):
    c = cfg.costs
    c.commission_pct = 0.001
    c.commission_min_eur = 5.0
    c.exchange_fees_pct = 0.0003
    c.sell_tax_pct = 0.002
    c.spread_bps_base = 10
    c.impact_coef = 0.1
    return cfg


def test_min_commission_applies_to_small_orders(qcfg):
    small = quant.side_cost_pct(1000.0, 1e7, 10.0, qcfg, "buy")
    big = quant.side_cost_pct(20000.0, 1e7, 10.0, qcfg, "buy")
    assert small["commission"] == pytest.approx(5.0 / 1000.0)        # 0,5% > 0,1%
    assert big["commission"] == pytest.approx(0.001)                 # 5/20000 = 0,025% < 0,1%
    # ακριβώς στο όριο: 5 € / 0,1% = 5.000 €
    assert quant.side_cost_pct(5000.0, 1e7, 10.0, qcfg, "buy")["commission"] == pytest.approx(0.001)


def test_tax_only_on_sell(qcfg):
    buy = quant.side_cost_pct(2000.0, 1e6, 10.0, qcfg, "buy")
    sell = quant.side_cost_pct(2000.0, 1e6, 10.0, qcfg, "sell")
    assert buy["tax"] == 0.0
    assert sell["tax"] == pytest.approx(0.002)
    for k in ("commission", "fees", "spread", "impact"):
        assert buy[k] == pytest.approx(sell[k])
    assert sell["total"] - buy["total"] == pytest.approx(0.002)


def test_total_is_sum_of_parts_and_round_trip_is_buy_plus_sell(qcfg):
    for side in ("buy", "sell"):
        c = quant.side_cost_pct(1500.0, 4e5, 3.0, qcfg, side, vol=0.03)
        assert c["total"] == pytest.approx(sum(c[k] for k in ("commission", "fees", "tax", "spread", "impact")))
    rt = quant.round_trip_cost_pct(1500.0, 4e5, 3.0, qcfg, vol=0.03)
    b = quant.side_cost_pct(1500.0, 4e5, 3.0, qcfg, "buy", vol=0.03)
    s = quant.side_cost_pct(1500.0, 4e5, 3.0, qcfg, "sell", vol=0.03)
    assert set(rt) == set(quant.COST_KEYS)
    for k in quant.COST_KEYS:
        assert rt[k] == pytest.approx(b[k] + s[k])


def test_spread_formula_and_caps(qcfg):
    base = qcfg.costs.spread_bps_base / 1e4
    assert quant.spread_pct(5e6, qcfg) == pytest.approx(base)          # ακριβώς στο σημείο αναφοράς
    assert quant.spread_pct(5e8, qcfg) == pytest.approx(base)          # κάτω όριο ×1
    assert quant.spread_pct(1.25e6, qcfg) == pytest.approx(base * 2)   # √(5e6/1.25e6) = 2
    assert quant.spread_pct(1e3, qcfg) == pytest.approx(base * 8)      # άνω όριο ×8
    assert quant.spread_pct(float("nan"), qcfg) == pytest.approx(base * 8)   # άγνωστη ρευστότητα = ακριβή
    assert quant.spread_pct(0.0, qcfg) == pytest.approx(base * 8)


def test_spread_and_impact_monotonic_in_illiquidity(qcfg):
    advs = [3e7, 1e7, 3e6, 1e6, 3e5, 1e5, 3e4]
    spreads = [quant.spread_pct(a, qcfg) for a in advs]
    impacts = [quant.impact_pct(2000.0, a, qcfg) for a in advs]
    totals = [quant.side_cost_pct(2000.0, a, 5.0, qcfg, "buy")["total"] for a in advs]
    assert all(b >= a for a, b in zip(spreads, spreads[1:]))
    assert spreads[-1] > spreads[0]
    assert all(b > a for a, b in zip(impacts, impacts[1:]))
    assert all(b > a for a, b in zip(totals, totals[1:]))


def test_impact_monotonic_in_size_and_formula(qcfg):
    sizes = [500.0, 1000.0, 2000.0, 5000.0, 20000.0]
    imp = [quant.impact_pct(s, 1e6, qcfg) for s in sizes]
    assert all(b > a for a, b in zip(imp, imp[1:]))
    # impact = coef × √(εντολή / τζίρος) × vol (προεπιλογή vol = 0,02)
    assert quant.impact_pct(10000.0, 1e6, qcfg) == pytest.approx(0.1 * math.sqrt(0.01) * 0.02)
    assert quant.impact_pct(10000.0, 1e6, qcfg, vol=0.04) == pytest.approx(0.1 * math.sqrt(0.01) * 0.04)
    assert quant.impact_pct(10000.0, 1e6, qcfg, vol=float("nan")) == pytest.approx(0.1 * 0.1 * 0.02)
    # το κόστος σε € αυξάνει πάντα με το ποσό (απαραίτητο για τον έλεγχο μετρητών στο backtest)
    eur = [s * quant.side_cost_pct(s, 2e5, 5.0, qcfg, "buy")["total"] for s in np.linspace(50, 30000, 200)]
    assert all(b >= a - 1e-12 for a, b in zip(eur, eur[1:]))


def test_zero_or_invalid_order(qcfg):
    c = quant.side_cost_pct(0.0, 1e6, 5.0, qcfg, "buy")
    assert c["commission"] == pytest.approx(qcfg.costs.commission_pct)
    assert c["impact"] == 0.0
    assert quant.side_cost_eur(0.0, 1e6, 5.0, qcfg, "buy") == 0.0
    with pytest.raises(ValueError):
        quant.side_cost_pct(1000.0, 1e6, 5.0, qcfg, "short")


def test_costs_always_positive(qcfg):
    rng = np.random.default_rng(0)
    for _ in range(200):
        order = float(rng.uniform(1, 50000))
        adv = float(10 ** rng.uniform(3, 8))
        for side in ("buy", "sell"):
            assert quant.side_cost_pct(order, adv, 5.0, qcfg, side)["total"] > 0


def test_vectorized_and_fast_versions_match_scalar(qcfg):
    rng = np.random.default_rng(1)
    n = 300
    order = rng.uniform(1, 30000, n)
    adv = 10 ** rng.uniform(3, 8, n)
    vol = rng.uniform(0.005, 0.06, n)
    order[:5] = [0.0, np.nan, -10.0, 100.0, 5000.0]
    adv[5:10] = [np.nan, 0.0, -1.0, 1e3, 1e9]
    vol[10:13] = [np.nan, 0.0, -0.1]
    for side in ("buy", "sell"):
        v = quant.side_cost_pct_vec(order, adv, qcfg, side, vol=vol)
        for i in range(n):
            s = quant.side_cost_pct(order[i], adv[i], 5.0, qcfg, side, vol=vol[i])
            for k in quant.COST_KEYS:
                assert v[k][i] == pytest.approx(s[k], rel=1e-12, abs=1e-15), (i, k)
            assert quant.side_cost_total(float(order[i]), float(adv[i]), qcfg, side, float(vol[i])) == \
                pytest.approx(s["total"], rel=1e-12)
    rt = quant.round_trip_cost_pct_vec(order, adv, qcfg, vol=vol)
    for i in range(0, n, 17):
        assert rt["total"][i] == pytest.approx(quant.round_trip_cost_pct(order[i], adv[i], 5.0, qcfg,
                                                                         vol=vol[i])["total"])


def test_expected_value_formula():
    ev = quant.expected_value(0.4, 0.45, 0.15, 0.06, 0.03, 0.005, 0.008)
    gross = 0.4 * 0.06 - 0.45 * 0.03 + 0.15 * 0.005
    assert ev["ev_gross"] == pytest.approx(gross)
    assert ev["ev_net"] == pytest.approx(gross - 0.008)
    # δεκτό και το dict του round_trip_cost_pct
    ev2 = quant.expected_value(0.4, 0.45, 0.15, 0.06, 0.03, 0.005, {"total": 0.008})
    assert ev2 == ev
    assert all(isinstance(v, float) for v in ev.values())


@pytest.mark.parametrize("p_to,tm,cost", [(0.15, 0.005, 0.008), (0.0, 0.0, 0.0), (0.3, -0.01, 0.02), (0.6, 0.02, 0.0)])
def test_breakeven_p_tp_zeroes_ev(p_to, tm, cost):
    tp_ret, sl_ret = 0.06, 0.03
    be = quant.expected_value(0.3, 1 - 0.3 - p_to, p_to, tp_ret, sl_ret, tm, cost)["breakeven_p_tp"]
    ev_be = quant.expected_value(be, 1 - p_to - be, p_to, tp_ret, sl_ret, tm, cost)["ev_net"]
    assert ev_be == pytest.approx(0.0, abs=1e-12)
    # πάνω από το νεκρό σημείο το EV είναι θετικό, κάτω αρνητικό
    assert quant.expected_value(be + 0.01, 1 - p_to - be - 0.01, p_to, tp_ret, sl_ret, tm, cost)["ev_net"] > 0
    assert quant.expected_value(be - 0.01, 1 - p_to - be + 0.01, p_to, tp_ret, sl_ret, tm, cost)["ev_net"] < 0


def test_breakeven_known_value_and_degenerate():
    # χωρίς κόστος και χωρίς λήξη, με στόχο 2× το stop: νεκρό σημείο = 1/3
    assert quant.expected_value(0.5, 0.5, 0.0, 0.02, 0.01, 0.0, 0.0)["breakeven_p_tp"] == pytest.approx(1 / 3)
    assert math.isnan(quant.expected_value(0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.01)["breakeven_p_tp"])


def test_expected_value_vectorized():
    p_tp = np.array([0.2, 0.3, 0.5])
    p_to = np.array([0.1, 0.2, 0.0])
    p_sl = 1 - p_tp - p_to
    r = quant.expected_value(p_tp, p_sl, p_to, 0.05, 0.025, 0.0, 0.01)
    assert r["ev_net"].shape == (3,)
    for i in range(3):
        s = quant.expected_value(float(p_tp[i]), float(p_sl[i]), float(p_to[i]), 0.05, 0.025, 0.0, 0.01)
        assert r["ev_net"][i] == pytest.approx(s["ev_net"])
        assert r["breakeven_p_tp"][i] == pytest.approx(s["breakeven_p_tp"])
