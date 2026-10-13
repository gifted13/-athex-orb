"""«Φύλακας»: έλεγχοι κινδύνου και μέγεθος θέσης."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from athex_orb import guard
from athex_orb.features import FEATURE_COLUMNS


def good_row(**over) -> pd.Series:
    """Γραμμή panel που περνά όλους τους ελέγχους με τις προεπιλεγμένες ρυθμίσεις."""
    d = {c: 0.01 for c in FEATURE_COLUMNS}
    d.update({"close": 10.0, "atr_pct": 0.03, "adv20_eur": 2e6, "history_days": 600, "mkt_drawdown_10": -0.02,
              "regime": "neutral", "sector": "Τράπεζες"})
    d.update(over)
    return pd.Series(d)


def by_id(checks):
    return {c.id: c for c in checks}


def test_good_row_passes_everything(cfg):
    checks = guard.stock_checks(good_row(), cfg)
    assert [c.id for c in checks] == guard.STOCK_CHECK_IDS
    assert all(c.passed for c in checks)
    assert guard.passed(checks)
    assert guard.first_failure(checks) is None
    for c in checks:
        assert c.label_el and c.detail_el
        assert set(c.to_dict()) == {"id", "passed", "label_el", "detail_el", "blocking"}


@pytest.mark.parametrize("field,bad,good,check_id", [
    ("adv20_eur", 299_999.0, 300_000.0, "liquidity"),
    ("close", 0.49, 0.50, "price"),
    ("atr_pct", 0.081, 0.08, "volatility"),
    ("history_days", 259, 260, "history"),
    ("mkt_drawdown_10", -0.08, -0.0799, "market"),
])
def test_each_stock_check_pass_fail(cfg, field, bad, good, check_id):
    fail = by_id(guard.stock_checks(good_row(**{field: bad}), cfg))
    ok = by_id(guard.stock_checks(good_row(**{field: good}), cfg))
    assert not fail[check_id].passed and fail[check_id].blocking
    assert ok[check_id].passed
    assert not guard.passed(fail.values())
    assert guard.first_failure(fail.values()).id == check_id
    # οι υπόλοιποι έλεγχοι δεν επηρεάζονται
    assert all(c.passed for cid, c in fail.items() if cid != check_id)


def test_missing_values_fail_except_market(cfg):
    for field, cid in [("adv20_eur", "liquidity"), ("close", "price"), ("atr_pct", "volatility"),
                       ("history_days", "history")]:
        assert not by_id(guard.stock_checks(good_row(**{field: np.nan}), cfg))[cid].passed
    # χωρίς δεδομένα δείκτη ο έλεγχος αγοράς παραλείπεται (περνά) — με εξήγηση
    m = by_id(guard.stock_checks(good_row(mkt_drawdown_10=np.nan), cfg))["market"]
    assert m.passed and "παραλείπεται" in m.detail_el


def test_features_check(cfg):
    row = good_row(dist_sma200=np.nan, ret_60=np.nan)
    c = by_id(guard.stock_checks(row, cfg))["features"]
    assert not c.passed
    assert "2" in c.detail_el
    row = good_row().drop("rsi_14")            # στήλη που λείπει εντελώς
    assert not by_id(guard.stock_checks(row, cfg))["features"].passed


def test_regime_is_informational_only(cfg):
    checks = guard.stock_checks(good_row(regime="bear"), cfg)
    reg = by_id(checks)["regime"]
    assert not reg.passed and not reg.blocking
    assert guard.passed(checks)


def test_thresholds_follow_config(cfg):
    cfg.guard.min_adv_eur = 5e6
    assert not by_id(guard.stock_checks(good_row(), cfg))["liquidity"].passed
    cfg.guard.min_adv_eur = 1e5
    assert by_id(guard.stock_checks(good_row(adv20_eur=2e5), cfg))["liquidity"].passed


def test_vectorized_checks_match_scalar(cfg, panel):
    sample = panel.iloc[np.linspace(0, len(panel) - 1, 400).astype(int)]
    vec = guard.stock_checks_vec(sample, cfg)
    mask = guard.stock_pass_mask(sample, cfg)
    assert set(vec) == set(guard.STOCK_CHECK_IDS)
    for i in range(len(sample)):
        checks = guard.stock_checks(sample.iloc[i], cfg)
        for c in checks:
            assert bool(vec[c.id][i]) == c.passed, (i, c.id)
        assert bool(mask[i]) == guard.passed(checks)
    assert mask.any() and not mask.all()         # το δείγμα έχει και τις δύο περιπτώσεις


def test_decision_checks(cfg):
    cfg.guard.min_ev_net = 0.003
    cfg.guard.min_p_tp = 0.3
    d = by_id(guard.decision_checks({"ev_net": 0.004}, 0.35, cfg))
    assert d["ev"].passed and d["p_tp"].passed
    d = by_id(guard.decision_checks({"ev_net": 0.0029}, 0.35, cfg))
    assert not d["ev"].passed and d["p_tp"].passed
    d = by_id(guard.decision_checks({"ev_net": 0.01}, 0.29, cfg))
    assert d["ev"].passed and not d["p_tp"].passed
    d = by_id(guard.decision_checks({"ev_net": 0.003}, 0.3, cfg))       # ακριβώς στο όριο ⇒ περνά
    assert d["ev"].passed and d["p_tp"].passed
    d = by_id(guard.decision_checks({"ev_net": float("nan")}, float("nan"), cfg))
    assert not d["ev"].passed and not d["p_tp"].passed
    assert "+0,40%" in guard.decision_checks({"ev_net": 0.004}, 0.35, cfg)[0].detail_el


def test_portfolio_checks(cfg):
    cfg.portfolio.max_positions = 3
    cfg.portfolio.max_per_sector = 2
    pos = [{"ticker": "ETE.AT", "sector": "Τράπεζες"}, {"ticker": "PPC.AT", "sector": "Ενέργεια"}]
    c = by_id(guard.portfolio_checks("OPAP.AT", "Ψυχαγωγία", pos, cfg))
    assert all(x.passed for x in c.values())
    # ήδη ανοιχτή θέση
    assert not by_id(guard.portfolio_checks("ETE.AT", "Τράπεζες", pos, cfg))["not_held"].passed
    # όριο κλάδου
    pos2 = pos + [{"ticker": "EUROB.AT", "sector": "Τράπεζες"}]
    c = by_id(guard.portfolio_checks("TPEIR.AT", "Τράπεζες", pos2, cfg))
    assert not c["sector"].passed
    # μέγιστες θέσεις (3 ανοιχτές, όριο 3)
    assert not c["max_positions"].passed
    c = by_id(guard.portfolio_checks("HTO.AT", "Τηλεπικοινωνίες", pos2, cfg))
    assert not c["max_positions"].passed and c["sector"].passed and c["not_held"].passed
    assert not guard.passed(c.values())
    # άδειο χαρτοφυλάκιο
    assert guard.passed(guard.portfolio_checks("HTO.AT", "Τηλεπικοινωνίες", [], cfg))


def test_position_size_caps(cfg):
    cfg.portfolio.position_eur_max = 2000
    cfg.portfolio.max_adv_fraction = 0.01
    assert guard.position_size_eur(10_000, 1e7, cfg) == 2000            # όριο ανά θέση
    assert guard.position_size_eur(1_500, 1e7, cfg) == 1500             # διαθέσιμα
    assert guard.position_size_eur(10_000, 120_000, cfg) == pytest.approx(1200)   # 1% του τζίρου
    assert guard.position_size_eur(10_000, float("nan"), cfg) == 0.0
    assert guard.position_size_eur(-50, 1e7, cfg) == 0.0
    assert guard.position_size_eur(float("nan"), 1e7, cfg) == 0.0
    assert guard.MIN_POSITION_EUR == 300.0


def test_greek_formatting_helpers():
    assert guard.fmt_num(1234.5, 1) == "1.234,5"
    assert guard.fmt_eur(1_250_000) == "1,25 εκ. €"
    assert guard.fmt_eur(300_000) == "300 χιλ. €"
    assert guard.fmt_eur(4.5) == "4,50 €"
    assert guard.fmt_pct(-0.0321) == "−3,2%"
    assert guard.fmt_pct(0.0045, 2, sign=True) == "+0,45%"
    assert guard.fmt_pct(float("nan")) == "—"
