"""Αξιολόγηση/κατανομή των σημερινών υποψηφίων και οι έξι κάρτες πρακτόρων."""
from __future__ import annotations

import copy

import pytest

from athex_orb import explain, guard

GOOD = {"p_tp": 0.62, "p_sl": 0.28, "p_to": 0.10, "timeout_mean": 0.0}
BAD = {"p_tp": 0.15, "p_sl": 0.75, "p_to": 0.10, "timeout_mean": 0.0}


@pytest.fixture
def setup(panel, small_universe, session_cfg):
    cfg = copy.deepcopy(session_cfg)
    cfg.guard.min_adv_eur = 0.0                       # όλες περνούν τη ρευστότητα, για καθαρό έλεγχο κατανομής
    last = panel.index.get_level_values("date").max()
    rows = {t: panel.loc[(last, t)] for t in panel.loc[last].index}
    sectors = {s.ticker: s.sector for s in small_universe}
    return cfg, last, rows, sectors


def _assess(cfg, last, rows, pred_for):
    return [explain.prepare_assessment(t, r, pred_for(t), cfg, as_of=last, row_date=last) for t, r in rows.items()]


def test_allocate_respects_limits_and_ranks_by_ev(setup):
    cfg, last, rows, sectors = setup
    cfg.portfolio.max_positions = 2
    a = _assess(cfg, last, rows, lambda t: GOOD)
    cands = [x for x in a if x["candidate"]]
    assert len(cands) >= 4
    explain.allocate(a, sectors, [], cfg)
    takes = [x for x in a if x["decision"] == "take"]
    assert len(takes) == 2
    best = sorted(cands, key=lambda x: -x["ev"]["ev_net"])[:2]
    assert {x["ticker"] for x in takes} == {x["ticker"] for x in best}
    for x in takes:
        assert guard.MIN_POSITION_EUR <= x["size_eur"] <= cfg.portfolio.position_eur_max
        assert x["plan"]["size_eur"] == x["size_eur"] and x["plan"]["shares"] == pytest.approx(
            x["size_eur"] / x["entry_ref"])
        assert x["plan"]["tp_price"] > x["entry_ref"] > x["plan"]["sl_price"]
    rest = [x for x in cands if x["decision"] != "take"]
    assert rest and all(x["decision"] == "skip" and "θέση" in x["reason_el"] for x in rest)
    assert all(x["plan"]["size_eur"] is None for x in rest)


def test_allocate_sector_cap_held_and_cash(setup):
    cfg, last, rows, sectors = setup
    cfg.portfolio.max_positions = 10
    cfg.portfolio.max_per_sector = 1
    a = _assess(cfg, last, rows, lambda t: GOOD)
    explain.allocate(a, sectors, [{"ticker": "PPC.AT", "sector": "Ενέργεια", "size_eur": 1000.0}], cfg)
    by = {x["ticker"]: x for x in a}
    banks = [t for t in by if sectors[t] == "Τράπεζες" and by[t]["decision"] == "take"]
    assert len(banks) <= 1
    assert by["PPC.AT"]["decision"] == "skip"                     # ο κλάδος (και η μετοχή) έχει ήδη θέση
    assert by["MOH.AT"]["decision"] == "skip" and "κλάδου" in by["MOH.AT"]["short_el"]
    # λίγα μετρητά: μόνο μία θέση χωρά, οι επόμενες απορρίπτονται για ανεπαρκή διαθέσιμα
    cfg.portfolio.max_per_sector = 10
    cfg.portfolio.capital_eur = 700.0
    a = _assess(cfg, last, rows, lambda t: GOOD)
    explain.allocate(a, sectors, [], cfg)
    takes = [x for x in a if x["decision"] == "take"]
    assert len(takes) == 1 and takes[0]["size_eur"] <= 700.0
    assert any(x["short_el"] == "ανεπαρκή διαθέσιμα" for x in a if x["candidate"] and x["decision"] == "skip")


def test_decisions_skip_blocked_none(setup):
    cfg, last, rows, _ = setup
    a = {x["ticker"]: x for x in _assess(cfg, last, rows, lambda t: BAD)}
    assert all(x["decision"] == "skip" and "κάτω από το ελάχιστο" in x["reason_el"] for x in a.values())
    cfg.guard.min_price_eur = 1e6
    t = next(iter(rows))
    b = explain.prepare_assessment(t, rows[t], GOOD, cfg, as_of=last, row_date=last)
    assert b["decision"] == "blocked" and "Τιμή" in b["reason_el"] and b["orb"] is not None
    c = explain.prepare_assessment(t, rows[t], None, cfg, as_of=last, row_date=last, model_ready=False)
    assert c["decision"] == "blocked"                        # ο φύλακας προηγείται
    cfg.guard.min_price_eur = 0.5
    c = explain.prepare_assessment(t, rows[t], None, cfg, as_of=last, row_date=last, model_ready=False)
    assert c["decision"] == "none" and "μοντέλο" in c["reason_el"]
    old = rows[t].copy()
    d = explain.prepare_assessment(t, old, GOOD, cfg, as_of=last, row_date="2000-01-03")
    assert d["decision"] == "none" and "σταματούν" in d["reason_el"]
    e = explain.prepare_assessment(t, None, None, cfg, as_of=last)
    assert e["decision"] == "none" and e["has_row"] is False


def test_agent_report_cards(setup, panel, labels_df, small_universe):
    cfg, last, _, _ = setup
    stock = small_universe[0]
    cards = explain.agent_report(stock.ticker, panel, labels_df, GOOD, None, stock, cfg,
                                 open_positions=[{"ticker": stock.ticker, "sector": stock.sector, "size_eur": 900}],
                                 context={"mode": "demo", "as_of": last, "sector_peers": 3, "index_ok": False})
    assert [c["id"] for c in cards] == explain.AGENT_IDS
    for c in cards:
        assert c["name_el"] and c["role_el"] and c["headline_el"] and c["status"] in {"ok", "warn", "block", "info"}
        assert len(c["metrics"]) <= 4
        for m in c["metrics"]:
            assert m["fmt"] in {"pct", "eur", "num", "int", "text", "date"}
            assert m["value"] is None or isinstance(m["value"], (int, float, str))
    scout, market, judge, quant_, guard_, ex = cards
    assert scout["headline_el"].startswith("Ρευστότητα") and "€" in scout["headline_el"]
    assert any("ΕΠΙΔΕΙΞΗ" in n for n in scout["notes_el"])
    assert any(w in market["headline_el"] for w in ("Ανοδική", "Πτωτική", "Ουδέτερη"))
    assert any("ισοσταθμικός" in n for n in market["notes_el"])
    assert judge["drivers"] == []                             # χωρίς μοντέλο logreg δεν υπάρχουν παράγοντες
    assert quant_["headline_el"].startswith("EV καθαρό")
    ids = [c["id"] for c in guard_["checks"]]
    assert {"liquidity", "price", "volatility", "history", "market", "features", "ev", "not_held"} <= set(ids)
    held = next(c for c in guard_["checks"] if c["id"] == "not_held")
    assert held["passed"] is False                            # ήδη ανοιχτή θέση στη μετοχή
    assert guard_["status"] == "warn" and ex["status"] == "info" and ex["headline_el"].startswith("Καμία θέση")
    none_cards = explain.agent_report("XYZ.AT", None, None, None, None, None, cfg)
    assert [c["id"] for c in none_cards] == explain.AGENT_IDS and none_cards[4]["checks"] == []


def test_eur_short_and_lc():
    assert explain.eur_short(4_200_000) == "4,2 εκ. €"
    assert explain.eur_short(350_000) == "350 χιλ. €"
    assert explain.eur_short(1250) == "1.250 €"
    assert explain.eur_short(4.5) == "4,50 €"
    assert explain.eur_short(float("nan")) == "—"
    assert explain._lc("Μέσος τζίρος") == "μέσος τζίρος" and explain._lc("ATR / τιμή") == "ATR / τιμή"
