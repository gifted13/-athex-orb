"""Pipeline: ανθεκτικότητα (μετοχές χωρίς/με λίγα δεδομένα), συνέπεια caches, αυστηρό JSON κατάστασης."""
from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from athex_orb import journal, labels, pipeline, scout, synthetic
from athex_orb.config import load_config
from athex_orb.features import build_panel
from athex_orb.universe import Stock


def _fast(cfg):
    cfg.data.history_years = 3
    cfg.model.min_train_days = 250
    cfg.model.retrain_every_days = 63
    cfg.backtest.random_runs = 10
    return cfg


@pytest.fixture(scope="module")
def demo_cfg(tmp_path_factory, small_universe):
    cfg = _fast(load_config(data_dir=tmp_path_factory.mktemp("pipe_demo"), use_user_settings=False))
    pipeline.run_refresh(cfg, demo=True, stocks=small_universe)
    return cfg


def _strict(path):
    def bad(c):
        raise AssertionError(f"{path.name}: μη έγκυρη τιμή JSON {c}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=bad)


def test_state_files_are_strict_json(demo_cfg, small_universe):
    sd = pipeline.state_dir(demo_cfg)
    for name in ("universe", "signals", "meta"):
        assert _strict(sd / f"{name}.json")
    files = sorted((sd / "stocks").glob("*.json"))
    assert len(files) == len(small_universe)
    for f in files:
        d = _strict(f)
        assert d["has_data"] is True and len(d["candles"]) == pipeline.N_CANDLES
    for name in pipeline.PICKLES:
        assert (sd / f"{name}.pkl").exists(), name
    meta = pipeline.load_state(demo_cfg, "meta")
    assert meta["mode"] == "demo" and meta["model"]["kind"] == "logreg" and meta["preds_id"]
    assert meta["fingerprint"]["barriers"]["k_tp"] == demo_cfg.barriers.k_tp
    assert pipeline.load_state(demo_cfg, "nope") is None


def test_splice_matches_full_rebuild(synth, small_universe, session_cfg):
    prices, index_df = synth
    full = build_panel(prices, index_df, small_universe, session_cfg)
    full_lab = labels.triple_barrier(prices, full, session_cfg)
    for dropped in ("TPEIR.AT", "OPAP.AT"):              # τράπεζα (κλάδος με 3) και κλάδος με 1 μετοχή
        part_prices = {t: df for t, df in prices.items() if t != dropped}
        part = build_panel(part_prices, index_df, small_universe, session_cfg)
        part_lab = labels.triple_barrier(part_prices, part, session_cfg)
        have = set(part.index.get_level_values("ticker"))
        # προσθήκη: το αποτέλεσμα πρέπει να είναι ίδιο με πλήρη επανυπολογισμό
        sp, sl = pipeline._splice_panel(part, part_lab, prices, index_df, small_universe, have, session_cfg)
        pd.testing.assert_frame_equal(sp, full)
        pd.testing.assert_frame_equal(sl, full_lab)
        # αφαίρεση
        sp, sl = pipeline._splice_panel(full, full_lab, part_prices, index_df, small_universe,
                                        set(prices), session_cfg)
        pd.testing.assert_frame_equal(sp, part)
        pd.testing.assert_frame_equal(sl, part_lab)


def test_refresh_from_user_csv_with_missing_and_short_history(cfg, small_universe):
    """Πηγή CSV (χωρίς δίκτυο): μία μετοχή χωρίς αρχείο, μία με πολύ σύντομο ιστορικό, χωρίς δείκτη."""
    cfg = _fast(cfg)
    cfg.data.source = "csv"
    stocks = [Stock(**s.to_dict()) for s in small_universe[:6]] + [Stock("NODATA.AT", "Χωρίς δεδομένα", "Άλλο")]
    prices, _ = synthetic.generate_universe(stocks[:6], start="2022-01-03", end="2024-12-31", seed=3)
    csv_dir = scout.user_csv_dir(cfg)
    csv_dir.mkdir(parents=True)
    short = stocks[5].ticker
    for t, df in prices.items():
        out = df.tail(120) if t == short else df
        out.to_csv(csv_dir / f"{t.removesuffix('.AT')}.csv", index_label="date")
    msgs = []
    summary = pipeline.run_refresh(cfg, progress=lambda f, m: msgs.append(m), stocks=stocks)
    assert summary["mode"] == "live" and summary["n_with_data"] == 6 and summary["index_ok"] is False
    assert "NODATA.AT" in summary["failed"]
    assert any("Εκπαίδευση" in m for m in msgs) and msgs[-1] == "Η ενημέρωση ολοκληρώθηκε"
    uni = {i["ticker"]: i for i in pipeline.load_state(cfg, "universe")["items"]}
    assert uni["NODATA.AT"]["has_data"] is False and uni["NODATA.AT"]["decision"] == "none"
    assert uni[short]["has_data"] is True and uni[short]["decision"] == "blocked" and uni[short]["orb"] is None
    full = [t for t in uni if t not in (short, "NODATA.AT")]
    assert all(uni[t]["orb"] is not None for t in full)
    st = pipeline.load_stock_state(cfg, short)
    assert "Ιστορικό" in st["decision_reason_el"] and st["agents"][4]["status"] == "block"
    nod = pipeline.load_stock_state(cfg, "NODATA.AT")
    assert nod["has_data"] is False and nod["agents"][0]["id"] == "scout"
    market_notes = " ".join(pipeline.load_stock_state(cfg, full[0])["agents"][1]["notes_el"])
    assert "ισοσταθμικός" in market_notes
    # το ημερολόγιο κράτησε μόνο μετοχές με πρόβλεψη· δεύτερος υπολογισμός δεν διπλασιάζει
    n = journal.summary(cfg, mode="live")["total"]
    assert n == len(full)
    pipeline.recompute_signals(cfg)
    assert journal.summary(cfg, mode="live")["total"] == n


def test_refresh_without_any_data_raises_greek(cfg, small_universe):
    with pytest.raises(pipeline.PipelineError, match="Δεν ήταν δυνατή η λήψη"):
        pipeline.run_refresh(_fast(cfg), stocks=small_universe)        # το yfinance είναι μπλοκαρισμένο στα tests
    assert scout.data_mode(cfg) == "empty"
    with pytest.raises(pipeline.PipelineError, match="τρέξτε πρώτα"):
        pipeline.run_backtest_job(cfg)
    assert pipeline.recompute_signals(cfg) is None


def test_failed_download_keeps_demo_data(demo_cfg, small_universe):
    cfg = copy.deepcopy(demo_cfg)
    before = pipeline.load_state(cfg, "meta")["generated_at"]
    with pytest.raises(pipeline.PipelineError, match="παραμένουν τα δεδομένα επίδειξης"):
        pipeline.run_refresh(cfg, stocks=small_universe)
    assert scout.data_mode(cfg) == "demo"
    assert pipeline.load_state(cfg, "meta")["generated_at"] == before


def test_recompute_keeps_trained_barriers(demo_cfg):
    cfg = copy.deepcopy(demo_cfg)
    cfg.barriers.k_tp = 3.5
    meta = pipeline.load_state(cfg, "meta")
    assert pipeline.needs_retrain(cfg, meta) and not pipeline.needs_retrain(demo_cfg, meta)
    assert pipeline.cfg_for_model(cfg, meta["fingerprint"]).barriers.k_tp == demo_cfg.barriers.k_tp
    pipeline.recompute_signals(cfg)
    st = pipeline.load_stock_state(cfg, "OPAP.AT")
    plan = st["plan"]
    atr_tp = plan["tp_price"] / plan["entry_ref"] - 1
    atr_sl = 1 - plan["sl_price"] / plan["entry_ref"]
    assert atr_tp / atr_sl == pytest.approx(demo_cfg.barriers.k_tp / demo_cfg.barriers.k_sl, rel=1e-3)
    assert pipeline.load_state(cfg, "meta")["fingerprint"] == meta["fingerprint"]


def test_backtest_job_payload(demo_cfg):
    out = pipeline.run_backtest_job(demo_cfg)
    bt = pipeline.load_state(demo_cfg, "backtest")
    assert bt["available"] and bt["random"]["runs"] == demo_cfg.backtest.random_runs
    assert len(bt["equity"]) == len(bt["drawdown"]) and bt["equity"][-1]["d"] == bt["period"]["end"]
    assert bt["n_trades_total"] == out["n_trades"] and len(bt["trades"]) == min(out["n_trades"], 300)
    assert sum(bt["by_label"].values()) == out["n_trades"]
    assert bt["notes_el"][0].startswith("ΕΠΙΔΕΙΞΗ")
    assert "→" in pipeline.verdict_line(out)


def test_stale_index_is_not_used(cfg, small_universe):
    """Η λήψη του δείκτη απέτυχε (έμεινε το παλιό GD.AT.csv) ενώ οι μετοχές ήρθαν: ο δείκτης ΔΕΝ χρησιμοποιείται
    «παγωμένος» — ισοσταθμικός δείκτης στη θέση του, index_ok = False και σαφής σημείωση."""
    cfg = _fast(cfg)
    cfg.data.source = "csv"
    stocks = [Stock(**s.to_dict()) for s in small_universe[:5]]
    prices, index_df = synthetic.generate_universe(stocks, start="2021-01-04", end="2024-06-28", seed=8)
    csv_dir = scout.user_csv_dir(cfg)
    csv_dir.mkdir(parents=True)
    cut = pd.Timestamp("2024-03-29")
    for t, df in prices.items():
        df[df.index <= cut].to_csv(csv_dir / f"{t}.csv", index_label="date")
    index_df[index_df.index <= cut].to_csv(csv_dir / f"{cfg.data.index_ticker}.csv", index_label="date")
    first = pipeline.run_refresh(cfg, stocks=stocks)
    assert first["index_ok"] is True
    # δεύτερη ενημέρωση: οι μετοχές έως 28/06, ο δείκτης λείπει από την πηγή
    for t, df in prices.items():
        df.to_csv(csv_dir / f"{t}.csv", index_label="date")
    (csv_dir / f"{cfg.data.index_ticker}.csv").unlink()
    second = pipeline.run_refresh(cfg, stocks=stocks)
    assert second["index_ok"] is False and second["as_of"] == "2024-06-28"
    meta = pipeline.load_state(cfg, "meta")
    assert meta["index_ok"] is False and "29/03/2024" in meta["index_note_el"]
    panel = pipeline.load_pickle(cfg, "panel")
    tail = panel.xs(stocks[0].ticker, level="ticker").loc["2024-04-01":]
    assert tail["mkt_ret_20"].nunique() > 20                 # όχι μία «παγωμένη» τιμή για 3 μήνες
    notes = " ".join(pipeline.load_stock_state(cfg, stocks[0].ticker)["agents"][1]["notes_el"])
    assert "μόνο έως 29/03/2024" in notes
    bt = pipeline.run_backtest_job(cfg)
    assert bt["metrics"]["buy_hold"]["end"] == bt["period"]["end"] == "2024-06-28"


def test_settings_recompute_never_rewrites_journal(demo_cfg):
    before = {(e["signal_date"], e["ticker"]): (e["decision"], e["reason_el"]) for e in journal.list_entries(demo_cfg)}
    assert before
    cfg = copy.deepcopy(demo_cfg)
    cfg.guard.min_ev_net = -0.05                           # όλα θα γίνονταν «αγορές»
    pipeline.recompute_signals(cfg)
    after = {(e["signal_date"], e["ticker"]): (e["decision"], e["reason_el"]) for e in journal.list_entries(cfg)}
    assert after == before
    pipeline.recompute_signals(demo_cfg)


def test_index_ticker_is_never_a_custom_stock(demo_cfg):
    idx_csv = scout._csv_path(demo_cfg, demo_cfg.data.index_ticker)
    assert idx_csv.exists()
    md5 = idx_csv.read_bytes()
    res = pipeline.add_stock(demo_cfg, Stock(demo_cfg.data.index_ticker, "GD", "Άλλο"))
    assert res["ok"] is False and "Γενικός Δείκτης" in res["error_el"]
    pipeline.remove_stock(demo_cfg, demo_cfg.data.index_ticker)
    assert idx_csv.exists() and idx_csv.read_bytes() == md5


def test_usable_index_lag_rule():
    idx = pd.bdate_range("2024-01-01", "2024-06-28")
    px = {"A.AT": pd.DataFrame({"close": 1.0}, index=idx)}
    fresh = pd.DataFrame({"close": 1.0}, index=idx[:-2])
    stale = pd.DataFrame({"close": 1.0}, index=idx[:-3])
    assert pipeline.usable_index(fresh, px)[0] is fresh
    df, note = pipeline.usable_index(stale, px)
    assert df is None and "ισοσταθμικός" in note
    assert pipeline.usable_index(None, px)[0] is None
