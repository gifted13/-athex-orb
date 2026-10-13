"""Backtest χαρτοφυλακίου και σύγκριση με την τύχη — τα «τεστ εντιμότητας».

(a) Σε καθαρό τυχαίο περίπατο (planted_signal = 0) η στρατηγική ΔΕΝ πρέπει να δείχνει πλεονέκτημα.
(b) Με φυτεμένο σήμα (planted_signal = 2) πρέπει να νικά καθαρά την τύχη, με λογική βαθμονόμηση.
(c) Λογιστική μετρητών, κόστη, όρια θέσεων και κανόνες εισόδου/εξόδου ακριβώς όπως στις ετικέτες.

Σημείωση για τα όρια του (a): με τον τρόπο που φτιάχνονται οι τυχαίες στρατηγικές (ίδιες ημέρες, ίδιο πλήθος
υποψηφίων ανά ημέρα, ίδια κόστη, προβλέψεις σε ανακατεμένες μετοχές) το percentile της στρατηγικής σε
δεδομένα χωρίς σήμα είναι περίπου ομοιόμορφο στο [0, 1]. Ελέγχθηκε εκτός tests σε 16 σπόρους × 31 μετοχές ×
8 έτη (μέσος όρος 0,53, κανένας ≥ 0,95) και σε 6 σπόρους του μικρού συνόλου των tests (0,15–0,94).
Γι' αυτό: ένας σταθερός σπόρος με όριο < 0,97 (≈ 3% πιθανότητα «άτυχης» αποτυχίας αν αλλάξει κάτι άσχετο)
και ο μέσος όρος 4 σπόρων < 0,85 (πιθανότητα < 1% υπό τη μηδενική υπόθεση). Μια μεροληπτική σύγκριση —
π.χ. τυχαίες επιλογές που αγνοούν το κόστος ή παίρνουν περισσότερες θέσεις από το μοντέλο — δίνει
συστηματικά υψηλά percentiles (μέσος ≈ 0,7, 6/16 ≥ 0,95) και θα «έσπαγε» αυτά τα όρια.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import pytest

from athex_orb import benchmarks, executor, features, guard, labels, orb_model, quant, synthetic
from athex_orb.labels import LABEL_TP, barrier_prices, simulate_exit

RUNS = 200


def run_all(prices, index_df, stocks, cfg, runs=RUNS, panel=None, lab=None):
    if panel is None:
        panel = features.build_panel(prices, index_df, stocks, cfg)
        lab = labels.triple_barrier(prices, panel, cfg)
    preds, _ = orb_model.walk_forward(panel, lab, cfg)
    ctx = executor.prepare(prices, panel, preds, lab, cfg)
    res = executor.run_backtest(prices, panel, preds, lab, cfg, context=ctx)
    rb = benchmarks.random_benchmark(prices, panel, preds, lab, cfg, runs, context=ctx)
    capital = cfg.portfolio.capital_eur
    bh = benchmarks.buy_and_hold(index_df, ctx.cal[0], ctx.cal[-1], capital)
    bhm = benchmarks.metrics(bh, None, capital)
    from athex_orb import market
    bhu = benchmarks.buy_and_hold(market.equal_weight_index(prices), ctx.cal[0], ctx.cal[-1], capital)
    bhum = benchmarks.metrics(bhu, None, capital)
    sk = benchmarks.skill(preds, lab)
    v = benchmarks.verdict(res["metrics"], rb["final_returns"], bhm, bh_universe_metrics=bhum, skill_result=sk)
    return {"prices": prices, "index_df": index_df, "panel": panel, "labels": lab, "preds": preds, "ctx": ctx,
            "res": res, "rb": rb, "bh": bh, "bhm": bhm, "bhum": bhum, "skill": sk, "verdict": v, "cfg": cfg}


@pytest.fixture(scope="module")
def bt_cfg(session_cfg):
    return copy.deepcopy(session_cfg)


@pytest.fixture(scope="module")
def null_run(synth, panel, labels_df, small_universe, bt_cfg):
    prices, index_df = synth
    return run_all(prices, index_df, small_universe, bt_cfg, panel=panel, lab=labels_df)


@pytest.fixture(scope="module")
def planted_run(small_universe, synth_range, bt_cfg):
    start, end, seed = synth_range
    prices, index_df = synthetic.generate_universe(small_universe, start=start, end=end, seed=seed,
                                                   planted_signal=2.0)
    return run_all(prices, index_df, small_universe, bt_cfg)


# ---------------------------------------------------------------- (a) καμία ψεύτικη «ικανότητα»

def test_no_edge_on_random_walk(null_run):
    m, v = null_run["res"]["metrics"], null_run["verdict"]
    assert m["n_trades"] >= 30, "το τεστ έχει νόημα μόνο αν η στρατηγική όντως συναλλάσσεται"
    assert v["edge"] is False
    assert v["level"] != "good"
    assert v["percentile"] < 0.97
    assert v["text_el"]


def test_skill_gate_on_random_walk_and_planted_signal(null_run, planted_run):
    # χωρίς σήμα οι πιθανότητες εκτός δείγματος δεν νικούν τις σταθερές συχνότητες· με σήμα τις νικούν
    assert null_run["skill"]["ok"] is False and null_run["skill"]["n"] > 1000
    assert planted_run["skill"]["ok"] is True
    assert planted_run["skill"]["logloss_model"] < planted_run["skill"]["logloss_base"]


def test_random_metrics_are_medians_across_runs(null_run):
    """Sharpe/μεταβλητότητα/πτώση των «τυχαίων» = διάμεσος των μετρικών κάθε εκτέλεσης — όχι οι μετρικές της
    διάμεσης καμπύλης σημείο προς σημείο (που δεν την παρήγαγε καμία εκτέλεση και είναι πιο «ομαλή»)."""
    rb = null_run["rb"]
    for k in ("volatility", "max_drawdown", "sharpe", "cagr"):
        vals = np.asarray(rb["run_metrics"][k], dtype=float)
        assert len(vals) == rb["runs"]
        assert rb["metrics_median"][k] == pytest.approx(float(np.nanmedian(vals)))
    curve = benchmarks.metrics(rb["equity_median"], None, null_run["cfg"].portfolio.capital_eur)
    assert curve["volatility"] < rb["metrics_median"]["volatility"]      # η «καμπύλη διαμέσου» υποτιμά τον κίνδυνο


def test_random_benchmark_takes_comparable_number_of_trades(null_run, planted_run):
    """Οι τυχαίες στρατηγικές εξετάζουν όσες υποψήφιες και το μοντέλο ⇒ παρόμοιο πλήθος συναλλαγών (και
    κόστος). Αν έπαιρναν συστηματικά περισσότερες θέσεις, θα πλήρωναν περισσότερα και το μοντέλο θα «νικούσε»
    χωρίς πραγματική ικανότητα."""
    for run in (null_run, planted_run):
        n_model = run["res"]["metrics"]["n_trades"]
        n_rand = np.median(run["rb"]["n_trades"])
        assert abs(n_rand - n_model) <= 0.2 * n_model + 5, (n_model, n_rand)


@pytest.mark.parametrize("seeds", [(1, 3, 5)])
def test_no_edge_on_random_walk_across_seeds(null_run, small_universe, synth_range, bt_cfg, seeds):
    start, end, _ = synth_range
    pcts = [null_run["verdict"]["percentile"]]
    for seed in seeds:
        prices, index_df = synthetic.generate_universe(small_universe, start=start, end=end, seed=seed)
        run = run_all(prices, index_df, small_universe, bt_cfg, runs=100)
        assert run["verdict"]["edge"] is False
        pcts.append(run["verdict"]["percentile"])
    assert max(pcts) < 0.99, pcts
    assert np.mean(pcts) < 0.85, pcts


# ---------------------------------------------------------------- (b) βρίσκει πραγματικό σήμα

def test_planted_signal_beats_random(planted_run):
    v, rb = planted_run["verdict"], planted_run["rb"]
    m = planted_run["res"]["metrics"]
    assert m["n_trades"] >= 100
    assert v["percentile"] >= 0.9
    assert m["total_return"] > np.percentile(rb["final_returns"], 95)
    assert v["edge"] is True and v["level"] == "good"


def test_planted_signal_calibration_is_monotonic(planted_run):
    cal = benchmarks.calibration(planted_run["preds"], planted_run["labels"], bins=5)
    assert len(cal) == 5
    realized = [c["realized"] for c in cal]
    predicted = [c["predicted"] for c in cal]
    assert predicted == sorted(predicted)
    # κάθε επόμενος κάδος τουλάχιστον όσο ο προηγούμενος (με μικρή ανοχή) και καθαρή συνολική άνοδος
    assert all(b >= a - 0.02 for a, b in zip(realized, realized[1:])), realized
    assert realized[-1] - realized[0] >= 0.15, realized
    # και οι πιθανότητες είναι ρεαλιστικές (όχι μόνο σωστά ταξινομημένες)
    assert all(abs(c["predicted"] - c["realized"]) < 0.08 for c in cal), cal
    assert sum(c["n"] for c in cal) == int(planted_run["preds"].join(planted_run["labels"]["label"])
                                            ["label"].notna().sum())


# ---------------------------------------------------------------- (c) λογιστική και κανόνες

def check_accounting(res, prices, panel, cfg, model_rules=True):
    """Ανεξάρτητη ανακατασκευή της καμπύλης κεφαλαίου και έλεγχος όλων των κανόνων εκτέλεσης."""
    eq, tr = res["equity"], res["trades"]
    capital = float(cfg.portfolio.capital_eur)
    p = cfg.portfolio
    assert len(tr) > 0
    assert list(tr.columns[:17]) == executor.TRADE_COLUMNS[:17]

    # τελικό κεφάλαιο = αρχικό + καθαρά αποτελέσματα (όλες οι θέσεις έχουν κλείσει)
    assert eq.iloc[-1] == pytest.approx(capital + tr["net_pnl_eur"].sum(), abs=1e-6)
    assert eq.iloc[0] == pytest.approx(capital, abs=1e-9)
    # κόστη θετικά σε κάθε συναλλαγή, και το καθαρό = μικτό − κόστη
    assert (tr["cost_eur"] > 0).all()
    assert np.allclose(tr["net_pnl_eur"], tr["size_eur"] * tr["gross_ret"] - tr["cost_eur"], rtol=0, atol=1e-8)
    assert np.allclose(tr["net_ret"], tr["net_pnl_eur"] / tr["size_eur"], rtol=0, atol=1e-12)
    # μέγεθος θέσης
    assert (tr["size_eur"] >= guard.MIN_POSITION_EUR - 1e-9).all()
    assert (tr["size_eur"] <= p.position_eur_max + 1e-9).all()

    cal = eq.index
    end = cal[-1]
    k_tp, k_sl, horizon = cfg.barriers.k_tp, cfg.barriers.k_sl, cfg.barriers.horizon_days
    buy_cost = np.empty(len(tr))
    for i, t in enumerate(tr.itertuples(index=False)):
        df = prices[t.ticker]
        idx = df.index
        row = panel.loc[(t.signal_date, t.ticker)]
        assert t.size_eur <= p.max_adv_fraction * row["adv20_eur"] + 1e-6
        # είσοδος στο επόμενο άνοιγμα της μετοχής
        sp = idx.get_loc(t.signal_date)
        assert t.entry_date == idx[sp + 1]
        assert t.entry_price == df["open"].iloc[sp + 1]
        # έξοδος ακριβώς όπως ο κοινός προσομοιωτής των ετικετών
        o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
        tp, sl = barrier_prices(t.entry_price, row["atr_pct"], k_tp, k_sl)
        lab, ex_i, ex_px = simulate_exit(o, h, l, c, sp + 1, t.entry_price, tp, sl, horizon)
        if t.label >= 0:
            assert (t.label, t.exit_date, t.exit_price) == (lab, idx[ex_i], pytest.approx(ex_px))
            assert t.bars_held == ex_i - sp
        else:   # ανοιχτή στο τέλος ⇒ κλείσιμο στο τελευταίο κλείσιμο
            assert lab == -1 or idx[ex_i] > end
            assert t.exit_date == end
            assert t.exit_price == df["close"].loc[:end].iloc[-1]
        # κόστη από τον «Ποσοτικό», αγορά στο ποσό της εντολής, πώληση στο ποσό της πώλησης
        vol = row["vol_20"]
        b = quant.side_cost_pct(t.size_eur, row["adv20_eur"], t.entry_price, cfg, "buy", vol=vol)["total"]
        proceeds = t.size_eur / t.entry_price * t.exit_price
        s = quant.side_cost_pct(proceeds, row["adv20_eur"], t.exit_price, cfg, "sell", vol=vol)["total"]
        buy_cost[i] = t.size_eur * b
        assert t.cost_eur == pytest.approx(t.size_eur * b + proceeds * s, rel=1e-9)
        if model_rules:
            # μόνο υποψήφιες που πέρασαν τους ελέγχους του Φύλακα
            assert guard.passed(guard.stock_checks(row, cfg))
            assert guard.passed(guard.decision_checks({"ev_net": t.ev_net}, t.p_tp, cfg))

    # ποτέ δύο ταυτόχρονες θέσεις στην ίδια μετοχή (η θέση «δεσμεύεται» από την ημέρα του σήματος)
    for _, g in tr.sort_values("signal_date").groupby("ticker"):
        assert (g["signal_date"].to_numpy()[1:] >= g["exit_date"].to_numpy()[:-1]).all()
        assert (g["entry_date"].to_numpy()[1:] > g["exit_date"].to_numpy()[:-1]).all()

    # ανοιχτές θέσεις στο κλείσιμο κάθε ημέρας: σήμα ≤ d < έξοδος
    sig = cal.searchsorted(pd.DatetimeIndex(tr["signal_date"]))
    ext = cal.searchsorted(pd.DatetimeIndex(tr["exit_date"]))
    ent = cal.searchsorted(pd.DatetimeIndex(tr["entry_date"]))
    n_open = np.zeros(len(cal) + 1, dtype=int)
    np.add.at(n_open, sig, 1)
    np.add.at(n_open, ext, -1)
    n_open = np.cumsum(n_open)[:-1]
    assert n_open.max() <= p.max_positions
    for sector, g in tr.groupby("sector"):
        cnt = np.zeros(len(cal) + 1, dtype=int)
        np.add.at(cnt, cal.searchsorted(pd.DatetimeIndex(g["signal_date"])), 1)
        np.add.at(cnt, cal.searchsorted(pd.DatetimeIndex(g["exit_date"])), -1)
        assert np.cumsum(cnt)[:-1].max() <= p.max_per_sector, sector

    # ανεξάρτητη αποτίμηση: μετρητά + θέσεις στο κλείσιμο (ή δεσμευμένα μετρητά πριν από την είσοδο)
    closes = pd.DataFrame({t: prices[t]["close"] for t in tr["ticker"].unique()}).reindex(cal).ffill()
    size = tr["size_eur"].to_numpy()
    back = size + buy_cost + tr["net_pnl_eur"].to_numpy()       # = έσοδα πώλησης − κόστος πώλησης
    shares = size / tr["entry_price"].to_numpy()
    tick = tr["ticker"].to_numpy()
    rebuilt = np.empty(len(cal))
    for d in range(len(cal)):
        cash = capital - (size + buy_cost)[sig <= d].sum() + back[ext <= d].sum()
        assert cash >= -1e-6
        val = cash
        for j in np.flatnonzero((sig <= d) & (ext > d)):
            val += shares[j] * closes.at[cal[d], tick[j]] if ent[j] <= d else size[j] + buy_cost[j]
        rebuilt[d] = val
    assert np.max(np.abs(rebuilt - eq.to_numpy())) < 1e-6


def test_cash_accounting_and_rules_model(planted_run):
    r = planted_run
    check_accounting(r["res"], r["prices"], r["panel"], r["cfg"])
    assert (r["res"]["trades"]["label"] == -1).sum() <= r["cfg"].portfolio.max_positions


def test_cash_accounting_and_rules_null(null_run):
    r = null_run
    check_accounting(r["res"], r["prices"], r["panel"], r["cfg"])


def test_cash_accounting_random_selector_and_reproducibility(planted_run):
    r = planted_run
    a = executor.run_backtest(r["prices"], r["panel"], r["preds"], r["labels"], r["cfg"], selector="random",
                              rng=123, context=r["ctx"])
    check_accounting(a, r["prices"], r["panel"], r["cfg"], model_rules=False)
    b = executor.run_backtest(r["prices"], r["panel"], r["preds"], r["labels"], r["cfg"], selector="random",
                              rng=np.random.default_rng(123), context=r["ctx"])
    c = executor.run_backtest(r["prices"], r["panel"], r["preds"], r["labels"], r["cfg"], selector="random",
                              rng=7, context=r["ctx"])
    assert a["equity"].equals(b["equity"])
    assert not a["equity"].equals(c["equity"])
    # η τυχαία επιλογή παίρνει μόνο υποψήφιες που περνούν τους ελέγχους μετοχής
    for t in a["trades"].itertuples(index=False):
        assert guard.passed(guard.stock_checks(r["panel"].loc[(t.signal_date, t.ticker)], r["cfg"]))
    # και ο αριθμός νέων θέσεων ανά ημέρα δεν ξεπερνά όσες υποψήφιες εξέτασε το μοντέλο
    per_day = a["trades"].groupby("signal_date").size()
    want = pd.Series(r["ctx"].n_want, index=r["ctx"].cal)
    assert (per_day <= want.reindex(per_day.index)).all()


@pytest.mark.parametrize("over", [
    {"capital_eur": 2600.0},                                   # τα μετρητά περιορίζουν τη 2η θέση
    {"max_positions": 2, "max_per_sector": 1},                 # αυστηρά όρια χαρτοφυλακίου
    {"position_eur_max": 400.0},                               # η ελάχιστη προμήθεια βαραίνει
])
def test_cash_accounting_under_tight_constraints(planted_run, over):
    r = planted_run
    cfg = copy.deepcopy(r["cfg"])
    for k, v in over.items():
        setattr(cfg.portfolio, k, v)
    res = executor.run_backtest(r["prices"], r["panel"], r["preds"], r["labels"], cfg)
    check_accounting(res, r["prices"], r["panel"], cfg)
    reasons = set(res["decisions"]["reason_el"])
    if "capital_eur" in over:
        assert executor.REASONS_EL[executor.R_CASH] in reasons
    if "max_positions" in over:
        assert executor.REASONS_EL[executor.R_MAXPOS] in reasons


def test_decisions_frame(planted_run):
    r = planted_run
    dec, tr = r["res"]["decisions"], r["res"]["trades"]
    assert list(dec.columns) == executor.DECISION_COLUMNS
    assert len(dec) == len(r["ctx"].index)
    assert int(dec["taken"].sum()) == len(tr)
    taken = set(zip(dec.loc[dec["taken"], "date"], dec.loc[dec["taken"], "ticker"]))
    assert taken == set(zip(tr["signal_date"], tr["ticker"]))
    assert dec["reason_el"].map(lambda s: isinstance(s, str) and len(s) > 3).all()
    assert set(dec["reason_el"]) <= set(executor.REASONS_EL.values())
    # οι γραμμές που δεν πέρασαν τον Φύλακα έχουν αιτιολογία «Φύλακας: …»
    assert dec["reason_el"].str.startswith("Φύλακας").any()


def test_metrics_of_backtest(planted_run):
    r = planted_run
    m, tr = r["res"]["metrics"], r["res"]["trades"]
    assert m["n_trades"] == len(tr)
    assert m["total_costs_eur"] == pytest.approx(tr["cost_eur"].sum())
    assert 0 < m["exposure"] <= 1
    assert 0 <= m["win_rate"] <= 1
    assert m["max_drawdown"] <= 0
    assert m["avg_bars_held"] == pytest.approx(tr["bars_held"].mean())
    assert 1 <= m["avg_bars_held"] <= r["cfg"].barriers.horizon_days
    assert m["avg_win"] > 0 > m["avg_loss"]


def test_backtest_follows_current_barrier_settings(planted_run):
    """Οι έξοδοι υπολογίζονται από τις τιμές με τα όρια του cfg (όχι από παλιές ετικέτες)."""
    r = planted_run
    cfg = copy.deepcopy(r["cfg"])
    cfg.barriers.k_tp, cfg.barriers.horizon_days = 3.0, 5
    res = executor.run_backtest(r["prices"], r["panel"], r["preds"], r["labels"], cfg)
    check_accounting(res, r["prices"], r["panel"], cfg)
    assert res["trades"]["bars_held"].max() <= 5


def test_empty_inputs(cfg, panel, labels_df, synth):
    empty_preds = orb_model._empty_preds()
    res = executor.run_backtest(synth[0], panel, empty_preds, labels_df, cfg)
    assert len(res["equity"]) == 0 and len(res["trades"]) == 0
    assert res["metrics"]["n_trades"] == 0
    rb = benchmarks.random_benchmark(synth[0], panel, empty_preds, labels_df, cfg, 10)
    assert rb["final_returns"] == [] and rb["runs"] == 0 and rb["costs_eur"] == [] and rb["win_rates"] == []
    with pytest.raises(ValueError):
        executor.run_backtest(synth[0], panel, empty_preds, labels_df, cfg, selector="best")


# ---------------------------------------------------------------- ταχύτητα

def test_random_benchmark_speed(full_universe, session_cfg):
    """200 τυχαίες εκτελέσεις σε 31 μετοχές × ~6 έτη πρέπει να τελειώνουν άνετα κάτω από 60 δ."""
    cfg = copy.deepcopy(session_cfg)
    cfg.guard.min_ev_net = -1.0            # πολλές υποψήφιες κάθε ημέρα: η πιο «βαριά» περίπτωση
    prices, index_df = synthetic.generate_universe(full_universe, start="2019-01-01", end="2024-12-31", seed=3)
    panel = features.build_panel(prices, index_df, full_universe, cfg)
    lab = labels.triple_barrier(prices, panel, cfg)
    ok = np.isfinite(panel[features.FEATURE_COLUMNS].to_numpy(dtype="float64")).all(axis=1)
    ok &= np.asarray(panel.index.get_level_values("date") >= pd.Timestamp("2020-01-01"))
    rng = np.random.default_rng(0)
    p = rng.dirichlet([3, 5, 2], int(ok.sum()))
    preds = pd.DataFrame({"p_tp": p[:, 0], "p_sl": p[:, 1], "p_to": p[:, 2], "timeout_mean": 0.0,
                          "model_date": pd.Timestamp("2020-01-01")}, index=panel.index[ok])
    t0 = time.perf_counter()
    ctx = executor.prepare(prices, panel, preds, lab, cfg)
    rb = benchmarks.random_benchmark(prices, panel, preds, lab, cfg, 200, context=ctx)
    elapsed = time.perf_counter() - t0
    assert len(rb["final_returns"]) == 200
    assert np.median(rb["n_trades"]) > 300
    # κόστη και ποσοστό κερδοφόρων ανά εκτέλεση (για τη στήλη «Τυχαίες (διάμεσος)» του UI)
    assert len(rb["costs_eur"]) == len(rb["win_rates"]) == 200 and min(rb["costs_eur"]) > 0
    assert all(0.0 <= w <= 1.0 for w in rb["win_rates"])
    assert elapsed < 60, f"{elapsed:.1f} s"


# ---------------------------------------------------------------- benchmarks: μονάδες

def _eq(values, start="2021-01-04"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values), name="date"), dtype="float64")


def test_metrics_known_values():
    eq = _eq([100, 110, 99, 121, 121])
    m = benchmarks.metrics(eq, None, 100)
    assert m["total_return"] == pytest.approx(0.21)
    assert m["max_drawdown"] == pytest.approx(99 / 110 - 1)
    assert m["exposure"] == 1.0 and m["n_trades"] == 0
    rets = np.array([0.0, 0.1, 99 / 110 - 1, 121 / 99 - 1, 0.0])
    assert m["volatility"] == pytest.approx(rets.std(ddof=1) * np.sqrt(252))
    assert m["sharpe"] == pytest.approx(rets.mean() / rets.std(ddof=1) * np.sqrt(252))
    assert m["start"] == "2021-01-04" and m["end"] == "2021-01-08"
    # πτώση από το αρχικό κεφάλαιο μετρά ως drawdown
    assert benchmarks.metrics(_eq([90, 95]), None, 100)["max_drawdown"] == pytest.approx(-0.1)


def test_metrics_trade_stats_and_exposure():
    eq = _eq([1000.0] * 10)
    d = eq.index
    trades = pd.DataFrame({
        "entry_date": [d[1], d[5]], "exit_date": [d[2], d[5]],
        "net_pnl_eur": [30.0, -10.0], "net_ret": [0.03, -0.01], "cost_eur": [2.0, 3.0], "bars_held": [2, 1],
    })
    m = benchmarks.metrics(eq, trades, 1000)
    assert m["n_trades"] == 2
    assert m["win_rate"] == 0.5
    assert m["avg_win"] == pytest.approx(0.03) and m["avg_loss"] == pytest.approx(-0.01)
    assert m["profit_factor"] == pytest.approx(3.0)
    assert m["total_costs_eur"] == pytest.approx(5.0)
    assert m["avg_bars_held"] == pytest.approx(1.5)
    assert m["exposure"] == pytest.approx(3 / 10)
    m0 = benchmarks.metrics(eq, trades.iloc[:0], 1000)
    assert m0["n_trades"] == 0 and m0["exposure"] == 0.0 and np.isnan(m0["win_rate"])


def test_buy_and_hold():
    idx = pd.bdate_range("2021-01-04", periods=6, name="date")
    index_df = pd.DataFrame({"close": [100, 102, 104, 103, 110, 120]}, index=idx, dtype="float64")
    bh = benchmarks.buy_and_hold(index_df, idx[1], idx[4], 5000)
    assert list(bh.index) == list(idx[1:5])
    assert bh.iloc[0] == 5000
    assert bh.iloc[-1] == pytest.approx(5000 * 110 / 102)
    assert len(benchmarks.buy_and_hold(None, idx[0], idx[-1], 5000)) == 0


def test_verdict_rules():
    rand = list(np.linspace(-0.2, 0.2, 200))
    good = {"total_return": 0.5, "n_trades": 80}
    v = benchmarks.verdict(good, rand, {"total_return": 0.1})
    assert v["edge"] and v["level"] == "good" and v["percentile"] == 1.0 and "100%" in v["text_el"]
    assert v["p_value"] == pytest.approx(1 / 201)
    # νικά την τύχη αλλά όχι την αγορά και διακράτηση ⇒ όχι πλεονέκτημα
    v = benchmarks.verdict(good, rand, {"total_return": 0.6})
    assert not v["edge"] and v["level"] == "weak" and not v["beats_buy_hold"]
    # νικά τον δείκτη τιμών αλλά όχι το ισοσταθμικό χαρτοφυλάκιο των ίδιων μετοχών ⇒ όχι πλεονέκτημα
    v = benchmarks.verdict(good, rand, {"total_return": 0.1}, bh_universe_metrics={"total_return": 0.7})
    assert not v["edge"] and v["level"] == "weak" and "ισοσταθμική λίστα" in v["text_el"]
    # λίγες συναλλαγές ⇒ κανένα συμπέρασμα
    v = benchmarks.verdict({"total_return": 0.5, "n_trades": 5}, rand, {"total_return": 0.1})
    assert not v["edge"] and v["level"] == "none"
    # χάνει χρήματα (έστω λιγότερα από την τύχη και τον δείκτη) ⇒ όχι πλεονέκτημα
    v = benchmarks.verdict({"total_return": -0.05, "n_trades": 80}, list(np.linspace(-0.6, -0.3, 150)),
                           {"total_return": -0.4})
    assert not v["edge"] and v["level"] == "weak" and "έχασε" in v["text_el"]
    # μέτριο
    v = benchmarks.verdict({"total_return": 0.0, "n_trades": 80}, rand, {"total_return": -0.1})
    assert not v["edge"] and v["level"] == "none" and 0.4 < v["percentile"] < 0.6
    # χειρότερα από τις τυχαίες: το λέει ρητά
    v = benchmarks.verdict({"total_return": -0.18, "n_trades": 80}, rand, {"total_return": -0.1})
    assert v["level"] == "none" and v["percentile"] < 0.1 and "χειρότερα από το" in v["text_el"]
    # ισοπαλίες μετρούν μισές
    v = benchmarks.verdict({"total_return": 0.0, "n_trades": 80}, [0.0] * 10, {"total_return": -0.1})
    assert v["percentile"] == 0.5
    # χωρίς συναλλαγές / χωρίς τυχαίες εκτελέσεις
    assert benchmarks.verdict({"total_return": 0.0, "n_trades": 0}, rand, {"total_return": 0})["level"] == "none"
    v = benchmarks.verdict(good, [], {"total_return": 0.1})
    assert not v["edge"] and np.isnan(v["percentile"])


@pytest.mark.parametrize("n_runs", [1, 5, 19, 99])
def test_verdict_needs_enough_random_runs(n_runs):
    # με n τυχαίες εκτελέσεις μια στρατηγική χωρίς δεξιότητα τις ξεπερνά όλες με πιθανότητα 1/(n+1)
    v = benchmarks.verdict({"total_return": 0.05, "n_trades": 40}, [0.01] * n_runs, {"total_return": 0.02})
    assert not v["edge"] and v["level"] == "none" and "τουλάχιστον 100" in v["text_el"]


def test_verdict_requires_out_of_sample_skill():
    rand = list(np.linspace(-0.2, 0.2, 200))
    good = {"total_return": 0.5, "n_trades": 80}
    no_skill = {"ok": False, "logloss_model": 0.937, "logloss_base": 0.928, "n": 5000, "p_value": 0.9}
    v = benchmarks.verdict(good, rand, {"total_return": 0.1}, skill_result=no_skill)
    assert not v["edge"] and v["level"] == "none" and "log-loss" in v["text_el"] and v["skill_ok"] is False
    # «weak» (80–95%) επίσης μόνο με δεξιότητα
    v = benchmarks.verdict({"total_return": 0.15, "n_trades": 80}, rand, {"total_return": 0.1}, skill_result=no_skill)
    assert v["level"] == "none"
    with_skill = dict(no_skill, ok=True, logloss_model=0.90)
    v = benchmarks.verdict(good, rand, {"total_return": 0.1}, skill_result=with_skill)
    assert v["edge"] and v["level"] == "good"


def _skill_frames(n_dates=600, n_tickers=8, informative=True, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, [f"T{i}.AT" for i in range(n_tickers)]], names=["date", "ticker"])
    n = len(idx)
    base = np.array([0.3, 0.5, 0.2])
    if informative:
        z = rng.normal(size=n)
        logits = np.log(base)[None, :] + np.outer(z, [0.8, -0.8, 0.0])
        P = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    else:
        P = np.clip(base[None, :] + rng.normal(0, 0.08, size=(n, 3)), 0.02, None)
        P /= P.sum(axis=1, keepdims=True)
    truth = P if informative else np.repeat(base[None, :], n, axis=0)
    y = np.array([rng.choice(3, p=row) for row in truth], dtype="float64")
    preds = pd.DataFrame({"p_tp": P[:, 0], "p_sl": P[:, 1], "p_to": P[:, 2]}, index=idx)
    labs = pd.DataFrame({"label": y}, index=idx)
    return preds, labs


def test_skill_detects_real_signal_and_rejects_noise():
    preds, labs = _skill_frames(informative=True)
    sk = benchmarks.skill(preds, labs, n_boot=300)
    assert sk["ok"] and sk["logloss_model"] < sk["logloss_base"] and sk["p_value"] <= 0.05 and sk["improvement"] > 0
    preds, labs = _skill_frames(informative=False, seed=1)
    sk = benchmarks.skill(preds, labs, n_boot=300)
    assert not sk["ok"] and sk["logloss_model"] > sk["logloss_base"]
    # λίγα δεδομένα ⇒ κανένα συμπέρασμα (όχι σφάλμα)
    assert benchmarks.skill(preds.head(10), labs.head(10))["ok"] is False


def test_calibration_bins():
    rng = np.random.default_rng(0)
    n = 20000
    idx = pd.MultiIndex.from_arrays([pd.bdate_range("2020-01-01", periods=n), ["A.AT"] * n], names=["date", "ticker"])
    p = rng.uniform(0.05, 0.6, n)
    y = np.where(rng.uniform(size=n) < p, LABEL_TP, 1).astype(float)
    y[:100] = np.nan                                     # ανεπίλυτα: εκτός
    preds = pd.DataFrame({"p_tp": p}, index=idx)
    labs = pd.DataFrame({"label": y}, index=idx)
    cal = benchmarks.calibration(preds, labs, bins=10)
    assert len(cal) == 10
    assert sum(c["n"] for c in cal) == n - 100
    assert all(abs(c["n"] - (n - 100) / 10) <= 2 for c in cal)        # ίσο πλήθος ανά κάδο
    for c in cal:
        assert c["bin_lo"] <= c["predicted"] <= c["bin_hi"]
        assert abs(c["predicted"] - c["realized"]) < 0.06
    assert benchmarks.calibration(preds.iloc[:0], labs, bins=10) == []
