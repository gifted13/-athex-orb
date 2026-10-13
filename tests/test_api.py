"""HTTP API (FastAPI TestClient) πάνω σε πραγματικό pipeline επίδειξης σε προσωρινό φάκελο δεδομένων.

Το demo τρέχει μία φορά ανά module με σύντομο ιστορικό (3 έτη), μικρότερο ελάχιστο ιστορικό εκπαίδευσης
και 20 τυχαίες στρατηγικές, ώστε να είναι γρήγορο. Ελέγχονται τα κλειδιά του συμβολαίου HTTP του
ARCHITECTURE.md (αυτά που περιμένει το UI), η αναζήτηση, τα αγαπημένα, η προσθήκη μετοχής, οι ρυθμίσεις
και ο κύκλος ζωής των εργασιών.
"""
from __future__ import annotations

import copy
import json
import threading
import time
import warnings

import pytest

from athex_orb import cli, pipeline
from athex_orb.config import load_config
from athex_orb.universe import DEFAULT_UNIVERSE

ITEM_KEYS = {"ticker", "name", "sector", "custom", "has_data", "last_close", "change_1d", "change_5d", "spark",
             "adv20_eur", "orb", "decision", "favorite"}
ORB_KEYS = {"p_tp", "p_sl", "p_to", "ev_net", "dominant"}
STATUS_KEYS = {"mode", "data_last_date", "data_updated_at", "model_trained_until", "n_tickers", "n_with_data",
               "horizon_days", "k_tp", "k_sl", "jobs", "disclaimer_el"}
STOCK_KEYS = {"ticker", "name", "sector", "custom", "favorite", "has_data", "as_of", "candles", "orb_history", "orb",
              "plan", "decision", "decision_reason_el", "agents", "outcomes"}
AGENT_KEYS = {"id", "name_el", "role_el", "status", "headline_el", "metrics", "notes_el"}
AGENT_IDS = ["scout", "market", "judge", "quant", "guard", "executor"]
SIGNAL_KEYS = {"ticker", "name", "sector", "decision", "reason_el", "p_tp", "p_sl", "p_to", "ev_net", "entry_ref",
               "tp_price", "sl_price", "size_eur"}
JOB_KEYS = {"id", "type", "status", "progress", "message_el", "started_at", "finished_at", "error"}
METRIC_KEYS = {"total_return", "cagr", "max_drawdown", "sharpe", "volatility", "n_trades", "win_rate", "avg_win",
               "avg_loss", "profit_factor", "avg_bars_held", "total_costs_eur", "exposure"}
DECISIONS = {"take", "skip", "blocked", "none"}
N_STOCKS = len(DEFAULT_UNIVERSE)


def _test_client_cls():
    # η εισαγωγή του starlette.testclient βγάζει DeprecationWarning (anyio) που δεν αφορά τον κώδικά μας
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from fastapi.testclient import TestClient
    return TestClient


def _fast_cfg(data_dir):
    cfg = load_config(data_dir=data_dir, use_user_settings=False)
    cfg.data.history_years = 3
    cfg.model.min_train_days = 250
    cfg.model.retrain_every_days = 63
    cfg.backtest.random_runs = 20
    return cfg


@pytest.fixture(scope="module")
def api_cfg(tmp_path_factory):
    cfg = _fast_cfg(tmp_path_factory.mktemp("api_data"))
    t0 = time.perf_counter()
    pipeline.run_demo(cfg)
    assert time.perf_counter() - t0 < 60
    return cfg


@pytest.fixture(scope="module")
def client(api_cfg):
    from athex_orb.server import create_app

    return _test_client_cls()(create_app(api_cfg))


@pytest.fixture
def empty_client(tmp_path):
    from athex_orb.server import create_app

    return _test_client_cls()(create_app(_fast_cfg(tmp_path)))


def _wait_job(client, jid, timeout=60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.05)
    raise AssertionError(f"η εργασία {jid} δεν τελείωσε")


# ---------------------------------------------------------------- χωρίς δεδομένα (onboarding)

def test_empty_state_answers_gracefully(empty_client):
    c = empty_client
    s = c.get("/api/status").json()
    assert STATUS_KEYS <= set(s) and s["mode"] == "empty" and s["n_with_data"] == 0 and s["jobs"] == []
    assert s["data_last_date"] is None and s["model_trained_until"] is None
    u = c.get("/api/universe").json()
    assert u["as_of"] is None and len(u["items"]) == N_STOCKS
    for it in u["items"]:
        assert ITEM_KEYS <= set(it)
        assert it["has_data"] is False and it["orb"] is None and it["decision"] == "none" and it["spark"] == []
    assert c.get("/api/signals").json()["items"] == []
    assert c.get("/api/backtest").json()["available"] is False
    j = c.get("/api/journal").json()
    assert j["items"] == [] and j["summary"]["total"] == 0
    d = c.get("/api/stock/OPAP.AT").json()
    assert STOCK_KEYS <= set(d) and d["has_data"] is False and d["candles"] == [] and d["orb"] is None
    assert [a["id"] for a in d["agents"]] == AGENT_IDS
    assert c.get("/api/search", params={"q": "οπαπ"}).json()["items"][0]["ticker"] == "OPAP.AT"
    assert c.get("/api/jobs").json() == {"items": []}
    # χωρίς δεδομένα η προσθήκη μετοχής απλώς τη βάζει στη λίστα
    r = c.post("/api/universe", json={"ticker": "intek"}).json()
    assert r["ok"] and r["item"]["ticker"] == "INTEK.AT" and r["item"]["has_data"] is False and r["note_el"]
    assert "INTEK.AT" in [i["ticker"] for i in c.get("/api/universe").json()["items"]]
    assert c.post("/api/jobs", json={"type": "backtest"}).status_code == 200


def test_backtest_job_without_state_reports_greek_error(empty_client):
    jid = empty_client.post("/api/jobs", json={"type": "backtest"}).json()["job_id"]
    j = _wait_job(empty_client, jid)
    assert j["status"] == "error" and "Ενημέρωση" in j["error"]


# ---------------------------------------------------------------- μετά το demo

def test_status(client):
    s = client.get("/api/status").json()
    assert STATUS_KEYS <= set(s)
    assert s["mode"] == "demo" and s["n_tickers"] == N_STOCKS and s["n_with_data"] == N_STOCKS
    assert s["data_last_date"] and s["model_trained_until"] and s["horizon_days"] == 10
    assert s["k_tp"] == 2.0 and s["k_sl"] == 1.0 and isinstance(s["jobs"], list)
    assert "επενδυτική συμβουλή" in s["disclaimer_el"]
    assert s["retrain_needed"] is False and s["backtest_available"] is True
    # κλίμα αγοράς της τελευταίας συνεδρίασης (το UI δείχνει «Καθεστώς: …» στην Αγορά)
    assert s["regime"] in {"bull", "bear", "neutral"} and s["regime_el"].endswith("αγορά")
    assert s["market"]["regime"] == s["regime"] and "mkt_drawdown_10" in s["market"]


def test_universe_items(client):
    u = client.get("/api/universe").json()
    assert set(u) >= {"as_of", "items"} and u["as_of"]
    assert [i["ticker"] for i in u["items"]] == [s.ticker for s in DEFAULT_UNIVERSE]
    n_orb = 0
    for it in u["items"]:
        assert ITEM_KEYS <= set(it)
        assert it["decision"] in DECISIONS and it["has_data"] is True
        assert len(it["spark"]) == 60 and it["last_close"] == it["spark"][-1]
        assert it["adv20_eur"] > 0
        if it["orb"] is not None:
            n_orb += 1
            assert ORB_KEYS <= set(it["orb"]) and it["orb"]["dominant"] in {"tp", "sl", "to"}
            assert abs(it["orb"]["p_tp"] + it["orb"]["p_sl"] + it["orb"]["p_to"] - 1) < 1e-3
    assert n_orb == N_STOCKS


def test_stock_payload(client):
    d = client.get("/api/stock/OPAP.AT").json()
    assert STOCK_KEYS <= set(d)
    assert d["has_data"] is True and d["ticker"] == "OPAP.AT" and d["as_of"]
    assert len(d["candles"]) == 250 and {"d", "o", "h", "l", "c", "v"} <= set(d["candles"][0])
    assert d["candles"][-1]["d"] == d["as_of"]
    assert d["orb_history"] and {"d", "p_tp", "p_sl", "p_to"} <= set(d["orb_history"][0])
    last = d["orb_history"][-1]
    assert last["d"] == d["as_of"] and abs(last["p_tp"] - d["orb"]["p_tp"]) < 1e-3   # ίδια με τη Σφαίρα
    assert ORB_KEYS <= set(d["orb"])
    assert {"entry_ref", "tp_price", "sl_price", "horizon_days", "size_eur", "shares"} <= set(d["plan"])
    assert d["plan"]["tp_price"] > d["plan"]["entry_ref"] > d["plan"]["sl_price"]
    assert d["decision"] in DECISIONS and d["decision_reason_el"]
    assert [a["id"] for a in d["agents"]] == AGENT_IDS
    for a in d["agents"]:
        assert AGENT_KEYS <= set(a) and a["status"] in {"ok", "warn", "block", "info"} and a["headline_el"]
        for m in a["metrics"]:
            assert m["fmt"] in {"pct", "eur", "num", "int", "text", "date"} and m["label_el"]
    guard = d["agents"][4]
    assert guard["checks"] and {"id", "passed", "label_el", "detail_el", "blocking"} <= set(guard["checks"][0])
    judge = d["agents"][2]
    assert judge["drivers"] and {"feature", "label_el", "contribution"} <= set(judge["drivers"][0])
    assert d["agents"][0]["headline_el"].startswith("Ρευστότητα")
    assert d["outcomes"] and len(d["outcomes"]) <= 20
    assert {"signal_date", "label", "gross_ret"} <= set(d["outcomes"][0])
    sig = [o["signal_date"] for o in d["outcomes"]]
    assert sig == sorted(sig)
    # days, πεζά/χωρίς κατάληξη, 404
    assert len(client.get("/api/stock/OPAP.AT", params={"days": 60}).json()["candles"]) == 60
    assert client.get("/api/stock/opap").json()["ticker"] == "OPAP.AT"
    r = client.get("/api/stock/NOPE.AT")
    assert r.status_code == 404 and "δεν βρέθηκε" in r.json()["detail"]


def test_signals(client):
    sig = client.get("/api/signals").json()
    assert sig["as_of"] and len(sig["items"]) == N_STOCKS
    for i in sig["items"]:
        assert SIGNAL_KEYS <= set(i) and i["decision"] in DECISIONS and i["reason_el"]
        if i["decision"] == "take":
            assert i["size_eur"] >= 300 and i["tp_price"] > i["entry_ref"] > i["sl_price"]
        else:
            assert i["size_eur"] is None
    order = {"take": 0, "skip": 1, "blocked": 2, "none": 3}
    keys = [(order[i["decision"]], -(i["ev_net"] if i["ev_net"] is not None else -1e9)) for i in sig["items"]]
    assert keys == sorted(keys)


def test_backtest(client):
    bt = client.get("/api/backtest").json()
    assert {"available", "generated_at", "period", "params", "metrics", "equity", "drawdown", "trades",
            "calibration", "by_label", "verdict"} <= set(bt)
    assert bt["available"] is True and bt["stale"] is False and bt["mode"] == "demo"
    assert set(bt["metrics"]) == {"strategy", "buy_hold", "buy_hold_universe", "random_median"}
    for m in bt["metrics"].values():
        assert METRIC_KEYS <= set(m)
    assert {"d", "strategy", "buy_hold", "buy_hold_universe", "random_median", "random_p05",
            "random_p95"} <= set(bt["equity"][0])
    # έλεγχος δεξιότητας εκτός δείγματος + 20 τυχαίες εκτελέσεις είναι λίγες για ετυμηγορία
    assert {"logloss_model", "logloss_base", "p_value", "ok"} <= set(bt["skill"])
    assert bt["verdict"]["level"] == "none" and "τουλάχιστον 100" in bt["verdict"]["text_el"]
    assert bt["equity"][0]["d"] == bt["period"]["start"] and bt["equity"][-1]["d"] == bt["period"]["end"]
    assert {"d", "strategy"} <= set(bt["drawdown"][0]) and all(x["strategy"] <= 1e-9 for x in bt["drawdown"])
    assert len(bt["trades"]) <= 300 and bt["trades"]
    assert {"ticker", "signal_date", "entry_date", "exit_date", "label", "net_ret", "cost_eur"} <= set(bt["trades"][0])
    assert bt["calibration"] and {"bin_lo", "bin_hi", "predicted", "realized", "n"} <= set(bt["calibration"][0])
    assert set(bt["by_label"]) >= {"tp", "sl", "to"}
    assert {"edge", "percentile", "text_el", "level"} <= set(bt["verdict"])
    assert bt["verdict"]["level"] in {"good", "weak", "none"} and bt["random"]["runs"] == 20
    assert bt["metrics"]["random_median"]["exposure"] is None
    # η διάμεσος των τυχαίων πληρώνει κι αυτή κόστη (όχι «0 €») και ο δείκτης δεν έχει «συναλλαγές»
    rm = bt["metrics"]["random_median"]
    assert rm["n_trades"] > 0 and rm["total_costs_eur"] > 0 and 0.0 <= rm["win_rate"] <= 1.0
    assert bt["metrics"]["buy_hold"]["n_trades"] is None
    assert any(n.startswith("ΕΠΙΔΕΙΞΗ") for n in bt["notes_el"]) and bt["random"]["method_el"]


def test_journal(client):
    j = client.get("/api/journal").json()
    assert set(j) >= {"items", "summary"} and j["items"]
    assert len(j["items"]) == j["summary"]["total"] <= N_STOCKS
    for e in j["items"]:
        assert e["decision"] in ("take", "skip") and e["status"] == "pending" and e["name"] and e["mode"] == "demo"
    s = j["summary"]
    assert {"total", "take", "skip", "pending", "open"} <= set(s)
    assert {"n", "resolved", "hit_rate", "avg_net_ret"} <= set(s["take"])
    n_take = sum(1 for i in client.get("/api/signals").json()["items"] if i["decision"] == "take")
    assert s["take"]["n"] == n_take


def test_favorites_add_remove_persist(client, api_cfg):
    from athex_orb.server import create_app

    assert client.get("/api/favorites").json() == {"tickers": []}
    assert client.put("/api/favorites/OPAP.AT").json() == {"tickers": ["OPAP.AT"]}
    assert client.put("/api/favorites/ppc").json() == {"tickers": ["OPAP.AT", "PPC.AT"]}       # κανονικοποίηση
    assert client.put("/api/favorites/OPAP.AT").json() == {"tickers": ["OPAP.AT", "PPC.AT"]}   # χωρίς διπλό
    assert client.put("/api/favorites/ETE.AT").status_code == 200
    r = client.put("/api/favorites/NOPE.AT")
    assert r.status_code == 404 and "δεν υπάρχει" in r.json()["detail"]
    # επιμονή: νέο στιγμιότυπο εφαρμογής, ίδια σειρά
    other = _test_client_cls()(create_app(api_cfg))
    assert other.get("/api/favorites").json()["tickers"] == ["OPAP.AT", "PPC.AT", "ETE.AT"]
    saved = json.loads((api_cfg.data_dir / "favorites.json").read_text(encoding="utf-8"))
    assert saved["tickers"] == ["OPAP.AT", "PPC.AT", "ETE.AT"]
    items = {i["ticker"]: i for i in client.get("/api/universe").json()["items"]}
    assert items["OPAP.AT"]["favorite"] and not items["MOH.AT"]["favorite"]
    assert client.get("/api/stock/PPC.AT").json()["favorite"] is True
    assert client.delete("/api/favorites/PPC.AT").json() == {"tickers": ["OPAP.AT", "ETE.AT"]}
    assert client.delete("/api/favorites/PPC.AT").status_code == 200                        # ήδη αφαιρεμένη
    assert other.get("/api/favorites").json()["tickers"] == ["OPAP.AT", "ETE.AT"]
    for t in ("OPAP.AT", "ETE.AT"):
        client.delete(f"/api/favorites/{t}")
    assert client.get("/api/favorites").json() == {"tickers": []}


@pytest.mark.parametrize("q", ["οπαπ", "ΟΠΑΠ", "Οπάπ", "opap", "OPAP.AT", "allwyn"])
def test_search_accent_insensitive(client, q):
    r = client.get("/api/search", params={"q": q}).json()
    assert r["items"] and r["items"][0]["ticker"] == "OPAP.AT"
    assert ITEM_KEYS <= set(r["items"][0])
    assert r["suggest"] is None


def test_search_other_cases(client):
    banks = {i["ticker"] for i in client.get("/api/search", params={"q": "τραπεζα"}).json()["items"]}
    assert {"ETE.AT", "TPEIR.AT"} <= banks
    assert client.get("/api/search", params={"q": "δεη"}).json()["items"][0]["ticker"] == "PPC.AT"
    r = client.get("/api/search", params={"q": "zzzz"}).json()
    assert r["items"] == [] and r["suggest"] == {"ticker": "ZZZZ.AT", "valid": True}
    r = client.get("/api/search", params={"q": "αβγδ"}).json()      # ελληνικά: όχι πρόταση συμβόλου
    assert r["items"] == [] and r["suggest"] is None
    assert len(client.get("/api/search", params={"q": ""}).json()["items"]) == 20


def test_add_and_remove_custom_ticker_demo(client, api_cfg):
    r = client.post("/api/universe", json={"ticker": "intek", "name": "Intek", "sector": "Κατασκευές"}).json()
    assert r["ok"] is True, r
    it = r["item"]
    assert ITEM_KEYS <= set(it) and it["ticker"] == "INTEK.AT" and it["custom"] is True
    assert it["has_data"] is True and len(it["spark"]) == 60 and it["name"] == "Intek"
    assert it["orb"] is not None and it["decision"] in DECISIONS          # υπολογίστηκε με το αποθηκευμένο μοντέλο
    again = client.post("/api/universe", json={"ticker": "INTEK.AT"}).json()
    assert again["ok"] is False and "υπάρχει ήδη" in again["error_el"]
    bad = client.post("/api/universe", json={"ticker": "!!"}).json()
    assert bad["ok"] is False and bad["error_el"]
    assert client.post("/api/universe", json={}).json()["ok"] is False
    tick = [i["ticker"] for i in client.get("/api/universe").json()["items"]]
    assert tick[-1] == "INTEK.AT" and len(tick) == N_STOCKS + 1
    d = client.get("/api/stock/INTEK.AT").json()
    assert d["has_data"] is True and d["custom"] is True and d["candles"] and d["orb"] is not None
    assert [a["id"] for a in d["agents"]] == AGENT_IDS
    assert "INTEK.AT" in {i["ticker"] for i in client.get("/api/signals").json()["items"]}
    assert client.get("/api/search", params={"q": "intek"}).json()["items"][0]["ticker"] == "INTEK.AT"
    # αφαίρεση: μόνο μετοχές χρήστη
    client.put("/api/favorites/INTEK.AT")
    r = client.delete("/api/universe/OPAP.AT")
    assert r.status_code == 404 and "προσθέσατε" in r.json()["detail"]
    assert client.delete("/api/universe/INTEK.AT").status_code == 200
    assert "INTEK.AT" not in [i["ticker"] for i in client.get("/api/universe").json()["items"]]
    assert "INTEK.AT" not in client.get("/api/favorites").json()["tickers"]
    assert client.get("/api/stock/INTEK.AT").status_code == 404
    assert "INTEK.AT" not in {i["ticker"] for i in client.get("/api/signals").json()["items"]}


def test_settings_get_and_validation(client, api_cfg):
    st = client.get("/api/settings").json()
    assert set(st) == {"portfolio", "barriers", "guard", "costs", "model"}
    assert st["model"] == {"kind": "logreg"} and st["barriers"]["k_tp"] == 2.0
    before = copy.deepcopy(st)
    bad = [
        {"portfolio": {"max_positions": 0}},
        {"portfolio": {"max_positions": 2.5}},
        {"barriers": {"k_tp": "abc"}},
        {"barriers": {"horizon_days": True}},
        {"guard": {"min_ev_net": 5}},
        {"costs": {"commission_pct": -0.01}},
        {"model": {"kind": "xyz"}},
        {"portfolio": {"nope": 1}},
        {"nosection": {"x": 1}},
        {"guard": 3},
    ]
    for body in bad:
        r = client.put("/api/settings", json=body)
        assert r.status_code == 422, body
        detail = r.json()["detail"]
        assert isinstance(detail, str) and any("Α" <= ch <= "ω" for ch in detail), detail
    assert client.put("/api/settings", json=[1, 2]).status_code == 422
    assert client.get("/api/settings").json() == before                  # τίποτα δεν αποθηκεύτηκε
    assert not (api_cfg.data_dir / "settings.json").exists()


def test_settings_save_recompute_and_retrain_flag(client, api_cfg):
    r = client.put("/api/settings", json={"guard": {"min_ev_net": -0.05}, "portfolio": {"capital_eur": 12500.5}})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] and body["retrain_needed"] is False and body["signals_updated"] is True
    assert set(body["changed"]) == {"guard.min_ev_net", "portfolio.capital_eur"}
    assert body["settings"]["portfolio"]["capital_eur"] == 12500.5
    saved = json.loads((api_cfg.data_dir / "settings.json").read_text(encoding="utf-8"))
    assert saved["guard"]["min_ev_net"] == -0.05
    # με χαμηλό όριο EV υπάρχουν αγορές· τα σήματα ξαναϋπολογίστηκαν αμέσως
    sig = client.get("/api/signals").json()["items"]
    assert any(i["decision"] == "take" for i in sig)
    assert sum(1 for i in sig if i["decision"] == "take") <= api_cfg.portfolio.max_positions
    assert client.get("/api/status").json()["settings_changed"] is False
    assert client.get("/api/backtest").json()["stale"] is True            # οι ρυθμίσεις άλλαξαν μετά το backtest
    # αλλαγή ορίων ⇒ χρειάζεται νέα εκπαίδευση· τα σήματα μένουν συνεπή με το εκπαιδευμένο μοντέλο
    r = client.put("/api/settings", json={"barriers": {"k_tp": 2.5}}).json()
    assert r["retrain_needed"] is True and r["changed"] == ["barriers.k_tp"]
    s = client.get("/api/status").json()
    assert s["retrain_needed"] is True and s["k_tp"] == 2.0
    # επαναφορά
    r = client.put("/api/settings", json={"barriers": {"k_tp": 2.0}, "guard": {"min_ev_net": 0.003},
                                          "portfolio": {"capital_eur": 10000}}).json()
    assert r["retrain_needed"] is False
    assert client.get("/api/status").json()["retrain_needed"] is False


def test_jobs_lifecycle(client, monkeypatch):
    gate = threading.Event()
    started = threading.Event()

    def slow_backtest(cfg, progress=None):
        started.set()
        progress(0.5, "Μισή δουλειά…")
        assert gate.wait(10)
        return {"verdict": {"level": "none"}, "n_trades": 0}

    monkeypatch.setattr(pipeline, "run_backtest_job", slow_backtest)
    r = client.post("/api/jobs", json={"type": "backtest"})
    assert r.status_code == 200
    jid = r.json()["job_id"]
    assert started.wait(10)
    job = client.get(f"/api/jobs/{jid}").json()
    assert JOB_KEYS <= set(job) and job["type"] == "backtest" and job["status"] == "running"
    assert job["progress"] == 0.5 and job["message_el"] == "Μισή δουλειά…" and job["started_at"]
    conflict = client.post("/api/jobs", json={"type": "backtest"})
    assert conflict.status_code == 409 and "εκτελείται ήδη" in conflict.json()["detail"]
    assert jid in [j["id"] for j in client.get("/api/status").json()["jobs"]]
    assert client.get("/api/jobs").json()["items"][0]["id"] == jid
    gate.set()
    done = _wait_job(client, jid)
    assert done["status"] == "done" and done["progress"] == 1.0 and done["finished_at"] and done["error"] is None
    assert client.get("/api/status").json()["jobs"] == []

    def broken_refresh(cfg, progress=None):
        raise pipeline.PipelineError("Δοκιμαστικό σφάλμα λήψης.")

    monkeypatch.setattr(pipeline, "run_refresh", broken_refresh)
    jid = client.post("/api/jobs", json={"type": "refresh"}).json()["job_id"]
    err = _wait_job(client, jid)
    assert err["status"] == "error" and err["error"] == "Δοκιμαστικό σφάλμα λήψης."

    assert client.post("/api/jobs", json={"type": "nope"}).status_code == 400
    r = client.get("/api/jobs/doesnotexist")
    assert r.status_code == 404 and r.json()["detail"] == "Η εργασία δεν βρέθηκε."


def test_real_backtest_job_rewrites_state(client):
    before = client.get("/api/backtest").json()["generated_at"]
    time.sleep(1.1)                                  # generated_at έχει ακρίβεια δευτερολέπτου
    jid = client.post("/api/jobs", json={"type": "backtest"}).json()["job_id"]
    j = _wait_job(client, jid)
    assert j["status"] == "done", j
    assert j["result"]["verdict"]["level"] in {"good", "weak", "none"}
    bt = client.get("/api/backtest").json()
    assert bt["available"] and bt["generated_at"] != before and bt["stale"] is False


def test_static_ui_and_unknown_api(client):
    r = client.get("/")
    assert r.status_code == 200 and '<html lang="el"' in r.text
    assert client.get("/js/app.js").status_code == 200
    r = client.get("/some/deep/route")
    assert r.status_code == 200 and '<html lang="el"' in r.text           # επιστροφή στο index.html
    assert client.get("/js/does-not-exist.js").status_code == 404                # όχι HTML στη θέση αρχείου
    assert client.get("/dev/mock-server.py").status_code == 404                  # εργαλεία ανάπτυξης: όχι
    assert client.get("/dev/").status_code == 404
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.json() == {"detail": "Άγνωστο endpoint."}
    assert client.get("/api/status").headers["cache-control"] == "no-store"
    r = client.post("/api/universe", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422 and "Μη έγκυρο αίτημα" in r.json()["detail"]


def test_cli_reads_state(api_cfg, capsys):
    assert cli.main(["--data-dir", str(api_cfg.data_dir), "signals", "--all"]) == 0
    out = capsys.readouterr().out
    assert "Σήματα της" in out and "OPAP.AT" in out
    assert cli.main(["--data-dir", str(api_cfg.data_dir), "journal", "--limit", "5"]) == 0
    assert "Ημερολόγιο" in capsys.readouterr().out
    args = cli.build_parser().parse_args(["serve", "--port", "8123"])
    assert args.func is cli.cmd_serve and args.port == 8123
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["unknown"])


def test_add_is_refused_while_a_refresh_runs(client):
    jobs = client.app.state.jobs
    with jobs._lock:
        jobs._jobs["fake"] = {"id": "fake", "type": "refresh", "status": "running", "progress": 0.4}
    try:
        r = client.post("/api/universe", json={"ticker": "INLOT"})
        assert r.status_code == 409 and "περιμένετε" in r.json()["detail"]
        assert "INLOT.AT" not in {i["ticker"] for i in client.get("/api/universe").json()["items"]}
    finally:
        with jobs._lock:
            jobs._jobs.pop("fake")


def test_index_ticker_is_not_offered_or_added(client, api_cfg):
    r = client.get("/api/search", params={"q": "GD"}).json()
    assert r["suggest"] is None
    r = client.post("/api/universe", json={"ticker": "GD"}).json()
    assert r["ok"] is False and "Γενικός Δείκτης" in r["error_el"]


@pytest.mark.parametrize("content", ['{"tickers": [["OPAP.AT"]]}', '{"tickers": [{"t": 1}, "OPAP.AT"]}',
                                     '[["x"], "OPAP.AT"]', '{"tickers": "OPAP.AT"}', "σκουπίδια"])
def test_corrupt_favorites_file_never_breaks_the_api(client, api_cfg, content):
    path = api_cfg.data_dir / "favorites.json"
    old = path.read_text(encoding="utf-8") if path.exists() else None
    path.write_text(content, encoding="utf-8")
    try:
        for url in ("/api/universe", "/api/favorites", "/api/stock/OPAP.AT", "/api/signals"):
            assert client.get(url).status_code == 200, url
        r = client.put("/api/favorites/OPAP.AT")
        assert r.status_code == 200 and "OPAP.AT" in r.json()["tickers"]
    finally:
        if old is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(old, encoding="utf-8")


def test_settings_bounds_are_formatted_in_greek(client):
    r = client.put("/api/settings", json={"portfolio": {"capital_eur": -1}})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "1.000.000.000 €" in detail and "e+" not in detail
