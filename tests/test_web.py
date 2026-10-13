"""Έλεγχοι του frontend (web/) χωρίς browser.

* Όλα τα αρχεία που αναφέρει το index.html υπάρχουν· κανένα εξωτερικό JS.
* Κάθε `import` των ES modules δείχνει σε υπαρκτό αρχείο και σε όνομα που εξάγεται πράγματι.
* Σύνταξη JS με `node --check` (αν υπάρχει node).
* Ο mock διακομιστής (web/dev/mock-server.py) τηρεί το συμβόλαιο HTTP του docs/ARCHITECTURE.md,
  ώστε το UI που αναπτύσσεται πάνω του να ταιριάζει με τον πραγματικό διακομιστή.
"""
from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
JS_FILES = sorted((WEB / "js").rglob("*.js"))

IMPORT_RE = re.compile(r"""import\s+(?P<what>[\s\S]*?)\s+from\s+['"](?P<path>[^'"]+)['"]""")
EXPORT_DECL_RE = re.compile(r"export\s+(?:async\s+)?(?:function\*?|const|let|var|class)\s+([A-Za-z_$][\w$]*)")
EXPORT_LIST_RE = re.compile(r"export\s*\{([^}]*)\}")


def exported_names(path: Path) -> set[str]:
    src = path.read_text(encoding="utf-8")
    names = set(EXPORT_DECL_RE.findall(src))
    for group in EXPORT_LIST_RE.findall(src):
        for part in group.split(","):
            part = part.strip()
            if part:
                names.add(part.split(" as ")[-1].strip())
    return names


def test_index_references_exist():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert '<html lang="el"' in html
    refs = re.findall(r"""(?:src|href)=["']([^"'#]+)["']""", html)
    local = [r for r in refs if not r.startswith(("http://", "https://", "data:"))]
    assert "css/app.css" in local and "js/app.js" in local and "favicon.svg" in local
    for r in local:
        assert (WEB / r).is_file(), f"λείπει το {r}"


def test_no_external_javascript():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"""<script[^>]+src=["']https?://""", html), "δεν επιτρέπεται εξωτερικό JS"
    for f in JS_FILES:
        for m in IMPORT_RE.finditer(f.read_text(encoding="utf-8")):
            assert m.group("path").startswith("."), f"{f.name}: μη τοπικό import {m.group('path')}"


@pytest.mark.parametrize("path", JS_FILES, ids=lambda p: str(p.relative_to(WEB)))
def test_js_imports_resolve(path: Path):
    src = path.read_text(encoding="utf-8")
    for m in IMPORT_RE.finditer(src):
        target = (path.parent / m.group("path")).resolve()
        assert target.is_file(), f"{path.name}: δεν υπάρχει το {m.group('path')}"
        what = m.group("what").strip()
        if what.startswith("*"):
            continue
        brace = re.search(r"\{([^}]*)\}", what)
        if not brace:
            continue
        exports = exported_names(target)
        for part in brace.group(1).split(","):
            name = part.strip().split(" as ")[0].strip()
            if name:
                assert name in exports, f"{path.name}: το {target.name} δεν εξάγει «{name}»"


@pytest.mark.skipif(shutil.which("node") is None, reason="δεν υπάρχει node")
@pytest.mark.parametrize("path", JS_FILES, ids=lambda p: str(p.relative_to(WEB)))
def test_js_syntax(path: Path):
    r = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr


def test_css_has_both_themes_and_reduced_motion():
    css = (WEB / "css" / "app.css").read_text(encoding="utf-8")
    assert ':root[data-theme="light"]' in css and ':root[data-theme="dark"]' in css
    assert "prefers-reduced-motion" in css
    assert "tabular-nums" in css


# ---------------------------------------------------------------------------
# Mock διακομιστής ↔ συμβόλαιο API
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mock_client():
    from fastapi.testclient import TestClient

    spec = importlib.util.spec_from_file_location("athex_orb_mock_server", WEB / "dev" / "mock-server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.OPTS.mode = "demo"
    return TestClient(mod.app)


ITEM_KEYS = {"ticker", "name", "sector", "custom", "has_data", "last_close", "change_1d", "change_5d", "spark",
             "adv20_eur", "orb", "decision", "favorite"}
ORB_KEYS = {"p_tp", "p_sl", "p_to", "ev_net", "dominant"}


def test_mock_status_and_universe(mock_client):
    s = mock_client.get("/api/status").json()
    assert {"mode", "data_last_date", "data_updated_at", "model_trained_until", "n_tickers", "n_with_data",
            "horizon_days", "k_tp", "k_sl", "jobs", "disclaimer_el"} <= set(s)
    u = mock_client.get("/api/universe").json()
    assert set(u) == {"as_of", "items"} and u["items"]
    for it in u["items"]:
        assert ITEM_KEYS <= set(it)
        assert it["decision"] in {"take", "skip", "blocked", "none"}
        if it["orb"] is not None:
            assert ORB_KEYS <= set(it["orb"]) and it["orb"]["dominant"] in {"tp", "sl", "to"}
            assert abs(it["orb"]["p_tp"] + it["orb"]["p_sl"] + it["orb"]["p_to"] - 1) < 1e-3


def test_mock_search_favorites_and_add(mock_client):
    r = mock_client.get("/api/search", params={"q": "οπαπ"}).json()
    assert r["items"][0]["ticker"] == "OPAP.AT"
    r = mock_client.get("/api/search", params={"q": "zzzz"}).json()
    assert r["items"] == [] and r["suggest"] == {"ticker": "ZZZZ.AT", "valid": True}
    assert mock_client.put("/api/favorites/ALPHA.AT").status_code == 200
    assert "ALPHA.AT" in mock_client.get("/api/favorites").json()["tickers"]
    mock_client.delete("/api/favorites/ALPHA.AT")
    assert "ALPHA.AT" not in mock_client.get("/api/favorites").json()["tickers"]
    added = mock_client.post("/api/universe", json={"ticker": "intek"}).json()
    assert added["ok"] and added["item"]["ticker"] == "INTEK.AT" and added["item"]["custom"]
    assert mock_client.post("/api/universe", json={"ticker": "intek"}).json()["ok"] is False
    assert mock_client.delete("/api/universe/INTEK.AT").status_code == 200


def test_mock_stock_payload(mock_client):
    d = mock_client.get("/api/stock/OPAP.AT").json()
    assert {"ticker", "name", "sector", "custom", "favorite", "has_data", "as_of", "candles", "orb_history", "orb",
            "plan", "decision", "decision_reason_el", "agents", "outcomes"} <= set(d)
    assert {"d", "o", "h", "l", "c", "v"} <= set(d["candles"][0])
    assert {"entry_ref", "tp_price", "sl_price", "horizon_days", "size_eur", "shares"} <= set(d["plan"])
    ids = [a["id"] for a in d["agents"]]
    assert ids == ["scout", "market", "judge", "quant", "guard", "executor"]
    for a in d["agents"]:
        assert {"id", "name_el", "role_el", "status", "headline_el", "metrics", "notes_el"} <= set(a)
        assert a["status"] in {"ok", "warn", "block", "info"}
        for m in a["metrics"]:
            assert m["fmt"] in {"pct", "eur", "num", "int", "text", "date"}
    guard = d["agents"][4]
    assert {"id", "passed", "label_el", "detail_el", "blocking"} <= set(guard["checks"][0])
    assert {"feature", "label_el", "contribution"} <= set(d["agents"][2]["drivers"][0])
    nodata = mock_client.get("/api/stock/EYAPS.AT").json()
    assert nodata["has_data"] is False and nodata["orb"] is None and nodata["candles"] == []
    assert mock_client.get("/api/stock/NOPE.AT").status_code == 404


def test_mock_signals_backtest_journal_settings_jobs(mock_client):
    sig = mock_client.get("/api/signals").json()
    keys = {"ticker", "name", "sector", "decision", "reason_el", "p_tp", "p_sl", "p_to", "ev_net", "entry_ref",
            "tp_price", "sl_price", "size_eur"}
    assert all(keys <= set(i) for i in sig["items"])
    bt = mock_client.get("/api/backtest").json()
    assert {"available", "generated_at", "period", "params", "metrics", "equity", "drawdown", "trades",
            "calibration", "by_label", "verdict"} <= set(bt)
    assert set(bt["metrics"]) == {"strategy", "buy_hold", "buy_hold_universe", "random_median"}
    assert {"d", "strategy", "buy_hold", "buy_hold_universe", "random_median", "random_p05",
            "random_p95"} <= set(bt["equity"][0])
    assert {"bin_lo", "bin_hi", "predicted", "realized", "n"} <= set(bt["calibration"][0])
    assert {"edge", "percentile", "text_el", "level"} <= set(bt["verdict"])
    j = mock_client.get("/api/journal").json()
    assert set(j) == {"items", "summary"}
    st = mock_client.get("/api/settings").json()
    assert {"portfolio", "barriers", "guard", "costs", "model"} <= set(st)
    r = mock_client.put("/api/settings", json={"barriers": {"k_tp": 2.5}}).json()
    assert r["ok"] and r["retrain_needed"] is True
    mock_client.put("/api/settings", json={"barriers": {"k_tp": 2.0}})
    jid = mock_client.post("/api/jobs", json={"type": "backtest"}).json()["job_id"]
    assert mock_client.post("/api/jobs", json={"type": "backtest"}).status_code == 409
    job = mock_client.get(f"/api/jobs/{jid}").json()
    assert {"id", "type", "status", "progress", "message_el", "started_at", "finished_at", "error"} <= set(job)
