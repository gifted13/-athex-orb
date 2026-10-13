"""Ημερολόγιο εικονικών συναλλαγών: idempotent καταγραφή, επίλυση ίδια με τις ετικέτες, σύνοψη."""
from __future__ import annotations

import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
import pytest

from athex_orb import journal
from athex_orb.labels import LABEL_SL, LABEL_TO, LABEL_TP

STATUS_OF = {LABEL_TP: "tp", LABEL_SL: "sl", LABEL_TO: "timeout"}


def _row(ticker, decision="skip", p_tp=0.4, tp_ret=0.04, sl_ret=0.02, **kw):
    r = {"ticker": ticker, "sector": "Τράπεζες", "decision": decision, "reason_el": "δοκιμή", "p_tp": p_tp,
         "p_sl": 0.4, "p_to": 0.2, "ev_net": 0.001, "entry_ref": 10.0, "tp_ret": tp_ret, "sl_ret": sl_ret,
         "size_eur": 1500.0 if decision == "take" else None, "horizon_days": 10, "cost_rt": 0.006}
    r.update(kw)
    return r


def _count(cfg) -> int:
    with closing(sqlite3.connect(journal.journal_path(cfg))) as con:
        return con.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]


EVENING = pd.Timestamp("2024-03-01 19:00", tz="Europe/Athens")       # μετά το κλείσιμο, πριν την επόμενη συνεδρίαση


def test_record_is_idempotent_and_updates_pending(cfg):
    rows = [_row("ETE.AT", "take"), _row("PPC.AT", "skip")]
    assert journal.record(cfg, "2024-03-01", rows, now=EVENING) == 2
    assert journal.record(cfg, "2024-03-01", rows, now=EVENING) == 2          # ενημέρωση, όχι διπλή εγγραφή
    assert _count(cfg) == 2
    # νέα ενημέρωση την ίδια ημέρα με άλλη απόφαση: όσο είναι «σε αναμονή» η εγγραφή αλλάζει
    journal.record(cfg, pd.Timestamp("2024-03-01"), [_row("PPC.AT", "take", p_tp=0.55)], now=EVENING)
    assert _count(cfg) == 2
    e = {x["ticker"]: x for x in journal.list_entries(cfg)}
    assert e["PPC.AT"]["decision"] == "take" and e["PPC.AT"]["p_tp"] == pytest.approx(0.55)
    assert e["PPC.AT"]["size_eur"] == pytest.approx(1500.0)
    assert e["ETE.AT"]["status"] == "pending" and e["ETE.AT"]["mode"] == "live"
    # η απόρριψη δεν κρατά ποσό θέσης
    journal.record(cfg, "2024-03-01", [_row("ETE.AT", "skip", size_eur=999.0)], now=EVENING)
    assert {x["ticker"]: x for x in journal.list_entries(cfg)}["ETE.AT"]["size_eur"] is None
    # νέα ημέρα ⇒ νέα εγγραφή
    journal.record(cfg, "2024-03-04", rows)
    assert _count(cfg) == 4
    # άκυρες γραμμές αγνοούνται
    assert journal.record(cfg, "2024-03-04", [{"ticker": "", "decision": "take"}, {"ticker": "X.AT"}]) == 0
    with pytest.raises(ValueError):
        journal.record(cfg, None, rows)


def test_list_order_takes_first(cfg):
    journal.record(cfg, "2024-03-01", [_row("A.AT", "skip", ev_net=0.02), _row("B.AT", "take", ev_net=0.001),
                                       _row("C.AT", "skip", ev_net=0.03)])
    journal.record(cfg, "2024-03-04", [_row("D.AT", "skip")])
    order = [e["ticker"] for e in journal.list_entries(cfg)]
    assert order == ["D.AT", "B.AT", "C.AT", "A.AT"]
    assert len(journal.list_entries(cfg, limit=2)) == 2
    assert [e["ticker"] for e in journal.list_entries(cfg, ticker="C.AT")] == ["C.AT"]


def test_modes_are_kept_apart(cfg):
    journal.record(cfg, "2024-03-01", [_row("ETE.AT", "take")], mode="demo")
    journal.record(cfg, "2024-03-01", [_row("ETE.AT", "skip")], mode="live")
    assert _count(cfg) == 2
    assert [e["decision"] for e in journal.list_entries(cfg, mode="demo")] == ["take"]
    assert [e["decision"] for e in journal.list_entries(cfg, mode="live")] == ["skip"]
    assert journal.summary(cfg, mode="demo")["total"] == 1
    assert len(journal.open_positions(cfg, mode="demo")) == 1
    assert journal.open_positions(cfg, mode="live") == []


def _pick_rows(labels_df, want: dict[str, int]):
    """Μερικές γραμμές (ticker, date) με κάθε τύπο έκβασης από τις ετικέτες triple barrier."""
    out = []
    lab = labels_df["label"]
    for code, n in want.items():
        sel = labels_df[lab == code]
        idx = np.linspace(0, len(sel) - 1, n).astype(int)
        out += list(sel.index[idx])
    return out


def test_resolution_matches_triple_barrier(cfg, synth, labels_df):
    prices, _ = synth
    picks = _pick_rows(labels_df, {LABEL_TP: 6, LABEL_SL: 6, LABEL_TO: 6})
    by_date: dict = {}
    for d, t in picks:
        r = labels_df.loc[(d, t)]
        by_date.setdefault(d, []).append(_row(t, "take" if len(by_date) % 2 else "skip", tp_ret=r["tp_ret"],
                                               sl_ret=r["sl_ret"], horizon_days=cfg.barriers.horizon_days))
    for d, rows in by_date.items():
        journal.record(cfg, d, rows)
    assert journal.resolve(cfg, prices) == len(picks)
    entries = {(pd.Timestamp(e["signal_date"]), e["ticker"]): e for e in journal.list_entries(cfg)}
    for key in picks:
        r, e = labels_df.loc[key], entries[key]
        assert e["status"] == STATUS_OF[int(r["label"])], key
        assert e["entry_date"] == r["entry_date"].strftime("%Y-%m-%d")
        assert e["exit_date"] == r["exit_date"].strftime("%Y-%m-%d")
        assert e["entry_price"] == pytest.approx(r["entry_price"], rel=1e-12)
        assert e["exit_price"] == pytest.approx(r["exit_price"], rel=1e-12)
        assert e["gross_ret"] == pytest.approx(r["gross_ret"], rel=1e-9, abs=1e-12)
        assert e["net_ret"] == pytest.approx(r["gross_ret"] - 0.006, rel=1e-9, abs=1e-12)
        assert e["bars_held"] == int(r["bars_held"])
    # δεύτερη επίλυση: τίποτα νέο
    assert journal.resolve(cfg, prices) == 0


def test_pending_open_void_and_frozen(cfg, synth, labels_df):
    prices, _ = synth
    last = labels_df.index.get_level_values("date").max()
    # ανοιχτή: υπάρχει είσοδος αλλά όχι έκβαση
    open_rows = labels_df[labels_df["label"].isna() & labels_df["entry_date"].notna()]
    d_open, t_open = open_rows.index[0]
    r = labels_df.loc[(d_open, t_open)]
    journal.record(cfg, d_open, [_row(t_open, "take", tp_ret=r["tp_ret"], sl_ret=r["sl_ret"])])
    journal.record(cfg, last, [_row("ETE.AT", "take"), _row("PPC.AT", "skip", tp_ret=None)])
    journal.record(cfg, last, [_row("NODATA.AT", "skip")])
    journal.resolve(cfg, prices)
    e = {(x["signal_date"], x["ticker"]): x for x in journal.list_entries(cfg)}
    o = e[(d_open.strftime("%Y-%m-%d"), t_open)]
    assert o["status"] == "open" and o["entry_date"] == r["entry_date"].strftime("%Y-%m-%d")
    assert o["exit_date"] is None
    ld = last.strftime("%Y-%m-%d")
    assert e[(ld, "ETE.AT")]["status"] == "pending"                # δεν έχει έρθει η ημέρα εισόδου
    assert e[(ld, "PPC.AT")]["status"] == "void"                   # χωρίς έγκυρα όρια
    assert e[(ld, "NODATA.AT")]["status"] == "pending"             # χωρίς τιμές: περιμένει
    # μετά την είσοδο η απόφαση «παγώνει»
    journal.record(cfg, d_open, [_row(t_open, "skip", p_tp=0.01)])
    o2 = {(x["signal_date"], x["ticker"]): x for x in journal.list_entries(cfg)}[(d_open.strftime("%Y-%m-%d"), t_open)]
    assert o2["decision"] == "take" and o2["p_tp"] == pytest.approx(0.4)
    # ανοιχτή θέση μετρά στο χαρτοφυλάκιο· με before= εξαιρούνται τα σημερινά σήματα
    tick = {p["ticker"] for p in journal.open_positions(cfg, before=last)}
    assert t_open in tick and "ETE.AT" not in tick
    assert "ETE.AT" in {p["ticker"] for p in journal.open_positions(cfg)}


def test_summary_counts_and_calibration(cfg, synth, labels_df):
    prices, _ = synth
    picks = _pick_rows(labels_df, {LABEL_TP: 4, LABEL_SL: 4, LABEL_TO: 2})
    for i, (d, t) in enumerate(picks):
        r = labels_df.loc[(d, t)]
        journal.record(cfg, d, [_row(t, "take" if i % 2 == 0 else "skip", p_tp=0.15 + 0.07 * i,
                                     tp_ret=r["tp_ret"], sl_ret=r["sl_ret"])])
    journal.resolve(cfg, prices)
    s = journal.summary(cfg)
    assert s["total"] == len(picks) and s["pending"] == 0 and s["open"] == 0
    assert s["take"]["n"] + s["skip"]["n"] == len(picks)
    assert s["take"]["resolved"] == s["take"]["n"]
    assert s["take"]["tp"] + s["take"]["sl"] + s["take"]["timeout"] == s["take"]["resolved"]
    entries = journal.list_entries(cfg)
    takes = [e for e in entries if e["decision"] == "take"]
    assert s["take"]["hit_rate"] == pytest.approx(np.mean([e["net_ret"] > 0 for e in takes]))
    assert s["take"]["avg_net_ret"] == pytest.approx(np.mean([e["net_ret"] for e in takes]))
    cal = s["calibration"]
    assert cal and sum(b["n"] for b in cal) == len(picks)
    for b in cal:
        assert b["bin_lo"] <= b["predicted"] <= b["bin_hi"] and 0 <= b["realized"] <= 1
    assert s["first_date"] <= s["last_date"]


def test_empty_journal(cfg):
    assert journal.list_entries(cfg) == []
    s = journal.summary(cfg)
    assert s["total"] == 0 and s["take"]["hit_rate"] is None and s["calibration"] == []
    assert journal.resolve(cfg, {}) == 0


def test_signal_without_any_later_session_becomes_void(cfg):
    """Μετοχή που δεν διαπραγματεύτηκε ξανά μετά το σήμα (αναστολή): άκυρη αφού περάσουν > 2 × ορίζοντας."""
    dates = pd.bdate_range("2024-01-01", periods=60, name="date")
    df = pd.DataFrame({"open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "volume": 1000.0}, index=dates)
    other = df.copy()
    halted = df.iloc[:30]
    journal.record(cfg, dates[29], [_row("HALT.AT", "take"), _row("OK.AT", "take")])
    journal.record(cfg, dates[55], [_row("FRESH.AT", "take")])
    journal.resolve(cfg, {"HALT.AT": halted, "OK.AT": other, "FRESH.AT": df.iloc[:56]})
    st = {e["ticker"]: e["status"] for e in journal.list_entries(cfg)}
    assert st == {"HALT.AT": "void", "OK.AT": "timeout", "FRESH.AT": "pending"}
    assert {p["ticker"] for p in journal.open_positions(cfg)} == {"FRESH.AT"}


def test_decisions_freeze_once_the_entry_session_starts(cfg):
    # σήμα Παρασκευής 01/03/2024 ⇒ είσοδος Δευτέρα 04/03 στις 10:00 (ώρα Αθήνας)
    assert journal.entry_deadline("2024-03-01") == pd.Timestamp("2024-03-04 10:00", tz="Europe/Athens")
    assert not journal.is_frozen("2024-03-01", pd.Timestamp("2024-03-04 09:59", tz="Europe/Athens"))
    assert journal.is_frozen("2024-03-01", pd.Timestamp("2024-03-04 10:00", tz="Europe/Athens"))
    journal.record(cfg, "2024-03-01", [_row("ETE.AT", "skip")], now=EVENING)
    # την Δευτέρα στις 12:00 (η τιμή εισόδου είναι ήδη γνωστή) μια νέα καταγραφή ΔΕΝ αλλάζει την απόφαση
    noon = pd.Timestamp("2024-03-04 12:00", tz="Europe/Athens")
    assert journal.record(cfg, "2024-03-01", [_row("ETE.AT", "take", p_tp=0.9)], now=noon) == 0
    e = journal.list_entries(cfg)[0]
    assert e["decision"] == "skip" and e["p_tp"] == pytest.approx(0.4)


def test_blocked_flag_is_stored(cfg):
    journal.record(cfg, "2024-03-01", [_row("ETE.AT", "skip", blocked=True), _row("PPC.AT", "skip"),
                                       _row("OPAP.AT", "take", blocked=True)], now=EVENING)
    e = {x["ticker"]: x for x in journal.list_entries(cfg)}
    assert e["ETE.AT"]["blocked"] is True and e["PPC.AT"]["blocked"] is False
    assert e["OPAP.AT"]["blocked"] is False            # μόνο απορρίψεις μπορεί να είναι «μπλοκ»


def test_old_database_gets_blocked_column(cfg):
    path = journal.journal_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    old_schema = journal.SCHEMA.replace("    blocked      INTEGER NOT NULL DEFAULT 0,\n", "")
    with closing(sqlite3.connect(path)) as con:
        con.executescript(old_schema)
        con.execute("INSERT INTO decisions (mode, signal_date, ticker, decision, status, created_at) "
                    "VALUES ('live', '2024-02-01', 'ETE.AT', 'skip', 'pending', 'x')")
        con.commit()
    assert journal.list_entries(cfg)[0]["blocked"] is False
