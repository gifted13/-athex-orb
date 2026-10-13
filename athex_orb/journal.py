"""Ημερολόγιο εικονικών συναλλαγών (paper trading): η «μνήμη» των αποφάσεων.

Κάθε ενημέρωση καταγράφει τις αποφάσεις της ημέρας — και τις λήψεις («take») και τις απορρίψεις
(«skip») — ώστε αργότερα να φαίνεται αν το φίλτρο προσθέτει πραγματική αξία: οι απορρίψεις
επιλύονται κι αυτές («τι θα γινόταν αν τις είχαμε πάρει»).

Αποθήκευση: SQLite στο data/journal.sqlite, πίνακας `decisions`. Η επίλυση χρησιμοποιεί τον κοινό
προσομοιωτή `labels.simulate_exit` (είσοδος στο άνοιγμα της επόμενης συνεδρίασης μετά το σήμα,
όρια από την τιμή εισόδου), άρα τα αποτελέσματα ταυτίζονται με τις ετικέτες και το backtest.

Σε σχέση με το βασικό σχήμα του ARCHITECTURE.md υπάρχουν μερικές επιπλέον στήλες:
* `mode` («demo»/«live»): οι εγγραφές επίδειξης δεν ανακατεύονται ποτέ με τις πραγματικές
  (η μοναδικότητα είναι ανά (mode, signal_date, ticker)),
* `sector`, `horizon_days` (ο ορίζοντας τη στιγμή της απόφασης), `cost_rt` (εκτιμώμενο κόστος
  μετ' επιστροφής ⇒ net_ret = gross_ret − cost_rt), `bars_held`, `updated_at`,
* `blocked` (1 αν η απόρριψη έγινε από τον Φύλακα — η απόφαση μένει «skip»).

Οι αποφάσεις «παγώνουν» μόλις ξεκινήσει η συνεδρίαση εισόδου (10:00 ώρα Αθήνας της επόμενης εργάσιμης μετά
την ημερομηνία σήματος): μετά από αυτό, καμία νέα καταγραφή δεν τις αλλάζει — ώστε μια αλλαγή ρυθμίσεων αφού
φανούν οι τιμές να μην «ξαναγράφει την ιστορία».
"""
from __future__ import annotations

import math
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import LABEL_SL, LABEL_TO, LABEL_TP, simulate_exit

DB_NAME = "journal.sqlite"
DECISIONS = ("take", "skip")
STATUSES = ("pending", "open", "tp", "sl", "timeout", "void")
RESOLVED = ("tp", "sl", "timeout")
STALE_FACTOR = 2                # χωρίς συνεδρίαση εισόδου για > 2 × ορίζοντα ημέρες ⇒ άκυρη
ATHENS_TZ = "Europe/Athens"
ENTRY_OPEN_LOCAL = (10, 0)      # έναρξη της συνεδρίασης του ΧΑ (προ-δημοπρασία) — μετά «παγώνει» η απόφαση
STATUS_OF_LABEL = {LABEL_TP: "tp", LABEL_SL: "sl", LABEL_TO: "timeout"}
STATUS_LABELS_EL = {
    "pending": "Σε αναμονή εισόδου",
    "open": "Ανοιχτή",
    "tp": "Στόχος",
    "sl": "Stop",
    "timeout": "Λήξη χρόνου",
    "void": "Άκυρη",
}

COLUMNS = ["id", "mode", "signal_date", "ticker", "sector", "decision", "reason_el", "p_tp", "p_sl", "p_to",
           "ev_net", "entry_ref", "tp_ret", "sl_ret", "size_eur", "horizon_days", "cost_rt", "status",
           "entry_date", "entry_price", "exit_date", "exit_price", "gross_ret", "net_ret", "bars_held",
           "created_at", "updated_at", "blocked"]
# πεδία που ορίζει η απόφαση (και ενημερώνονται όσο η εγγραφή είναι ακόμη «σε αναμονή»)
DECISION_FIELDS = ["sector", "decision", "reason_el", "p_tp", "p_sl", "p_to", "ev_net", "entry_ref", "tp_ret",
                   "sl_ret", "size_eur", "horizon_days", "cost_rt", "blocked"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    mode         TEXT NOT NULL DEFAULT 'live',
    signal_date  TEXT NOT NULL,
    ticker       TEXT NOT NULL,
    sector       TEXT,
    decision     TEXT NOT NULL CHECK (decision IN ('take', 'skip')),
    reason_el    TEXT,
    p_tp         REAL,
    p_sl         REAL,
    p_to         REAL,
    ev_net       REAL,
    entry_ref    REAL,
    tp_ret       REAL,
    sl_ret       REAL,
    size_eur     REAL,
    horizon_days INTEGER,
    cost_rt      REAL,
    status       TEXT NOT NULL DEFAULT 'pending'
                 CHECK (status IN ('pending', 'open', 'tp', 'sl', 'timeout', 'void')),
    entry_date   TEXT,
    entry_price  REAL,
    exit_date    TEXT,
    exit_price   REAL,
    gross_ret    REAL,
    net_ret      REAL,
    bars_held    INTEGER,
    created_at   TEXT NOT NULL,
    updated_at   TEXT,
    blocked      INTEGER NOT NULL DEFAULT 0,
    UNIQUE (mode, signal_date, ticker)
);
CREATE INDEX IF NOT EXISTS decisions_status ON decisions (status);
CREATE INDEX IF NOT EXISTS decisions_date ON decisions (signal_date);
"""


# ---------------------------------------------------------------- βάση

def journal_path(cfg) -> Path:
    return Path(cfg.data_dir) / DB_NAME


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _connect(cfg) -> sqlite3.Connection:
    path = journal_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=15)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    have = {r[1] for r in con.execute("PRAGMA table_info(decisions)").fetchall()}
    if "blocked" not in have:                   # βάση από παλαιότερη έκδοση
        con.execute("ALTER TABLE decisions ADD COLUMN blocked INTEGER NOT NULL DEFAULT 0")
        con.commit()
    return con


def entry_deadline(signal_date) -> pd.Timestamp:
    """Η στιγμή (UTC) που ξεκινά η συνεδρίαση εισόδου ενός σήματος: 10:00 ώρα Αθήνας της επόμενης εργάσιμης."""
    nxt = pd.Timestamp(signal_date).normalize() + pd.offsets.BDay(1)
    local = pd.Timestamp(year=nxt.year, month=nxt.month, day=nxt.day, hour=ENTRY_OPEN_LOCAL[0],
                         minute=ENTRY_OPEN_LOCAL[1]).tz_localize(ATHENS_TZ)
    return local.tz_convert("UTC")


def is_frozen(signal_date, now=None) -> bool:
    """True αν η συνεδρίαση εισόδου έχει ήδη ξεκινήσει — η απόφαση δεν αλλάζει πια."""
    ts = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    return ts >= entry_deadline(signal_date)


def _date_str(d) -> str | None:
    if d is None:
        return None
    try:
        ts = pd.Timestamp(d)
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d")


def _num(x) -> float | None:
    """Αριθμός για αποθήκευση (None για κενές/άπειρες τιμές)."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _int(x) -> int | None:
    v = _num(x)
    return int(round(v)) if v is not None else None


# ---------------------------------------------------------------- καταγραφή

def record(cfg, signal_date, rows: list[dict], mode: str = "live", now=None) -> int:
    """Καταγράφει (upsert) τις αποφάσεις μιας ημέρας σήματος — λήψεις και απορρίψεις.

    Ταυτοποίηση ανά (mode, signal_date, ticker): αν η ίδια ενημέρωση τρέξει ξανά, η εγγραφή
    ενημερώνεται αντί να διπλασιαστεί. Μια εγγραφή αλλάζει μόνο όσο είναι ακόμη «σε αναμονή» **και**
    δεν έχει ξεκινήσει η συνεδρίαση εισόδου (`is_frozen`)· μετά η απόφαση «παγώνει» — δεν ξαναγράφουμε
    την ιστορία. `now`: η τρέχουσα στιγμή (για tests). Επιστρέφει πόσες εγγραφές γράφτηκαν.
    """
    sd = _date_str(signal_date)
    if sd is None:
        raise ValueError("Μη έγκυρη ημερομηνία σήματος για το ημερολόγιο")
    frozen = 1 if is_frozen(sd, now) else 0
    now = _now_iso()
    cols = ["mode", "signal_date", "ticker"] + DECISION_FIELDS + ["status", "created_at", "updated_at"]
    placeholders = ", ".join("?" for _ in cols)
    updates = ", ".join(f"{c} = excluded.{c}" for c in DECISION_FIELDS + ["updated_at"])
    sql = (f"INSERT INTO decisions ({', '.join(cols)}) VALUES ({placeholders}) "
           f"ON CONFLICT (mode, signal_date, ticker) DO UPDATE SET {updates} "
           f"WHERE decisions.status = 'pending' AND decisions.entry_date IS NULL AND {frozen} = 0")
    written = 0
    with closing(_connect(cfg)) as con, con:
        for r in rows:
            ticker = str(r.get("ticker") or "").strip()
            decision = r.get("decision")
            if not ticker or decision not in DECISIONS:
                continue
            vals = {
                "sector": r.get("sector"),
                "decision": decision,
                "reason_el": r.get("reason_el"),
                "p_tp": _num(r.get("p_tp")), "p_sl": _num(r.get("p_sl")), "p_to": _num(r.get("p_to")),
                "ev_net": _num(r.get("ev_net")),
                "entry_ref": _num(r.get("entry_ref")),
                "tp_ret": _num(r.get("tp_ret")), "sl_ret": _num(r.get("sl_ret")),
                "size_eur": _num(r.get("size_eur")) if decision == "take" else None,
                "horizon_days": _int(r.get("horizon_days", cfg.barriers.horizon_days)),
                "cost_rt": _num(r.get("cost_rt")),
                "blocked": 1 if (decision == "skip" and r.get("blocked")) else 0,
            }
            params = [mode, sd, ticker] + [vals[c] for c in DECISION_FIELDS] + ["pending", now, now]
            cur = con.execute(sql, params)
            written += max(cur.rowcount, 0)
    return written


def clear(cfg, mode: str) -> int:
    """Διαγράφει όλες τις εγγραφές μιας λειτουργίας (π.χ. «demo» όταν δημιουργούνται νέα δεδομένα επίδειξης)."""
    if not journal_path(cfg).exists():
        return 0
    with closing(_connect(cfg)) as con, con:
        return max(con.execute("DELETE FROM decisions WHERE mode = ?", [mode]).rowcount, 0)


# ---------------------------------------------------------------- επίλυση

def resolve(cfg, prices: dict[str, pd.DataFrame], mode: str | None = None) -> int:
    """Συμπληρώνει τις εκβάσεις όσων εγγραφών έχουν πλέον δεδομένα (και για λήψεις και για απορρίψεις).

    * Είσοδος: άνοιγμα της πρώτης συνεδρίασης *μετά* την ημερομηνία σήματος.
    * Στόχος/stop: τιμή εισόδου × (1 ± tp_ret/sl_ret)· ορίζοντας: αυτός που ίσχυε στην απόφαση.
    * Χωρίς συνεδρίαση εισόδου ⇒ μένει «σε αναμονή»· με είσοδο αλλά χωρίς έκβαση ⇒ «ανοιχτή».
    * Χωρίς έγκυρα όρια ⇒ «άκυρη».
    `mode`: μόνο εγγραφές αυτής της λειτουργίας (None = όλες). Επιστρέφει πόσες εγγραφές άλλαξαν.
    """
    sql = "SELECT * FROM decisions WHERE status IN ('pending', 'open')"
    args: list = []
    if mode is not None:
        sql += " AND mode = ?"
        args.append(mode)
    changed = 0
    now = _now_iso()
    arrays: dict[str, tuple] = {}
    ends = [df.index[-1] for df in (prices or {}).values() if df is not None and len(df)]
    market_last = max(ends) if ends else None
    with closing(_connect(cfg)) as con, con:
        for row in con.execute(sql, args).fetchall():
            df = prices.get(row["ticker"]) if prices else None
            if df is None or not len(df):
                continue
            if row["ticker"] not in arrays:
                arrays[row["ticker"]] = (df.index, df["open"].to_numpy("float64"), df["high"].to_numpy("float64"),
                                         df["low"].to_numpy("float64"), df["close"].to_numpy("float64"))
            idx, o, h, l, c = arrays[row["ticker"]]
            upd = _resolve_one(row, idx, o, h, l, c, int(row["horizon_days"] or cfg.barriers.horizon_days),
                               market_last)
            if upd is None:
                continue
            upd["updated_at"] = now
            sets = ", ".join(f"{k} = ?" for k in upd)
            con.execute(f"UPDATE decisions SET {sets} WHERE id = ?", [*upd.values(), row["id"]])
            changed += 1
    return changed


def _resolve_one(row, idx: pd.DatetimeIndex, o, h, l, c, horizon: int, market_last=None) -> dict | None:
    """Οι νέες τιμές μιας εγγραφής (ή None αν δεν αλλάζει τίποτα).

    Αν η μετοχή δεν διαπραγματεύτηκε καθόλου μετά το σήμα ενώ η αγορά προχώρησε πάνω από 2 × ορίζοντα
    συνεδριάσεις (αναστολή, διαγραφή), η εγγραφή γίνεται «άκυρη» ώστε να μη δεσμεύει θέση για πάντα.
    """
    tp_ret, sl_ret = _num(row["tp_ret"]), _num(row["sl_ret"])
    if tp_ret is None or sl_ret is None or tp_ret <= 0 or sl_ret <= 0 or horizon < 1:
        return {"status": "void"}
    sig = pd.Timestamp(row["signal_date"])
    entry_idx = int(idx.searchsorted(sig, side="right"))      # πρώτη συνεδρίαση αυστηρά μετά το σήμα
    if entry_idx >= len(idx):
        if market_last is not None and len(pd.bdate_range(sig, market_last)) - 1 > STALE_FACTOR * horizon:
            return {"status": "void"}                           # η μετοχή σταμάτησε να διαπραγματεύεται
        return None                                             # δεν έχει έρθει ακόμη η ημέρα εισόδου
    entry_price = float(o[entry_idx])
    if not (math.isfinite(entry_price) and entry_price > 0):
        return {"status": "void"}
    tp_price = entry_price * (1.0 + tp_ret)                     # ίδια πράξη με labels.barrier_prices
    sl_price = entry_price * (1.0 - sl_ret)
    label, exit_idx, exit_price = simulate_exit(o, h, l, c, entry_idx, entry_price, tp_price, sl_price, horizon)
    out = {"entry_date": idx[entry_idx].strftime("%Y-%m-%d"), "entry_price": entry_price}
    if label < 0:
        if row["status"] == "open" and row["entry_date"] == out["entry_date"]:
            return None
        out["status"] = "open"
        return out
    gross = exit_price / entry_price - 1.0
    cost = _num(row["cost_rt"]) or 0.0
    out.update({
        "status": STATUS_OF_LABEL[int(label)],
        "exit_date": idx[exit_idx].strftime("%Y-%m-%d"),
        "exit_price": float(exit_price),
        "gross_ret": float(gross),
        "net_ret": float(gross - cost),
        "bars_held": int(exit_idx - entry_idx + 1),
    })
    return out


# ---------------------------------------------------------------- ανάγνωση

def _row_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["status_el"] = STATUS_LABELS_EL.get(d.get("status"), d.get("status"))
    d["blocked"] = bool(d.get("blocked"))
    return d


def list_entries(cfg, limit: int = 500, mode: str | None = None, ticker: str | None = None) -> list[dict]:
    """Οι πιο πρόσφατες εγγραφές (νεότερες πρώτα· σε κάθε ημέρα οι λήψεις πρώτα, μετά κατά EV)."""
    if not journal_path(cfg).exists():
        return []
    sql = "SELECT * FROM decisions WHERE 1 = 1"
    args: list = []
    if mode is not None:
        sql += " AND mode = ?"
        args.append(mode)
    if ticker is not None:
        sql += " AND ticker = ?"
        args.append(ticker)
    sql += (" ORDER BY signal_date DESC, CASE decision WHEN 'take' THEN 0 ELSE 1 END, "
            "COALESCE(ev_net, -1e9) DESC, ticker LIMIT ?")
    args.append(max(int(limit), 0))
    with closing(_connect(cfg)) as con:
        return [_row_dict(r) for r in con.execute(sql, args).fetchall()]


def open_positions(cfg, mode: str | None = None, before=None) -> list[dict]:
    """Οι εικονικές θέσεις που είναι ακόμη ανοιχτές ή περιμένουν είσοδο (μόνο λήψεις).

    `before`: μόνο σήματα αυστηρά πριν από αυτή την ημερομηνία (π.χ. για να μη μετρά η τρέχουσα ημέρα).
    """
    if not journal_path(cfg).exists():
        return []
    sql = "SELECT * FROM decisions WHERE decision = 'take' AND status IN ('pending', 'open')"
    args: list = []
    if mode is not None:
        sql += " AND mode = ?"
        args.append(mode)
    b = _date_str(before)
    if b is not None:
        sql += " AND signal_date < ?"
        args.append(b)
    sql += " ORDER BY signal_date, ticker"
    with closing(_connect(cfg)) as con:
        return [_row_dict(r) for r in con.execute(sql, args).fetchall()]


def _group(rows: list[dict]) -> dict:
    res = [r for r in rows if r["status"] in RESOLVED]
    net = np.asarray([r["net_ret"] for r in res if r["net_ret"] is not None], dtype="float64")
    gross = np.asarray([r["gross_ret"] for r in res if r["gross_ret"] is not None], dtype="float64")
    n_res = len(res)
    return {
        "n": len(rows),
        "resolved": n_res,
        "tp": sum(1 for r in res if r["status"] == "tp"),
        "sl": sum(1 for r in res if r["status"] == "sl"),
        "timeout": sum(1 for r in res if r["status"] == "timeout"),
        "hit_rate": float((net > 0).mean()) if len(net) else None,        # κερδοφόρες μετά τα κόστη
        "tp_rate": (sum(1 for r in res if r["status"] == "tp") / n_res) if n_res else None,
        "avg_net_ret": float(net.mean()) if len(net) else None,
        "avg_gross_ret": float(gross.mean()) if len(gross) else None,
    }


def _calibration(rows: list[dict], width: float = 0.1) -> list[dict]:
    """Βαθμονόμηση του p_tp στις επιλυμένες εγγραφές (σταθεροί κάδοι πλάτους 10 ποσοστιαίων μονάδων)."""
    pts = [(r["p_tp"], 1.0 if r["status"] == "tp" else 0.0) for r in rows
           if r["status"] in RESOLVED and r["p_tp"] is not None]
    out = []
    n_bins = int(round(1.0 / width))
    for b in range(n_bins):
        lo, hi = b * width, (b + 1) * width
        sel = [(p, y) for p, y in pts if lo <= p < hi or (b == n_bins - 1 and p == hi)]
        if not sel:
            continue
        out.append({"bin_lo": round(lo, 4), "bin_hi": round(hi, 4),
                    "predicted": float(np.mean([p for p, _ in sel])),
                    "realized": float(np.mean([y for _, y in sel])), "n": len(sel)})
    return out


def summary(cfg, mode: str | None = None) -> dict:
    """Σύνοψη: πλήθη, ποσοστά επιτυχίας και μέση καθαρή απόδοση για λήψεις vs απορρίψεις, βαθμονόμηση."""
    rows = list_entries(cfg, limit=10**9, mode=mode)
    dates = sorted(r["signal_date"] for r in rows)
    return {
        "total": len(rows),
        "pending": sum(1 for r in rows if r["status"] == "pending"),
        "open": sum(1 for r in rows if r["status"] == "open"),
        "void": sum(1 for r in rows if r["status"] == "void"),
        "take": _group([r for r in rows if r["decision"] == "take"]),
        "skip": _group([r for r in rows if r["decision"] == "skip"]),
        "calibration": _calibration(rows),
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
    }
