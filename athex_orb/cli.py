"""Γραμμή εντολών: python -m athex_orb {demo,refresh,backtest,serve,signals,journal}.

* demo      — δεδομένα επίδειξης (συνθετικά) + ενημέρωση + backtest
* refresh   — λήψη πραγματικών δεδομένων + εκπαίδευση + σημερινά σήματα + ημερολόγιο
* backtest  — backtest με τις αποθηκευμένες προβλέψεις, σύγκριση με αγορά & διακράτηση και τυχαίες εισόδους
* serve     — ο διακομιστής με το UI (http://127.0.0.1:8000)
* signals   — τα σήματα της τελευταίας ενημέρωσης
* journal   — το ημερολόγιο εικονικών συναλλαγών
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import time

from . import __version__, journal, pipeline, scout
from .config import load_config
from .guard import fmt_num, fmt_pct

DECISION_EL = {"take": "ΑΓΟΡΑ", "skip": "απόρριψη", "blocked": "ΜΠΛΟΚ", "none": "—"}


class _Progress:
    """Ένδειξη προόδου σε μία γραμμή (ή απλές γραμμές αν η έξοδος δεν είναι τερματικό)."""

    def __init__(self, stream=sys.stderr):
        self.stream = stream
        self.tty = hasattr(stream, "isatty") and stream.isatty()
        self.last = None
        self.last_t = 0.0
        self.t0 = time.perf_counter()

    def __call__(self, fraction: float, message_el: str) -> None:
        elapsed = time.perf_counter() - self.t0
        line = f"[{fraction * 100:5.1f}% · {elapsed:5.1f}s] {message_el}"
        if self.tty:
            self.stream.write("\r\033[K" + line)
            self.stream.flush()
            return
        # χωρίς τερματικό: μία γραμμή ανά φάση (ή ανά 3 δευτερόλεπτα), όχι για κάθε βήμα
        kind = re.sub(r"[\d/.,]+", "#", message_el)
        if kind != self.last or elapsed - self.last_t >= 3.0 or fraction >= 1.0:
            self.stream.write(line + "\n")
            self.stream.flush()
            self.last, self.last_t = kind, elapsed

    def done(self) -> None:
        if self.tty:
            self.stream.write("\n")
            self.stream.flush()


def _cfg(args):
    return load_config(path=args.config, data_dir=args.data_dir)


def _print_refresh(summary: dict) -> None:
    c = summary.get("counts") or {}
    print(f"Δεδομένα έως: {summary.get('as_of') or '—'} · λειτουργία: {summary.get('mode', '—')}")
    print(f"Μετοχές με δεδομένα: {summary.get('n_with_data', 0)}/{summary.get('n_tickers', 0)}")
    print(f"Σήματα: {c.get('take', 0)} αγορές · {c.get('skip', 0)} απορρίψεις · {c.get('blocked', 0)} μπλοκ "
          f"· {c.get('none', 0)} χωρίς πρόβλεψη")
    if summary.get("failed"):
        print("Χωρίς δεδομένα: " + ", ".join(summary["failed"]))
    m = summary.get("model") or {}
    if m:
        print(f"Μοντέλο: {m.get('kind')} · {fmt_num(m.get('n_train') or 0)} δείγματα · έως {m.get('trained_until')}")
    t = summary.get("timings") or {}
    if t:
        print("Χρόνοι: " + " · ".join(f"{k} {v:.1f}s" for k, v in t.items()))


def _print_backtest(bt: dict) -> None:
    v = bt["verdict"]
    m = bt["metrics"]
    s, b, r = m["strategy"], m["buy_hold"], m["random_median"]
    u = m.get("buy_hold_universe") or {}
    cols = (s, b, u, r)
    widths = (12, 10, 14, 17)
    print(f"Περίοδος: {bt['period']['start']} → {bt['period']['end']} · συναλλαγές: {fmt_num(bt['n_trades'])}")
    print(f"{'':24}{'Στρατηγική':>12}{'Δείκτης':>10}{'Ισοσταθμική':>14}{'Τυχαίες (διάμ.)':>17}")
    for label, key in (("Συνολική απόδοση", "total_return"), ("CAGR", "cagr"), ("Μέγιστη πτώση", "max_drawdown")):
        print(f"{label:24}" + "".join(f"{fmt_pct(x.get(key), 1):>{w}}" for x, w in zip(cols, widths)))
    sh = [x.get("sharpe") for x in cols]
    print(f"{'Sharpe':24}" + "".join(f"{(fmt_num(x, 2).replace('-', '−') if x is not None and math.isfinite(x) else '—'):>{w}}"
                                     for x, w in zip(sh, widths)))
    sk = bt.get("skill") or {}
    if sk.get("n"):
        print(f"Δεξιότητα εκτός δείγματος: log-loss {fmt_num(sk['logloss_model'], 3)} έναντι "
              f"{fmt_num(sk['logloss_base'], 3)} (σταθερές συχνότητες) · p = {fmt_num(sk['p_value'], 3)} → "
              f"{'ναι' if sk.get('ok') else 'όχι'}")
    print(f"Ετυμηγορία ({v['level']}): {v['text_el']}")


def cmd_demo(args) -> int:
    cfg = _cfg(args)
    pr = _Progress()
    t0 = time.perf_counter()
    try:
        out = pipeline.run_demo(cfg, progress=pr)
    except pipeline.PipelineError as e:
        pr.done()
        print(f"Σφάλμα: {e}", file=sys.stderr)
        return 1
    pr.done()
    print("ΕΠΙΔΕΙΞΗ — συνθετικά δεδομένα, όχι πραγματικές τιμές.")
    _print_refresh(out["refresh"])
    _print_backtest(out["backtest"])
    print(f"Συνολικός χρόνος: {time.perf_counter() - t0:.1f}s")
    print("Για το UI: python -m athex_orb serve")
    return 0


def cmd_refresh(args) -> int:
    cfg = _cfg(args)
    pr = _Progress()
    try:
        out = pipeline.run_refresh(cfg, progress=pr)
    except pipeline.PipelineError as e:
        pr.done()
        print(f"Σφάλμα: {e}", file=sys.stderr)
        return 1
    pr.done()
    _print_refresh(out)
    return 0


def cmd_backtest(args) -> int:
    cfg = _cfg(args)
    if args.runs is not None:
        cfg.backtest.random_runs = max(0, int(args.runs))
    pr = _Progress()
    try:
        out = pipeline.run_backtest_job(cfg, progress=pr)
    except pipeline.PipelineError as e:
        pr.done()
        print(f"Σφάλμα: {e}", file=sys.stderr)
        return 1
    pr.done()
    _print_backtest(out)
    return 0


def cmd_serve(args) -> int:
    cfg = _cfg(args)
    host = args.host or cfg.server.host
    port = int(args.port or cfg.server.port)
    try:
        import uvicorn
    except ImportError:
        print("Λείπει το uvicorn: pip install uvicorn", file=sys.stderr)
        return 1
    from .server import create_app

    busy = _port_in_use(host, port)
    if busy:
        print(f"Η θύρα {port} χρησιμοποιείται ήδη ({busy}) — ίσως τρέχει ήδη το ATHEX Orb ή άλλη εφαρμογή. "
              f"Δοκιμάστε: python -m athex_orb serve --port {port + 1}", file=sys.stderr)
        return 1
    app = create_app(cfg)
    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    print(f"Άνοιξε http://{shown}:{port}", flush=True)
    mode = scout.data_mode(cfg)
    if mode == "empty":
        print("Δεν υπάρχουν ακόμη δεδομένα: από το UI επιλέξτε «Δοκιμή με δεδομένα επίδειξης» "
              "ή «Λήψη πραγματικών δεδομένων».", flush=True)
    print("Τερματισμός: Ctrl+C", flush=True)
    uvicorn.run(app, host=host, port=port, log_level=args.log_level)
    return 0


def _port_in_use(host: str, port: int) -> str | None:
    """None αν η θύρα είναι ελεύθερη, αλλιώς σύντομη αιτία (ελέγχεται πριν τυπωθεί η διεύθυνση)."""
    import errno
    import socket

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
    except OSError as e:
        if e.errno in (errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", -1)):
            return "σε χρήση"
        if e.errno in (errno.EACCES,):
            return "χωρίς δικαίωμα πρόσβασης"
        return str(e.strerror or e)
    return None


def cmd_signals(args) -> int:
    cfg = _cfg(args)
    sig = pipeline.load_state(cfg, "signals")
    if not sig or not sig.get("items"):
        print("Δεν υπάρχουν σήματα — τρέξτε πρώτα: python -m athex_orb demo (ή refresh)")
        return 1
    items = sig["items"]
    if not args.all:
        items = [i for i in items if i["decision"] in ("take", "skip")]
    print(f"Σήματα της {sig.get('as_of')} ({'ΕΠΙΔΕΙΞΗ' if sig.get('mode') == 'demo' else 'πραγματικά δεδομένα'})")
    print(f"{'Μετοχή':<13}{'Απόφαση':<11}{'P(στόχος)':>10}{'P(stop)':>9}{'EV καθαρό':>11}{'Είσοδος':>10}"
          f"{'Στόχος':>10}{'Stop':>10}{'Ποσό':>9}  Αιτία")
    for i in items:
        def money(x):
            return fmt_num(x, 2) if x is not None else "—"
        print(f"{i['ticker']:<13}{DECISION_EL.get(i['decision'], i['decision']):<11}"
              f"{fmt_pct(i.get('p_tp'), 0):>10}{fmt_pct(i.get('p_sl'), 0):>9}"
              f"{fmt_pct(i.get('ev_net'), 2, sign=True):>11}{money(i.get('entry_ref')):>10}"
              f"{money(i.get('tp_price')):>10}{money(i.get('sl_price')):>10}"
              f"{(fmt_num(i['size_eur'], 0) if i.get('size_eur') else '—'):>9}  {i.get('reason_el', '')}")
    print("Εργαλείο έρευνας — όχι επενδυτική συμβουλή. Δεν εκτελεί εντολές.")
    return 0


def cmd_journal(args) -> int:
    cfg = _cfg(args)
    mode = scout.data_mode(cfg)
    jm = mode if mode in ("demo", "live") else None
    s = journal.summary(cfg, mode=jm)
    if not s["total"]:
        print("Το ημερολόγιο είναι άδειο — οι αποφάσεις καταγράφονται σε κάθε ενημέρωση.")
        return 0
    print(f"Ημερολόγιο ({jm or 'όλα'}): {s['total']} εγγραφές · {s['pending']} σε αναμονή · {s['open']} ανοιχτές")
    for key, label in (("take", "Λήψεις"), ("skip", "Απορρίψεις")):
        g = s[key]
        hr = fmt_pct(g["hit_rate"], 0) if g["hit_rate"] is not None else "—"
        av = fmt_pct(g["avg_net_ret"], 2, sign=True) if g["avg_net_ret"] is not None else "—"
        print(f"  {label}: {g['n']} · επιλυμένες {g['resolved']} (στόχος {g['tp']}, stop {g['sl']}, "
              f"λήξη {g['timeout']}) · κερδοφόρες {hr} · μέση καθαρή {av}")
    for e in journal.list_entries(cfg, limit=args.limit, mode=jm):
        net = fmt_pct(e["net_ret"], 2, sign=True) if e["net_ret"] is not None else ""
        print(f"  {e['signal_date']} {e['ticker']:<12} {e['decision']:<5} {e['status_el']:<20} {net:>8}  "
              f"{e.get('reason_el') or ''}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m athex_orb",
                                 description="ATHEX Orb — έρευνα, backtest και εικονικές συναλλαγές για το ΧΑ "
                                             "(δεν εκτελεί εντολές).")
    ap.add_argument("--version", action="version", version=f"ATHEX Orb {__version__}")
    ap.add_argument("--data-dir", default=None, help="φάκελος δεδομένων (προεπιλογή: ./data)")
    ap.add_argument("--config", default=None, help="αρχείο ρυθμίσεων TOML (προεπιλογή: config.toml)")
    sub = ap.add_subparsers(dest="command", required=True, metavar="εντολή")
    sub.add_parser("demo", help="δεδομένα επίδειξης + ενημέρωση + backtest").set_defaults(func=cmd_demo)
    sub.add_parser("refresh", help="λήψη δεδομένων, εκπαίδευση, σήματα").set_defaults(func=cmd_refresh)
    p = sub.add_parser("backtest", help="backtest με τις αποθηκευμένες προβλέψεις")
    p.add_argument("--runs", type=int, default=None, help="πλήθος τυχαίων στρατηγικών")
    p.set_defaults(func=cmd_backtest)
    p = sub.add_parser("serve", help="διακομιστής + UI")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--log-level", default="warning")
    p.set_defaults(func=cmd_serve)
    p = sub.add_parser("signals", help="τα σήματα της τελευταίας ενημέρωσης")
    p.add_argument("--all", action="store_true", help="και οι μετοχές με μπλοκ/χωρίς πρόβλεψη")
    p.set_defaults(func=cmd_signals)
    p = sub.add_parser("journal", help="ημερολόγιο εικονικών συναλλαγών")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(func=cmd_journal)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)
