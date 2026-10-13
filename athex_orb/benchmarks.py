"""Μέτρα σύγκρισης: δείκτες απόδοσης, αγορά και διακράτηση, τυχαίες στρατηγικές, βαθμονόμηση,
έλεγχος δεξιότητας του μοντέλου εκτός δείγματος, ετυμηγορία.

Η ετυμηγορία είναι σκόπιμα αυστηρή: «πλεονέκτημα» μόνο αν
* η στρατηγική ξεπερνά τις τυχαίες στρατηγικές με τους ίδιους κανόνες με p ≤ 5% (τουλάχιστον
  MIN_RANDOM_RUNS εκτελέσεις — με λίγες εκτελέσεις η «νίκη» επί όλων είναι συχνά τύχη),
* νικά την αγορά και διακράτηση **και** του δείκτη **και** ενός ισοσταθμικού χαρτοφυλακίου των ίδιων
  μετοχών (ίδια δεδομένα, ίδια προσαρμογή για μερίσματα, ίδια μεροληψία επιβίωσης με τη στρατηγική),
* οι πιθανότητες του μοντέλου εκτός δείγματος είναι πράγματι καλύτερες από τις σταθερές ιστορικές
  συχνότητες (log-loss, με block bootstrap ανά ημερομηνία — `skill`),
* είναι κερδοφόρα και έχει αρκετές συναλλαγές.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .guard import fmt_num, fmt_pct
from .labels import LABEL_TP

TRADING_DAYS = 252
EDGE_PERCENTILE = 0.95          # ποσοστό τυχαίων στρατηγικών που πρέπει να ξεπεραστεί
EDGE_P_VALUE = 0.05             # p = (1 + #{τυχαίες ≥ στρατηγική}) / (1 + εκτελέσεις)
WEAK_PERCENTILE = 0.80
POOR_PERCENTILE = 0.25          # κάτω από αυτό: «χειρότερα από τις τυχαίες»
MIN_TRADES_FOR_VERDICT = 30     # λιγότερες συναλλαγές ⇒ κανένα συμπέρασμα
MIN_RANDOM_RUNS = 100           # λιγότερες τυχαίες εκτελέσεις ⇒ κανένα συμπέρασμα
SKILL_BLOCK_DAYS = 20           # μήκος μπλοκ (ημερομηνίες) στο bootstrap — οι ετικέτες επικαλύπτονται χρονικά
SKILL_BOOT = 500
SKILL_P_VALUE = 0.05

NAN = float("nan")
RUN_METRICS = ("total_return", "cagr", "max_drawdown", "sharpe", "volatility", "final_equity")
RANDOM_METHOD_EL = ("Τυχαίες στρατηγικές με τους ίδιους κανόνες: ίδιες ημέρες και ίδιος αριθμός υποψηφίων "
                    "ανά ημέρα με το μοντέλο, ίδιοι έλεγχοι, μέγεθος θέσης και κόστη — αλλά οι προβλέψεις "
                    "αντιστοιχίζονται σε τυχαία ανακατεμένες μετοχές.")


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NAN
    return v


def _date(ts) -> str | None:
    return None if ts is None or pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m-%d")


# ---------------------------------------------------------------- δείκτες απόδοσης

def metrics(equity: pd.Series, trades: pd.DataFrame | None, capital: float) -> dict:
    """Δείκτες μιας καμπύλης κεφαλαίου (και των συναλλαγών της, αν δοθούν).

    total_return, cagr, max_drawdown (αρνητικό κλάσμα), sharpe & volatility (ετήσια, χωρίς επιτόκιο),
    n_trades, win_rate, avg_win/avg_loss (μέση καθαρή απόδοση κερδοφόρων/ζημιογόνων), profit_factor,
    avg_bars_held, total_costs_eur, exposure (μέρος ημερών με ≥ 1 θέση· 1 χωρίς συναλλαγές = πλήρης
    επένδυση, π.χ. αγορά και διακράτηση). Μη ορισμένες τιμές = NaN. Επιπλέον: final_equity, start, end.
    """
    capital = float(capital)
    eq = equity.dropna() if equity is not None else pd.Series([], dtype="float64")
    out = {k: NAN for k in ("total_return", "cagr", "max_drawdown", "sharpe", "volatility", "win_rate",
                            "avg_win", "avg_loss", "profit_factor", "avg_bars_held")}
    out.update({"n_trades": 0, "total_costs_eur": 0.0, "exposure": NAN, "final_equity": NAN,
                "start": None, "end": None})
    if len(eq):
        v = eq.to_numpy(dtype="float64")
        out["final_equity"] = float(v[-1])
        out["start"], out["end"] = _date(eq.index[0]), _date(eq.index[-1])
        out["total_return"] = float(v[-1] / capital - 1.0) if capital > 0 else NAN
        years = (pd.Timestamp(eq.index[-1]) - pd.Timestamp(eq.index[0])).days / 365.25
        if years > 0 and v[-1] > 0 and capital > 0:
            out["cagr"] = float((v[-1] / capital) ** (1.0 / years) - 1.0)
        prev = np.concatenate(([capital], v[:-1]))
        with np.errstate(divide="ignore", invalid="ignore"):
            rets = v / prev - 1.0
        rets = rets[np.isfinite(rets)]
        peak = np.maximum.accumulate(np.concatenate(([capital], v)))[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = v / peak - 1.0
        out["max_drawdown"] = float(np.nanmin(dd)) if np.isfinite(dd).any() else NAN
        if len(rets) > 1:
            sd = float(np.std(rets, ddof=1))
            out["volatility"] = sd * math.sqrt(TRADING_DAYS)
            if sd > 0:
                out["sharpe"] = float(np.mean(rets) / sd * math.sqrt(TRADING_DAYS))

    if trades is None:
        out["exposure"] = 1.0 if len(eq) else NAN
        return out

    n = int(len(trades))
    out["n_trades"] = n
    if n:
        pnl = trades["net_pnl_eur"].to_numpy(dtype="float64")
        nret = trades["net_ret"].to_numpy(dtype="float64")
        wins, losses = pnl > 0, pnl < 0
        out["win_rate"] = float(wins.mean())
        if wins.any():
            out["avg_win"] = float(nret[wins].mean())
        if losses.any():
            out["avg_loss"] = float(nret[losses].mean())
            out["profit_factor"] = float(pnl[wins].sum() / abs(pnl[losses].sum()))
        if "bars_held" in trades.columns:
            out["avg_bars_held"] = float(pd.to_numeric(trades["bars_held"], errors="coerce").mean())
        out["total_costs_eur"] = float(trades["cost_eur"].sum())
    if len(eq):
        idx = pd.DatetimeIndex(eq.index)
        held = np.zeros(len(idx) + 1, dtype="int64")
        if n:
            s = idx.searchsorted(pd.DatetimeIndex(trades["entry_date"]), side="left")
            e = idx.searchsorted(pd.DatetimeIndex(trades["exit_date"]), side="right")
            np.add.at(held, s, 1)
            np.add.at(held, e, -1)
        out["exposure"] = float((np.cumsum(held)[:-1] > 0).mean())
    return out


def buy_and_hold(index_df: pd.DataFrame, start, end, capital) -> pd.Series:
    """Αγορά του δείκτη στο κλείσιμο της `start` και διακράτηση έως την `end` (χωρίς κόστη)."""
    if index_df is None or not len(index_df):
        return pd.Series([], dtype="float64", index=pd.DatetimeIndex([], name="date"), name="buy_hold")
    c = index_df["close"].astype("float64").dropna()
    c = c[(c.index >= pd.Timestamp(start)) & (c.index <= pd.Timestamp(end))]
    if not len(c):
        return pd.Series([], dtype="float64", index=pd.DatetimeIndex([], name="date"), name="buy_hold")
    out = float(capital) * c / float(c.iloc[0])
    out.name = "buy_hold"
    out.index.name = "date"
    return out


# ---------------------------------------------------------------- τυχαίες στρατηγικές

def random_benchmark(prices, panel, preds, labels, cfg, runs, progress=None, context=None) -> dict:
    """Monte Carlo με τυχαίες επιλογές υπό τους ίδιους κανόνες (βλ. `executor`).

    Κάθε εκτέλεση: ίδιες ημέρες και ίδιος αριθμός υποψηφίων που εξετάζονται ανά ημέρα με το μοντέλο,
    ίδιοι έλεγχοι μετοχής/χαρτοφυλακίου, ίδιο μέγεθος θέσης και κόστη· κάθε μετοχή «δανείζεται» τις
    προβλέψεις μιας άλλης (τυχαία μετάθεση ανά εκτέλεση). Αναπαραγώγιμο από το cfg.backtest.seed.
    Σε δεδομένα χωρίς σήμα το percentile της στρατηγικής είναι περίπου ομοιόμορφο (βλ. tests/test_backtest.py).
    Επιστρέφει {"equity_median", "equity_p05", "equity_p95" (Series), "final_returns": [...],
    "n_trades": [...], "costs_eur": [...], "win_rates": [...], "run_metrics": {μετρική: [...]},
    "metrics_median": {μετρική: διάμεσος}, "runs"} (λίστες: μία τιμή ανά εκτέλεση· `win_rates` = μερίδιο
    κλεισμένων συναλλαγών με θετικό καθαρό αποτέλεσμα, NaN χωρίς συναλλαγές).

    Η διάμεσος καμπύλη (`equity_median`, σημείο προς σημείο) είναι μόνο για το γράφημα: δεν την παρήγαγε
    καμία εκτέλεση, οπότε Sharpe, μεταβλητότητα, πτώση και CAGR των «τυχαίων» είναι οι διάμεσοι των μετρικών
    κάθε εκτέλεσης (`metrics_median`).
    """
    from . import executor

    runs = max(int(runs), 0)
    empty = pd.Series([], dtype="float64", index=pd.DatetimeIndex([], name="date"))
    ctx = context if context is not None else executor.prepare(prices, panel, preds, labels, cfg)
    if ctx is None or runs == 0:
        return {"equity_median": empty, "equity_p05": empty.copy(), "equity_p95": empty.copy(),
                "final_returns": [], "n_trades": [], "costs_eur": [], "win_rates": [], "runs": 0,
                "run_metrics": {k: [] for k in RUN_METRICS}, "metrics_median": {k: NAN for k in RUN_METRICS},
                "method_el": RANDOM_METHOD_EL}

    rng = np.random.default_rng(int(cfg.backtest.seed))
    eq = np.empty((runs, ctx.n_days))
    n_trades, costs, wins = [], [], []
    per_run = {k: [] for k in RUN_METRICS}
    for i in range(runs):
        sim = executor.simulate(ctx, executor.random_order(ctx, rng), cfg, ctx.n_want, record=False)
        eq[i] = sim["equity"]
        tr = sim["trades"]                       # (γραμμή, ποσό, κόστος €, καθαρό αποτέλεσμα €)
        n_trades.append(len(tr))
        costs.append(float(sum(t[2] for t in tr)))
        wins.append(sum(1 for t in tr if t[3] > 0) / len(tr) if tr else float("nan"))
        m = metrics(pd.Series(eq[i], index=ctx.cal), None, ctx.capital)
        for k in RUN_METRICS:
            per_run[k].append(m[k])
        if progress is not None and (i % 10 == 0 or i == runs - 1):
            try:
                progress((i + 1) / runs, f"Τυχαίες στρατηγικές: {i + 1}/{runs}")
            except Exception:  # η πρόοδος δεν πρέπει να σταματά τον υπολογισμό
                pass

    capital = float(ctx.capital)

    def series(values, name):
        return pd.Series(values, index=ctx.cal, name=name)

    def med(values):
        v = np.asarray(values, dtype="float64")
        return float(np.nanmedian(v)) if np.isfinite(v).any() else NAN

    return {
        "equity_median": series(np.median(eq, axis=0), "random_median"),
        "equity_p05": series(np.percentile(eq, 5, axis=0), "random_p05"),
        "equity_p95": series(np.percentile(eq, 95, axis=0), "random_p95"),
        "final_returns": (eq[:, -1] / capital - 1.0).tolist(),
        "n_trades": n_trades,
        "costs_eur": costs,
        "win_rates": wins,
        "run_metrics": per_run,
        "metrics_median": {k: med(v) for k, v in per_run.items()},
        "runs": runs,
        "method_el": RANDOM_METHOD_EL,
    }


# ---------------------------------------------------------------- βαθμονόμηση

def calibration(preds: pd.DataFrame, labels: pd.DataFrame, bins: int = 10) -> list[dict]:
    """Βαθμονόμηση του p_tp: προβλεπόμενη vs πραγματική συχνότητα «στόχος πρώτα» (μόνο επιλυμένα).

    Οι κάδοι έχουν ίσο πλήθος δειγμάτων (ποσοστημόρια του p_tp), ώστε κάθε σημείο να έχει
    στατιστικό βάρος. [{"bin_lo","bin_hi","predicted","realized","n"}], ταξινομημένα κατά p_tp.
    """
    if preds is None or labels is None or not len(preds) or not len(labels):
        return []
    j = preds[["p_tp"]].join(labels[["label"]], how="inner")
    j = j[np.isfinite(j["p_tp"].to_numpy(dtype="float64", na_value=np.nan))
          & np.isfinite(j["label"].to_numpy(dtype="float64", na_value=np.nan))]
    if not len(j):
        return []
    p = j["p_tp"].to_numpy(dtype="float64")
    y = (j["label"].to_numpy(dtype="float64") == LABEL_TP).astype("float64")
    bins = max(int(bins), 1)
    edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        edges = np.array([p.min(), p.max()])
    code = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        m = code == b
        if not m.any():
            continue
        out.append({"bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]), "predicted": float(p[m].mean()),
                    "realized": float(y[m].mean()), "n": int(m.sum())})
    return out


# ---------------------------------------------------------------- δεξιότητα εκτός δείγματος

def skill(preds: pd.DataFrame | None, labels: pd.DataFrame | None, *, block_days: int = SKILL_BLOCK_DAYS,
          n_boot: int = SKILL_BOOT, seed: int = 0) -> dict:
    """Είναι οι πιθανότητες walk‑forward καλύτερες από τις σταθερές ιστορικές συχνότητες;

    Σύγκριση σε όλες τις επιλυμένες προβλέψεις εκτός δείγματος: log-loss (και Brier) του μοντέλου έναντι
    ενός «μοντέλου» που δίνει πάντα τις συχνότητες των τριών καταστάσεων στο ίδιο δείγμα (σκόπιμα
    ευνοϊκή βάση: τις ξέρει εκ των υστέρων). Η αβεβαιότητα εκτιμάται με block bootstrap ανά ημερομηνία
    (μπλοκ `block_days` συνεχόμενων ημερομηνιών, επειδή οι εκβάσεις γειτονικών ημερών επικαλύπτονται).
    Επιστρέφει {"n", "n_dates", "logloss_model", "logloss_base", "brier_model", "brier_base",
    "improvement" (σχετική βελτίωση log-loss), "p_value" (μερίδιο bootstrap χωρίς βελτίωση), "ok"}.
    """
    out = {"n": 0, "n_dates": 0, "logloss_model": NAN, "logloss_base": NAN, "brier_model": NAN,
           "brier_base": NAN, "improvement": NAN, "p_value": NAN, "ok": False, "block_days": int(block_days)}
    if preds is None or labels is None or not len(preds) or not len(labels):
        return out
    cols = ["p_tp", "p_sl", "p_to"]
    j = preds[cols].join(labels[["label"]], how="inner")
    P = j[cols].to_numpy(dtype="float64", na_value=np.nan)
    y = j["label"].to_numpy(dtype="float64", na_value=np.nan)
    ok = np.isfinite(P).all(axis=1) & np.isin(y, (0, 1, 2))
    if ok.sum() < 50:
        return out
    P, y = P[ok], y[ok].astype("int64")
    dates = np.asarray(j.index.get_level_values("date"))[ok]
    P = np.clip(P, 1e-6, 1.0)
    P /= P.sum(axis=1, keepdims=True)
    base = np.bincount(y, minlength=3).astype("float64") / len(y)
    base = np.clip(base, 1e-6, 1.0)
    onehot = np.eye(3)[y]
    ll_m = -np.log(P[np.arange(len(y)), y])
    ll_b = -np.log(base[y])
    br_m = ((P - onehot) ** 2).sum(axis=1)
    br_b = ((base[None, :] - onehot) ** 2).sum(axis=1)
    diff = ll_b - ll_m                                      # > 0: το μοντέλο είναι καλύτερο
    uniq, code = np.unique(dates, return_inverse=True)
    d_sum = np.bincount(code, weights=diff, minlength=len(uniq))
    d_cnt = np.bincount(code, minlength=len(uniq)).astype("float64")
    n_dates = len(uniq)
    out.update(n=int(len(y)), n_dates=int(n_dates), logloss_model=float(ll_m.mean()),
               logloss_base=float(ll_b.mean()), brier_model=float(br_m.mean()), brier_base=float(br_b.mean()),
               improvement=float(1.0 - ll_m.mean() / ll_b.mean()) if ll_b.mean() > 0 else NAN)
    block = max(1, min(int(block_days), n_dates))
    n_blocks = int(math.ceil(n_dates / block))
    rng = np.random.default_rng(int(seed))
    starts = rng.integers(0, n_dates - block + 1, size=(int(n_boot), n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(int(n_boot), -1)[:, :n_dates]
    boot = d_sum[idx].sum(axis=1) / np.maximum(d_cnt[idx].sum(axis=1), 1.0)
    p_value = float((np.sum(boot <= 0.0) + 1) / (len(boot) + 1))
    out["p_value"] = p_value
    out["ok"] = bool(diff.mean() > 0 and p_value <= SKILL_P_VALUE)
    return out


# ---------------------------------------------------------------- ετυμηγορία

def verdict(strategy_metrics: dict, random_final_returns, bh_metrics: dict | None, *,
            bh_universe_metrics: dict | None = None, skill_result: dict | None = None) -> dict:
    """Τίμια ετυμηγορία.

    * `percentile` = μέρος των τυχαίων εκτελέσεων που ξεπερνά η στρατηγική (οι ισοπαλίες μετρούν μισές)·
      `p_value` = (1 + #{τυχαίες ≥ στρατηγική}) / (1 + εκτελέσεις).
    * `edge` μόνο αν p ≤ 5% με ≥ MIN_RANDOM_RUNS εκτελέσεις **και** νικά την αγορά και διακράτηση του
      δείκτη **και** του ισοσταθμικού χαρτοφυλακίου της λίστας (`bh_universe_metrics`, αν δοθεί) **και**
      είναι κερδοφόρα **και** έχει ≥ 30 συναλλαγές **και** (αν δοθεί `skill_result`) το μοντέλο έχει
      δεξιότητα εκτός δείγματος.
    * Επίπεδα: "good" (edge), "weak" (ενδείξεις χωρίς απόδειξη), "none".
    """
    s = _f((strategy_metrics or {}).get("total_return"))
    bh = _f((bh_metrics or {}).get("total_return"))
    bhu = _f((bh_universe_metrics or {}).get("total_return"))
    n_trades = int((strategy_metrics or {}).get("n_trades") or 0)
    arr = np.asarray(list(random_final_returns or []), dtype="float64")
    arr = arr[np.isfinite(arr)]
    n_rand = int(len(arr))

    if n_rand and math.isfinite(s):
        percentile = float(((arr < s).sum() + 0.5 * (arr == s).sum()) / n_rand)
        p_value = float((1 + (arr >= s).sum()) / (1 + n_rand))
        rand_median = float(np.median(arr))
    else:
        percentile, p_value, rand_median = NAN, NAN, NAN
    benchmarks_ = [(v, lbl) for v, lbl in ((bh, "δείκτης"), (bhu, "ισοσταθμικό χαρτοφυλάκιο της λίστας"))
                   if math.isfinite(v)]
    beats_bh = bool(math.isfinite(s) and benchmarks_ and all(s > v for v, _ in benchmarks_))
    enough = n_trades >= MIN_TRADES_FOR_VERDICT
    enough_runs = n_rand >= MIN_RANDOM_RUNS
    profitable = bool(math.isfinite(s) and s > 0)
    has_skill = True if skill_result is None else bool(skill_result.get("ok"))
    beats_random = bool(math.isfinite(p_value) and p_value <= EDGE_P_VALUE)
    edge = bool(beats_random and enough_runs and beats_bh and enough and profitable and has_skill)

    pct_txt = f"{fmt_num(percentile * 100.0, 0)}%" if math.isfinite(percentile) else "—"
    worse_txt = f"{fmt_num((1.0 - percentile) * 100.0, 0)}%" if math.isfinite(percentile) else "—"
    s_txt = fmt_pct(s, 1, sign=True)
    rm_txt = fmt_pct(rand_median, 1, sign=True)
    bh_parts = []
    if math.isfinite(bh):
        bh_parts.append(f"δείκτης {fmt_pct(bh, 1, sign=True)}")
    if math.isfinite(bhu):
        bh_parts.append(f"ισοσταθμική λίστα {fmt_pct(bhu, 1, sign=True)}")
    bh_txt = ", ".join(bh_parts) or "—"
    skill_txt = ""
    if skill_result is not None and math.isfinite(_f(skill_result.get("logloss_model"))):
        skill_txt = (f" Εκτός δείγματος οι πιθανότητες του μοντέλου δεν είναι καλύτερες από τις σταθερές "
                     f"ιστορικές συχνότητες (log-loss {fmt_num(skill_result['logloss_model'], 3)} έναντι "
                     f"{fmt_num(skill_result['logloss_base'], 3)}).")
    elif skill_result is not None:
        skill_txt = " Δεν υπάρχουν αρκετές επιλυμένες προβλέψεις για να ελεγχθεί η δεξιότητα του μοντέλου."
    if n_trades == 0:
        level = "none"
        text = "Η στρατηγική δεν έκανε καμία συναλλαγή στην περίοδο — δεν μπορεί να κριθεί."
    elif not n_rand or not math.isfinite(percentile):
        level = "none"
        text = "Δεν υπάρχουν τυχαίες στρατηγικές για σύγκριση — δεν μπορεί να κριθεί αν υπάρχει πλεονέκτημα."
    elif not enough_runs:
        level = "none"
        text = (f"Μόνο {n_rand} τυχαίες στρατηγικές για σύγκριση — χρειάζονται τουλάχιστον {MIN_RANDOM_RUNS}. "
                f"Με τόσο λίγες, το να τις ξεπεράσει όλες μια στρατηγική χωρίς δεξιότητα είναι συχνό "
                f"(πιθανότητα 1 στις {n_rand + 1}). Τρέξτε το backtest με περισσότερες εκτελέσεις.")
    elif not enough:
        level = "none"
        text = (f"Μόνο {n_trades} συναλλαγές — πολύ λίγες για ασφαλές συμπέρασμα "
                f"(χρειάζονται τουλάχιστον {MIN_TRADES_FOR_VERDICT}). Η στρατηγική ξεπέρασε το {pct_txt} "
                f"των τυχαίων στρατηγικών, αλλά αυτό μπορεί να είναι τύχη.")
    elif edge:
        level = "good"
        text = (f"Η στρατηγική ({s_txt}) ξεπέρασε το {pct_txt} των {n_rand} τυχαίων στρατηγικών με τους ίδιους "
                f"κανόνες και κόστη (διάμεσος {rm_txt}) και την αγορά και διακράτηση ({bh_txt}), και οι "
                f"πιθανότητές της είναι καλύτερες από τις ιστορικές συχνότητες εκτός δείγματος. "
                f"Υπάρχει ένδειξη πραγματικού πλεονεκτήματος στο παρελθόν — όχι εγγύηση για το μέλλον.")
    elif beats_random and not has_skill:
        level = "none"
        text = (f"Η στρατηγική ({s_txt}) ξεπέρασε το {pct_txt} των τυχαίων στρατηγικών, αλλά αυτό δεν "
                f"στηρίζεται σε πρόβλεψη.{skill_txt} Πιθανότερα πρόκειται για τύχη ή για διαφορετική έκθεση "
                f"στην αγορά — όχι για πλεονέκτημα.")
    elif beats_random and not profitable:
        level = "weak"
        text = (f"Η στρατηγική ξεπέρασε το {pct_txt} των τυχαίων στρατηγικών, αλλά συνολικά έχασε χρήματα "
                f"({s_txt}) μετά από κόστη. Το να χάνει λιγότερο από την τύχη δεν είναι πλεονέκτημα.")
    elif beats_random:
        level = "weak"
        text = (f"Η στρατηγική ({s_txt}) ξεπέρασε το {pct_txt} των τυχαίων στρατηγικών, αλλά όχι την αγορά "
                f"και διακράτηση ({bh_txt}). Δεν αρκεί για να μιλήσουμε για πλεονέκτημα.")
    elif percentile >= WEAK_PERCENTILE and has_skill:
        level = "weak"
        text = (f"Η στρατηγική ({s_txt}) ξεπέρασε το {pct_txt} των τυχαίων στρατηγικών (διάμεσος {rm_txt}): "
                f"καλύτερα από τον μέσο όρο, αλλά όχι αρκετά (απαιτείται τουλάχιστον "
                f"{fmt_num(EDGE_PERCENTILE * 100, 0)}%) για να αποκλειστεί η τύχη.")
    elif percentile >= WEAK_PERCENTILE:
        level = "none"
        text = (f"Η στρατηγική ({s_txt}) ξεπέρασε το {pct_txt} των τυχαίων στρατηγικών (διάμεσος {rm_txt}) — "
                f"όχι αρκετά για να αποκλειστεί η τύχη.{skill_txt}")
    elif percentile < POOR_PERCENTILE:
        level = "none"
        text = (f"Η στρατηγική ({s_txt}) τα πήγε χειρότερα από το {worse_txt} των τυχαίων επιλογών με τους "
                f"ίδιους κανόνες και κόστη (διάμεσος {rm_txt}). Οι επιλογές του μοντέλου δεν βοήθησαν — "
                f"αντίθετα.")
    else:
        level = "none"
        text = (f"Η στρατηγική ({s_txt}) δεν ξεχωρίζει από τυχαίες επιλογές με τους ίδιους κανόνες και κόστη: "
                f"ξεπέρασε μόνο το {pct_txt} τους (διάμεσος {rm_txt}).")
    return {"edge": edge, "percentile": percentile, "p_value": p_value, "text_el": text, "level": level,
            "beats_buy_hold": beats_bh, "profitable": profitable, "n_random": n_rand, "n_trades": n_trades,
            "min_random_runs": MIN_RANDOM_RUNS, "skill_ok": has_skill if skill_result is not None else None,
            "strategy_return": s, "random_median_return": rand_median, "buy_hold_return": bh,
            "buy_hold_universe_return": bhu}
