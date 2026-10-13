"""«Εκτελεστής»: backtest χαρτοφυλακίου με τους ίδιους κανόνες εισόδου/εξόδου με τις ετικέτες.

Ροή κάθε ημέρας t του ημερολογίου:
1. Κλείνουν οι θέσεις που βγαίνουν την t (στόχος/stop μέσα στην ημέρα ή λήξη χρόνου στο κλείσιμο).
2. Στο κλείσιμο της t: υποψήφιες = γραμμές με πρόβλεψη την t που περνούν τους ελέγχους μετοχής
   και απόφασης, με σειρά κατά ev_net (φθίνουσα). Για καθεμία: έλεγχοι χαρτοφυλακίου, μέγεθος θέσης
   (παράλειψη αν < 300 €), δέσμευση μετρητών για αγορά στο άνοιγμα της t+1 (+ κόστος αγοράς).
3. Αποτίμηση στο κλείσιμο (θέσεις που δεν έχουν ακόμη ανοίξει μετρούν με τα δεσμευμένα μετρητά).
Η έξοδος κάθε θέσης υπολογίζεται με τον κοινό προσομοιωτή (`labels.simulate_all`, ίδιο με το
`simulate_exit`) από τις τιμές και τα τρέχοντα όρια του cfg. Θέσεις ανοιχτές στο τέλος κλείνουν στο
τελευταίο κλείσιμο (label −1). Κάθε πλευρά πληρώνει `quant.side_cost_pct` (μέσω της ισοδύναμης,
ταχύτερης `quant.side_cost_total`).

Τυχαία επιλογή (`selector="random"`, για το benchmark): ίδιο σύνολο υποψηφίων *πριν* από τους
ελέγχους απόφασης· κάθε ημέρα εξετάζονται τόσες υποψήφιες όσες εξέτασε και το μοντέλο (όσες πέρασαν τους
δικούς του ελέγχους απόφασης), με τους ίδιους ελέγχους χαρτοφυλακίου, μέγεθος θέσης και κόστη. Η σειρά
προκύπτει από ev_net με το πραγματικό κόστος/ATR κάθε μετοχής, αλλά με τις προβλέψεις μιας άλλης, τυχαία
αντιστοιχισμένης μετοχής (σταθερή μετάθεση σε όλη την εκτέλεση). Έτσι η μόνη διαφορά από τη στρατηγική
είναι η πληροφορία των προβλέψεων — όχι η αποφυγή ακριβών μετοχών (που δεν χρειάζεται μοντέλο), ούτε ο
αριθμός των συναλλαγών, ούτε η «συγκέντρωση» σε λίγες μετοχές (βλ. `random_order`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import guard, quant
from .labels import simulate_all

LABEL_MARKED = -1               # θέση που έκλεισε αναγκαστικά στο τέλος της περιόδου (αποτίμηση)
SELECTORS = ("model", "random")

# Κωδικοί αιτιολογίας αποφάσεων
R_TAKEN = 0
R_EV = 1
R_PTP = 2
R_MAXPOS = 3
R_SECTOR = 4
R_HELD = 5
R_CASH = 6
R_NO_ENTRY = 7
R_NOT_PICKED = 8
_GUARD_BASE = 20                # 20 + θέση στο guard.STOCK_CHECK_IDS
_GUARD_IDS = [c for c in guard.STOCK_CHECK_IDS if c != "regime"]

REASONS_EL = {
    R_TAKEN: "Άνοιγμα θέσης στο επόμενο άνοιγμα",
    R_EV: "Χαμηλή αναμενόμενη απόδοση μετά από κόστη",
    R_PTP: "Χαμηλή πιθανότητα στόχου",
    R_MAXPOS: "Συμπληρωμένο χαρτοφυλάκιο (μέγιστος αριθμός θέσεων)",
    R_SECTOR: "Συμπληρώθηκε το όριο θέσεων του κλάδου",
    R_HELD: "Υπάρχει ήδη ανοιχτή θέση στη μετοχή",
    R_CASH: "Ανεπαρκή διαθέσιμα για νέα θέση",
    R_NO_ENTRY: "Δεν υπάρχει επόμενη συνεδρίαση για είσοδο",
    R_NOT_PICKED: "Δεν επιλέχθηκε (τυχαία επιλογή)",
}
for _i, _cid in enumerate(_GUARD_IDS):
    REASONS_EL[_GUARD_BASE + _i] = f"Φύλακας: {guard.CHECK_LABELS_EL[_cid]}"

TRADE_COLUMNS = ["ticker", "sector", "signal_date", "entry_date", "entry_price", "exit_date", "exit_price",
                 "label", "size_eur", "gross_ret", "cost_eur", "net_pnl_eur", "net_ret", "p_tp", "p_sl", "p_to",
                 "ev_net", "bars_held"]
DECISION_COLUMNS = ["date", "ticker", "taken", "reason_el"]


@dataclass
class BacktestContext:
    """Προϋπολογισμένοι πίνακες (κοινοί για τη στρατηγική και όλες τις τυχαίες εκτελέσεις)."""
    cal: pd.DatetimeIndex                       # ημερολόγιο backtest
    tickers: list[str]
    sectors: list[str]
    close: np.ndarray                           # (ημέρες, μετοχές), με μεταφορά τελευταίας τιμής
    capital: float
    # ανά γραμμή πρόβλεψης (ταξινομημένες κατά ημέρα)
    index: pd.MultiIndex
    day: np.ndarray
    tix: np.ndarray
    sec: np.ndarray
    adv: np.ndarray
    vol: np.ndarray
    tp_ret: np.ndarray
    sl_ret: np.ndarray
    cost_rt: np.ndarray
    p: np.ndarray                               # (n, 3)
    timeout_mean: np.ndarray
    ev_net: np.ndarray
    entry_ok: np.ndarray
    entry_cal: np.ndarray
    entry_price: np.ndarray
    exit_cal: np.ndarray
    exit_price: np.ndarray
    label: np.ndarray
    bars_held: np.ndarray
    reason0: np.ndarray                         # αιτιολογία πριν από τους ελέγχους χαρτοφυλακίου (R_TAKEN = υποψήφια)
    stock_ok: np.ndarray
    candidate: np.ndarray
    n_want: np.ndarray                          # νέες θέσεις που θα ήθελε το μοντέλο ανά ημέρα
    row_of: np.ndarray                          # (ημέρες, μετοχές) → γραμμή πρόβλεψης ή −1
    _lists: dict = field(default_factory=dict)  # cache Python λιστών για τον βρόχο

    @property
    def n_days(self) -> int:
        return len(self.cal)


def _empty_result(capital: float) -> dict:
    from . import benchmarks

    equity = pd.Series([], dtype="float64", index=pd.DatetimeIndex([], name="date").as_unit("us"), name="equity")
    trades = pd.DataFrame({c: [] for c in TRADE_COLUMNS})
    decisions = pd.DataFrame({c: [] for c in DECISION_COLUMNS})
    return {"equity": equity, "trades": trades, "decisions": decisions,
            "metrics": benchmarks.metrics(equity, trades, capital)}


def prepare(prices: dict[str, pd.DataFrame], panel: pd.DataFrame, preds: pd.DataFrame, labels, cfg
            ) -> BacktestContext | None:
    """Προϋπολογισμός όλων όσων δεν εξαρτώνται από την κατάσταση του χαρτοφυλακίου.

    `labels` γίνεται δεκτό για συμβατότητα· οι έξοδοι υπολογίζονται ξανά από τις τιμές με τα όρια του cfg,
    ώστε το backtest να ακολουθεί πάντα τις τρέχουσες ρυθμίσεις.
    """
    if preds is None or not len(preds) or panel is None or not len(panel):
        return None
    pcols = ["p_tp", "p_sl", "p_to"]
    p_all = preds[pcols].to_numpy(dtype="float64", na_value=np.nan)
    keep = np.isfinite(p_all).all(axis=1)
    pidx = preds.index[keep]
    loc = panel.index.get_indexer(pidx)
    in_panel = loc >= 0
    with_prices = [t for t, df in prices.items() if df is not None and len(df)]
    has_px = np.asarray(pidx.get_level_values("ticker").isin(with_prices), dtype=bool)
    ok = in_panel & has_px
    if not ok.any():
        return None
    sel = np.flatnonzero(keep)[ok]
    pidx, loc = preds.index[sel], loc[ok]
    order = np.lexsort((np.asarray(pidx.get_level_values("ticker")), np.asarray(pidx.get_level_values("date"))))
    sel, pidx, loc = sel[order], pidx[order], loc[order]
    pn = panel.iloc[loc]

    sig_dates = pd.DatetimeIndex(pidx.get_level_values("date"))
    first, last = sig_dates.min(), sig_dates.max()
    tickers = sorted(set(pidx.get_level_values("ticker")))
    cal_parts = [prices[t].index[(prices[t].index >= first) & (prices[t].index <= last)] for t in tickers]
    cal = pd.DatetimeIndex(np.unique(np.concatenate([np.asarray(c.values, dtype="datetime64[us]")
                                                     for c in cal_parts] + [np.asarray(sig_dates.values,
                                                                                       dtype="datetime64[us]")])),
                           name="date")
    n_cal = len(cal)
    t_pos = {t: i for i, t in enumerate(tickers)}
    tix = np.asarray([t_pos[t] for t in pidx.get_level_values("ticker")], dtype="int64")
    day = cal.get_indexer(sig_dates)

    sector_col = pn["sector"].astype("str").to_numpy() if "sector" in pn.columns else np.full(len(pn), "Άλλο")
    sectors = sorted(set(sector_col.tolist()))
    s_pos = {s: i for i, s in enumerate(sectors)}
    sec = np.asarray([s_pos[s] for s in sector_col], dtype="int64")

    close = np.full((n_cal, len(tickers)), np.nan)
    n = len(pidx)
    entry_ok = np.zeros(n, dtype=bool)
    entry_cal = np.full(n, -1, dtype="int64")
    entry_price = np.full(n, np.nan)
    exit_cal = np.full(n, -1, dtype="int64")
    exit_price = np.full(n, np.nan)
    label = np.full(n, LABEL_MARKED, dtype="int64")
    bars_held = np.zeros(n, dtype="int64")

    k_tp, k_sl = float(cfg.barriers.k_tp), float(cfg.barriers.k_sl)
    horizon = int(cfg.barriers.horizon_days)
    atr_pct_rows = pn["atr_pct"].to_numpy(dtype="float64", na_value=np.nan)
    panel_atr = panel["atr_pct"]
    last_cal = cal[-1]
    for t, ti in t_pos.items():
        df = prices[t]
        idx = df.index
        close[:, ti] = df["close"].reindex(cal).ffill().to_numpy(dtype="float64", na_value=np.nan)
        # ATR% ανά ημέρα της μετοχής (από το panel)
        try:
            atr_t = panel_atr.xs(t, level="ticker")
        except KeyError:
            atr_t = pd.Series(dtype="float64")
        atr = atr_t.reindex(idx).to_numpy(dtype="float64", na_value=np.nan)
        r = simulate_all(df["open"].to_numpy("float64"), df["high"].to_numpy("float64"),
                         df["low"].to_numpy("float64"), df["close"].to_numpy("float64"),
                         atr, k_tp, k_sl, horizon)
        rows = np.flatnonzero(tix == ti)
        lt = idx.get_indexer(sig_dates[rows])
        good = lt >= 0
        rows, lt = rows[good], lt[good]
        n_t = len(idx)
        last_local = int(idx.searchsorted(last_cal, side="right")) - 1     # τελευταία ημέρα της μετοχής ≤ τέλος
        has_next = lt + 1 <= last_local
        valid = has_next & r["valid"][lt]
        rows, lt = rows[valid], lt[valid]
        if not len(rows):
            continue
        ent_local = lt + 1
        entry_ok[rows] = True
        entry_price[rows] = r["entry_price"][lt]
        entry_cal[rows] = cal.get_indexer(idx[ent_local])
        res = r["resolved"][lt]
        ex_local = r["exit_idx"][lt]
        natural = res & (ex_local <= last_local)
        nr, nl = rows[natural], ex_local[natural]
        exit_cal[nr] = cal.get_indexer(idx[nl])
        exit_price[nr] = r["exit_price"][lt[natural]]
        label[nr] = r["label"][lt[natural]].astype("int64")
        bars_held[nr] = nl - ent_local[natural] + 1
        # αναγκαστικό κλείσιμο στο τέλος της περιόδου, στο τελευταίο διαθέσιμο κλείσιμο
        fr = rows[~natural]
        exit_cal[fr] = n_cal - 1
        exit_price[fr] = float(df["close"].iloc[last_local]) if 0 <= last_local < n_t else np.nan
        label[fr] = LABEL_MARKED
        bars_held[fr] = last_local - ent_local[~natural] + 1
    entry_ok &= np.isfinite(entry_price) & (entry_price > 0) & np.isfinite(exit_price) & (entry_cal >= 0)

    # έλεγχοι μετοχής (διανυσματικά) και αναμενόμενη αξία με ονομαστικό μέγεθος θέσης
    checks = guard.stock_checks_vec(pn, cfg)
    reason0 = np.full(n, R_TAKEN, dtype="int64")
    for i in range(len(_GUARD_IDS) - 1, -1, -1):      # ο πρώτος αποτυχημένος έλεγχος κερδίζει
        reason0[~checks[_GUARD_IDS[i]]] = _GUARD_BASE + i
    stock_ok = reason0 == R_TAKEN

    def col(name):
        return pn[name].to_numpy(dtype="float64", na_value=np.nan) if name in pn.columns else np.full(n, np.nan)

    adv = col("adv20_eur")
    vol = col("vol_20")
    capital = float(cfg.portfolio.capital_eur)
    nominal = np.minimum(np.minimum(cfg.portfolio.position_eur_max, capital),
                         cfg.portfolio.max_adv_fraction * np.where(np.isfinite(adv), adv, 0.0))
    cost_rt = quant.round_trip_cost_pct_vec(nominal, adv, cfg, vol=vol)["total"]
    tp_ret, sl_ret = k_tp * atr_pct_rows, k_sl * atr_pct_rows
    p = p_all[sel]
    tm = preds["timeout_mean"].to_numpy(dtype="float64", na_value=np.nan)[sel] if "timeout_mean" in preds.columns \
        else np.zeros(n)
    tm = np.where(np.isfinite(tm), tm, 0.0)
    ev_net = quant.expected_value_vec(p[:, 0], p[:, 1], p[:, 2], tp_ret, sl_ret, tm, cost_rt)["ev_net"]

    with np.errstate(invalid="ignore"):
        ev_ok = np.isfinite(ev_net) & (ev_net >= cfg.guard.min_ev_net)
        ptp_ok = p[:, 0] >= cfg.guard.min_p_tp
    reason0[stock_ok & ~ev_ok] = R_EV
    reason0[stock_ok & ev_ok & ~ptp_ok] = R_PTP
    candidate = reason0 == R_TAKEN
    n_want = np.bincount(day[candidate], minlength=n_cal).astype("int64")
    row_of = np.full((n_cal, len(tickers)), -1, dtype="int64")
    row_of[day, tix] = np.arange(n)

    return BacktestContext(
        cal=cal, tickers=tickers, sectors=sectors, close=close, capital=capital, index=pidx,
        day=day, tix=tix, sec=sec, adv=adv, vol=vol, tp_ret=tp_ret, sl_ret=sl_ret, cost_rt=cost_rt, p=p,
        timeout_mean=tm, ev_net=ev_net, entry_ok=entry_ok, entry_cal=entry_cal, entry_price=entry_price,
        exit_cal=exit_cal, exit_price=exit_price, label=label, bars_held=bars_held, reason0=reason0,
        stock_ok=stock_ok, candidate=candidate, n_want=n_want, row_of=row_of,
    )


def _day_lists(ctx: BacktestContext, rows: np.ndarray, score: np.ndarray) -> list[list[int]]:
    """Για κάθε ημέρα: οι γραμμές `rows` ταξινομημένες κατά `score` (φθίνουσα)."""
    d = ctx.day[rows]
    order = np.lexsort((-score, d))
    rows, d = rows[order], d[order]
    bounds = np.searchsorted(d, np.arange(ctx.n_days + 1))
    flat = rows.tolist()
    return [flat[bounds[k]:bounds[k + 1]] for k in range(ctx.n_days)]


def model_order(ctx: BacktestContext) -> list[list[int]]:
    rows = np.flatnonzero(ctx.candidate)
    return _day_lists(ctx, rows, ctx.ev_net[rows])


def random_order(ctx: BacktestContext, rng: np.random.Generator, mode: str = "ticker") -> list[list[int]]:
    """Τυχαία κατάταξη των υποψηφίων (μετά τους ελέγχους μετοχής) για μία τυχαία εκτέλεση.

    mode="ticker" (προεπιλογή): τυχαία μετάθεση π των μετοχών, σταθερή σε όλη την εκτέλεση· η μετοχή s
    παίρνει κάθε ημέρα τις προβλέψεις της π(s) της ίδιας ημέρας (αν η π(s) δεν έχει πρόβλεψη, η s δεν
    εξετάζεται εκείνη την ημέρα). Κρατά τη «συγκέντρωση» του μοντέλου (που συχνά προτιμά τις ίδιες μετοχές
    για καιρό), οπότε η διασπορά των τυχαίων αποτελεσμάτων είναι συγκρίσιμη με της στρατηγικής.
    mode="day": οι προβλέψεις κάθε ημέρας ανακατεύονται ανεξάρτητα ανάμεσα στις υποψήφιες.
    Η σειρά προκύπτει από το ev_net με το κόστος/ATR της κάθε μετοχής.
    """
    pool = np.flatnonzero(ctx.stock_ok)
    if not len(pool):
        return [[] for _ in range(ctx.n_days)]
    if mode == "ticker":
        perm = rng.permutation(len(ctx.tickers))
        src = ctx.row_of[ctx.day[pool], perm[ctx.tix[pool]]]
        has = src >= 0
        pool, src = pool[has], src[has]
    elif mode == "day":
        d = ctx.day[pool]                               # ήδη ταξινομημένες κατά ημέρα
        src = pool[np.lexsort((rng.random(len(pool)), d))]
    else:
        raise ValueError(f"Άγνωστος τρόπος τυχαίας επιλογής «{mode}»")
    ev = quant.expected_value_vec(ctx.p[src, 0], ctx.p[src, 1], ctx.p[src, 2], ctx.tp_ret[pool],
                                  ctx.sl_ret[pool], ctx.timeout_mean[src], ctx.cost_rt[pool])["ev_net"]
    ev = np.where(np.isfinite(ev), ev, -np.inf)
    return _day_lists(ctx, pool, ev)


def _lists(ctx: BacktestContext) -> dict:
    if not ctx._lists:
        ctx._lists = {k: getattr(ctx, k).tolist() for k in (
            "tix", "sec", "adv", "vol", "entry_ok", "entry_cal", "entry_price", "exit_cal", "exit_price", "label",
            "bars_held")}
    return ctx._lists


def _buy_pct(size: float, adv: float, cfg, vol: float) -> float:
    return quant.side_cost_total(size, adv, cfg, "buy", vol)


def simulate(ctx: BacktestContext, order: list[list[int]], cfg, n_want: np.ndarray | None = None,
             record: bool = True) -> dict:
    """Ο βρόχος ημερών. `order[k]`: γραμμές προς εξέταση την ημέρα k, με σειρά προτίμησης.
    `n_want[k]` (προαιρετικό): πόσες από τις πρώτες γραμμές εξετάζονται την ημέρα k (οι υπόλοιπες
    αγνοούνται)· όσες αποτύχουν στους ελέγχους χαρτοφυλακίου χάνονται, όπως και στο μοντέλο."""
    L = _lists(ctx)
    tix, sec, adv, vol = L["tix"], L["sec"], L["adv"], L["vol"]
    entry_ok, entry_cal, entry_price = L["entry_ok"], L["entry_cal"], L["entry_price"]
    exit_cal, exit_price = L["exit_cal"], L["exit_price"]
    close = ctx.close
    max_pos = int(cfg.portfolio.max_positions)
    max_sec = int(cfg.portfolio.max_per_sector)
    min_pos = guard.MIN_POSITION_EUR
    n_days = ctx.n_days

    want = n_want.tolist() if n_want is not None else None
    cash = ctx.capital
    equity = np.empty(n_days)
    exits_at: list[list] = [[] for _ in range(n_days)]
    open_pos: list[list] = []      # [row, ticker, sector, shares, size, buy_cost, entry_cal]
    held: set[int] = set()
    sec_count = [0] * len(ctx.sectors)
    reasons = ctx.reason0.copy() if record else None
    trades: list[tuple] = []

    for k in range(n_days):
        # 1. έξοδοι της ημέρας
        if exits_at[k]:
            for pos in exits_at[k]:
                r, t, s, shares, size, buy_cost = pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]
                px = exit_price[r]
                proceeds = shares * px
                sell_cost = proceeds * quant.side_cost_total(proceeds, adv[r], cfg, "sell", vol[r])
                cash += proceeds - sell_cost
                for i, q in enumerate(open_pos):
                    if q is pos:
                        del open_pos[i]
                        break
                held.discard(t)
                sec_count[s] -= 1
                trades.append((r, size, buy_cost + sell_cost, proceeds - sell_cost - size - buy_cost))

        # 2. αποφάσεις στο κλείσιμο
        rows = order[k]
        if want is not None and len(rows) > want[k]:
            # εξετάζονται μόνο οι πρώτες n_want[k] (όσες υποψήφιες εξέτασε και το μοντέλο εκείνη την ημέρα)
            if record:
                for rr in rows[want[k]:]:
                    reasons[rr] = R_NOT_PICKED
            rows = rows[:want[k]]
        if rows:
            for j, r in enumerate(rows):
                if len(open_pos) >= max_pos:
                    if record:
                        for rr in rows[j:]:
                            reasons[rr] = R_MAXPOS
                    break
                if cash < min_pos:
                    if record:
                        for rr in rows[j:]:
                            reasons[rr] = R_CASH
                    break
                if not entry_ok[r]:
                    if record:
                        reasons[r] = R_NO_ENTRY
                    continue
                t = tix[r]
                if t in held:
                    if record:
                        reasons[r] = R_HELD
                    continue
                s = sec[r]
                if sec_count[s] >= max_sec:
                    if record:
                        reasons[r] = R_SECTOR
                    continue
                price = entry_price[r]
                size = guard.position_size_eur(cash, adv[r], cfg)
                if size < min_pos:
                    if record:
                        reasons[r] = R_CASH
                    continue
                pct = _buy_pct(size, adv[r], cfg, vol[r])
                if size * (1.0 + pct) > cash:
                    # τα μετρητά πρέπει να καλύπτουν και το κόστος αγοράς: επειδή το κόστος σε € αυξάνει
                    # με το ποσό, το «μετρητά − κόστος(μετρητά)» χωρά πάντα
                    size = cash - cash * _buy_pct(cash, adv[r], cfg, vol[r])
                    if size < min_pos:
                        if record:
                            reasons[r] = R_CASH
                        continue
                    pct = _buy_pct(size, adv[r], cfg, vol[r])
                buy_cost = size * pct
                cash -= size + buy_cost
                pos = [r, t, s, size / price, size, buy_cost, entry_cal[r]]
                open_pos.append(pos)
                exits_at[exit_cal[r]].append(pos)
                held.add(t)
                sec_count[s] += 1
                if record:
                    reasons[r] = R_TAKEN

        # 3. αποτίμηση στο κλείσιμο
        val = cash
        for pos in open_pos:
            if k >= pos[6]:
                val += pos[3] * close[k, pos[1]]
            else:
                val += pos[4] + pos[5]            # εντολή για το επόμενο άνοιγμα: δεσμευμένα μετρητά
        equity[k] = val

    return {"equity": equity, "trades": trades, "reasons": reasons, "cash": cash}


def _trades_frame(ctx: BacktestContext, raw: list[tuple]) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame({c: pd.Series([], dtype="float64") for c in TRADE_COLUMNS})
    r = np.asarray([x[0] for x in raw], dtype="int64")
    size = np.asarray([x[1] for x in raw], dtype="float64")
    cost = np.asarray([x[2] for x in raw], dtype="float64")
    pnl = np.asarray([x[3] for x in raw], dtype="float64")
    cal = ctx.cal
    sig = ctx.index[r]
    df = pd.DataFrame({
        "ticker": np.asarray(sig.get_level_values("ticker"), dtype=object),
        "sector": np.asarray([ctx.sectors[s] for s in ctx.sec[r]], dtype=object),
        "signal_date": pd.DatetimeIndex(sig.get_level_values("date")),
        "entry_date": cal[ctx.entry_cal[r]],
        "entry_price": ctx.entry_price[r],
        "exit_date": cal[ctx.exit_cal[r]],
        "exit_price": ctx.exit_price[r],
        "label": ctx.label[r],
        "size_eur": size,
        "gross_ret": ctx.exit_price[r] / ctx.entry_price[r] - 1.0,
        "cost_eur": cost,
        "net_pnl_eur": pnl,
        "net_ret": pnl / size,
        "p_tp": ctx.p[r, 0], "p_sl": ctx.p[r, 1], "p_to": ctx.p[r, 2],
        "ev_net": ctx.ev_net[r],
        "bars_held": ctx.bars_held[r],
    })
    df["ticker"] = df["ticker"].astype("str")
    df["sector"] = df["sector"].astype("str")
    return df.sort_values(["entry_date", "ticker"], kind="stable").reset_index(drop=True)


def _decisions_frame(ctx: BacktestContext, reasons: np.ndarray) -> pd.DataFrame:
    codes = pd.Series(reasons)
    return pd.DataFrame({
        "date": pd.DatetimeIndex(ctx.index.get_level_values("date")),
        "ticker": pd.Index(ctx.index.get_level_values("ticker")).astype("str"),
        "taken": reasons == R_TAKEN,
        "reason_el": codes.map(REASONS_EL).astype("str").to_numpy(),
    })


def _as_rng(rng, cfg) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        return np.random.default_rng(int(cfg.backtest.seed))
    return np.random.default_rng(rng)


def run_backtest(prices, panel, preds, labels, cfg, selector: str = "model", rng=None,
                 context: BacktestContext | None = None, record: bool = True) -> dict:
    """Backtest χαρτοφυλακίου. Επιστρέφει {"equity", "trades", "decisions", "metrics"}.

    `selector="model"`: κατάταξη κατά ev_net· `"random"`: τυχαία κατάταξη (βλ. `random_order`), με τόσες
    υποψήφιες ανά ημέρα όσες εξέτασε το μοντέλο (βλ. docstring του module). `rng`: Generator, σπόρος ή None
    (= cfg.backtest.seed). `context`: προϋπολογισμένο `prepare(...)`
    (για επαναλαμβανόμενες εκτελέσεις). `record=False`: χωρίς πίνακα αποφάσεων (ταχύτερο).
    """
    from . import benchmarks

    if selector not in SELECTORS:
        raise ValueError(f"Άγνωστος επιλογέας «{selector}» (επιτρέπονται: model, random)")
    capital = float(cfg.portfolio.capital_eur)
    ctx = context if context is not None else prepare(prices, panel, preds, labels, cfg)
    if ctx is None:
        return _empty_result(capital)

    if selector == "model":
        sim = simulate(ctx, model_order(ctx), cfg, None, record)
    else:
        sim = simulate(ctx, random_order(ctx, _as_rng(rng, cfg)), cfg, ctx.n_want, record)

    equity = pd.Series(sim["equity"], index=ctx.cal, name="equity")
    trades = _trades_frame(ctx, sim["trades"])
    if record:
        decisions = _decisions_frame(ctx, sim["reasons"])
    else:
        decisions = pd.DataFrame({c: [] for c in DECISION_COLUMNS})
    return {"equity": equity, "trades": trades, "decisions": decisions,
            "metrics": benchmarks.metrics(equity, trades, capital)}
