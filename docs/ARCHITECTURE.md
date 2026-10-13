# ATHEX Orb — Architecture & contracts

This document is the single source of truth for module interfaces. Code comments and
UI text are in Greek; identifiers are English. Python ≥ 3.11, pandas ≥ 2.1 (tested on 3.0),
numpy, scikit-learn, FastAPI, yfinance. Frontend: plain HTML/CSS/ES modules, **no build step,
no CDN JS** (the app must work offline; Google Fonts optional with system fallback).

## Philosophy (non‑negotiable)

* This is a **research / backtest / paper‑trading** tool. It never places orders.
* **No look‑ahead.** Features at date `t` use data ≤ `t`. A training sample for date `t` is usable at
  prediction date `T` only if its outcome resolved strictly before `T` (`resolved_date < T`).
* **Costs everywhere.** Every EV and every backtest trade pays commission, fees, tax, spread, impact.
* **Honest verdict.** Strategy is compared with buy & hold and with random entries under identical
  rules. If it does not beat random, the UI says so plainly.
* Demo mode (synthetic data) is clearly labelled in the UI.

## Layout

```
athex_orb/
  config.py      (done) Config dataclasses, load_config(), save_user_settings()
  universe.py    (done) Stock, DEFAULT_UNIVERSE, load_universe(), search(), normalize_ticker(), fold()
  synthetic.py   synthetic OHLCV generator (demo + tests)
  scout.py       data download / cache / load
  market.py      market regime & sector context
  features.py    per‑stock features (panel)
  labels.py      triple‑barrier labels + shared exit simulator
  orb_model.py   3‑state probability model + walk‑forward
  quant.py       execution costs + expected value
  guard.py       risk checks
  executor.py    portfolio backtest
  benchmarks.py  buy&hold, random‑entry Monte Carlo, metrics, calibration
  journal.py     SQLite paper‑trading journal
  pipeline.py    orchestration; writes state JSON files read by the server
  explain.py     per‑stock "six agents" report for the UI
  server.py      FastAPI app, background jobs, serves web/
  cli.py         python -m athex_orb <command>
  __main__.py
web/  index.html, css/app.css, js/*.js
tests/
```

## Data representations

* **prices**: `dict[str, pd.DataFrame]` — key is ticker (e.g. `"OPAP.AT"`). Each frame: `DatetimeIndex`
  (tz‑naive, normalized, ascending, unique, name `"date"`), float columns
  `open, high, low, close, volume` (split/dividend adjusted). Rows with NaN close are dropped.
* **index_df**: same shape, for the market index (`cfg.data.index_ticker`), or an equal‑weight index
  built by `market.equal_weight_index(prices)` when missing.
* **panel**: long `pd.DataFrame` with `MultiIndex` (`date`, `ticker`), sorted.

## synthetic.py

```python
def generate_universe(stocks: list[Stock], start="2017-01-02", end=None, seed=7,
                      planted_signal: float = 0.0) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]
```
Returns `(prices, index_df)` on business days. Correlated GBM with a market factor + sector factor +
idiosyncratic noise, realistic OHLC (high ≥ max(open,close), low ≤ min(open,close)), volume
log‑normal so that traded value spans ~€50k–€30m/day across stocks, occasional gaps.
`planted_signal > 0` injects a *learnable* edge (e.g. next‑days drift proportional to
a lagged, observable feature such as 5‑day return sign or volume spike) — used only by tests to
prove the model can find a real signal. `planted_signal = 0` ⇒ pure random walk (no edge).
`end` default = last business day before today. Deterministic for a given seed.

## scout.py  (agent: «Ανιχνευτής»)

```python
def prices_dir(cfg) -> Path                      # cfg.data_dir / "prices"
def fetch_yahoo(tickers: list[str], years: int) -> dict[str, pd.DataFrame]   # yfinance auto_adjust=False + dividend_adjust; skips failures
def update_cache(cfg, stocks: list[Stock], progress=None) -> dict            # downloads stocks + index, writes CSVs, returns {"ok": [...], "failed": [...]}
def write_demo_data(cfg, stocks, seed=7) -> None # synthetic → CSV cache + marks mode demo
def load_prices(cfg, tickers: list[str] | None = None) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]
def data_mode(cfg) -> str                        # "demo" | "live" | "empty"   (stored in data/prices/_meta.json)
def data_meta(cfg) -> dict                       # {"mode", "updated_at" ISO, "last_date", "tickers": n, "failed": [...]}
def clean_ohlcv(df) -> pd.DataFrame              # enforce schema, drop dups/NaN, fix high/low consistency
```
CSV format per ticker: `data/prices/<TICKER>.csv` with header `date,open,high,low,close,volume`.
`source="csv"` reads user files from `data/csv/<TICKER>.csv` (same header) instead of Yahoo.
`progress` is an optional callable `progress(fraction: float, message_el: str)`.

Additions (fix round): `dividend_adjust(raw)` — OHLC × (Adj Close / Close), volume ÷ the same factor, so that
`close × volume` is the as-of traded value (no look-ahead from later dividends in adv20/sizing/costs);
`drop_partial_bar(df, now)` — drops today's in-progress Yahoo bar before 17:35 Athens; user CSVs accept `;`/tab,
decimal comma, Greek headers and day-first dates (strict, whole-column parsing); a truncated download never wipes
cached history (spliced with re-scaling); `_meta.json` is written atomically and a `prices/_DEMO` marker keeps
`data_mode` fail-closed; `add_stock_data` trims a live series to the cache's `last_date`. `update_cache` also
returns `problems` ({ticker: Greek message}), `spliced`, `index_last_date`.

## market.py  (agent: «Χαρτογράφος» — market context)

```python
def equal_weight_index(prices) -> pd.DataFrame
def market_features(index_df) -> pd.DataFrame     # indexed by date: mkt_ret_5, mkt_ret_20, mkt_vol_20,
                                                  # mkt_trend (close/SMA100 − 1), mkt_drawdown_10 (close / max close last 10 − 1),
                                                  # regime ∈ {"bull","bear","neutral"}, regime_el (Greek label)
def sector_strength(prices, stocks) -> pd.DataFrame   # MultiIndex (date,ticker): sector_rel_20 = stock 20d ret − sector mean 20d ret
```

## features.py  (agent: «Κριτής» — per‑stock judgement)

```python
FEATURE_COLUMNS: list[str]   # exact model inputs, in order
FEATURE_LABELS_EL: dict[str, str]   # Greek human labels for UI explanations
def build_panel(prices, index_df, stocks, cfg) -> pd.DataFrame
```
`build_panel` returns the panel (MultiIndex date,ticker) with **all FEATURE_COLUMNS** plus
non‑model columns: `close, atr, atr_pct (=atr/close), adv20_eur (20‑day mean of close×volume),
history_days (rows so far), sector, mkt_drawdown_10, regime`.
Required features (all strictly causal, computed with rolling windows ending at t):
`ret_1, ret_5, ret_20, ret_60, dist_sma20, dist_sma50, dist_sma200, rsi_14, atr_pct,
vol_20 (std of daily ret), vol_ratio (volume / 20d mean volume), value_z (z‑score of log traded value vs 60d),
gap_1 (open_t / close_{t−1} − 1), range_pos_252 ((close−min252)/(max252−min252)), rel_mkt_20 (ret_20 − mkt_ret_20),
sector_rel_20, mkt_ret_20, mkt_vol_20, mkt_trend`.
NaN handling: rows where any FEATURE_COLUMN is NaN are kept in the panel (UI needs them) but are
excluded from training/prediction by the model layer. ATR = Wilder ATR(cfg.barriers.atr_days).
Market features are forward-filled at most `MKT_FFILL_LIMIT = 5` sessions (a stale index ⇒ NaN, not frozen values).

## labels.py  (the three Orb states)

```python
LABEL_TP, LABEL_SL, LABEL_TO = 0, 1, 2
LABEL_NAMES_EL = {0: "Στόχος πρώτα", 1: "Stop πρώτα", 2: "Λήξη χρόνου"}

def simulate_exit(o, h, l, c, entry_idx: int, entry_price: float, tp_price: float, sl_price: float,
                  horizon: int) -> tuple[int, int, float]   # (label, exit_idx, exit_price) or (-1, -1, nan) if unresolved
def triple_barrier(prices, panel, cfg) -> pd.DataFrame
```
Arrays `o,h,l,c` are numpy arrays of one ticker. Semantics (also used by executor — must be identical):
* Decision at close of day `t`; entry at **open of t+1** = `entry_idx`, `entry_price = o[entry_idx]`.
* Barriers: `tp_price = entry_price × (1 + k_tp × atr_pct_t)`, `sl_price = entry_price × (1 − k_sl × atr_pct_t)`.
* Bars `j = entry_idx … entry_idx + horizon − 1` (entry day counts as bar 1).
  For `j > entry_idx`: if `o[j] ≥ tp_price` → TP exit at `o[j]`; if `o[j] ≤ sl_price` → SL exit at `o[j]` (gap fill).
  Then intrabar: if `l[j] ≤ sl_price` and `h[j] ≥ tp_price` → **SL** (conservative) at `sl_price`;
  elif `l[j] ≤ sl_price` → SL at `sl_price`; elif `h[j] ≥ tp_price` → TP at `tp_price`.
* No barrier by bar `entry_idx + horizon − 1` → TIMEOUT exit at `c[entry_idx + horizon − 1]`.
* If data ends before resolution → unresolved `(-1, -1, nan)`.

`triple_barrier` output: DataFrame aligned to panel index with columns
`label (float, NaN if unresolved), entry_date, entry_price, tp_ret, sl_ret, tp_price, sl_price,
exit_date, exit_price, gross_ret (exit/entry − 1), resolved_date (= exit_date; NaT if unresolved), bars_held`.
`tp_ret = k_tp × atr_pct`, `sl_ret = k_sl × atr_pct`.

## orb_model.py  (agent: «Η Σφαίρα»)

```python
class OrbModel:
    def __init__(self, kind="logreg", random_state=0)
    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrbModel"      # standardizes internally (Pipeline)
    def predict_proba(self, X) -> np.ndarray                      # shape (n,3): [p_tp, p_sl, p_to]; always 3 cols
    def contributions(self, x_row: np.ndarray) -> dict[str, float]  # logreg: per‑feature contribution to logit(p_tp) − logit(p_sl); hgb: {} 
    class_timeout_mean: float   # mean gross_ret of TIMEOUT samples in training set (0.0 if none)
    trained_until: pd.Timestamp | None
    n_train: int

def walk_forward(panel: pd.DataFrame, labels: pd.DataFrame, cfg, progress=None) -> tuple[pd.DataFrame, OrbModel | None]
```
`walk_forward` iterates over unique dates. First prediction date is the first date with at least
`min_train_days` distinct dates of history before it and at least 200 usable samples. Retrains every
`retrain_every_days` dates; the training set at retrain date `T` = rows with valid features, non‑NaN
label and `resolved_date < T` (and `date ≥ T − train_window_days` dates if window > 0). Predictions for
dates in `[T, next retrain)` use that model. Returns
`preds`: DataFrame (index = panel index subset) with `p_tp, p_sl, p_to, timeout_mean, model_date`;
and the **final model** trained on everything resolved up to the last date (for live signals).
Missing classes in a training set are handled (probabilities for absent classes = small prior).

## quant.py  (agent: «Ποσοτικός»)

```python
def side_cost_pct(order_eur, adv_eur, price, cfg, side: str) -> dict   # {"commission","fees","tax","spread","impact","total"} as fractions of order value
def round_trip_cost_pct(order_eur, adv_eur, price, cfg) -> dict        # buy + sell, same keys
def expected_value(p_tp, p_sl, p_to, tp_ret, sl_ret, timeout_mean, cost_rt) -> dict
    # {"ev_gross": p_tp*tp_ret − p_sl*sl_ret + p_to*timeout_mean, "ev_net": ev_gross − cost_rt,
    #  "breakeven_p_tp": p_tp that makes ev_net = 0 holding p_to fixed}
```
Spread (half‑spread per side, fraction) = `spread_bps_base/1e4 × clip(sqrt(5e6 / adv_eur), 1, 8)`.
Impact = `impact_coef × sqrt(order_eur / adv_eur) × daily_vol` where daily_vol default 0.02 if unknown
(accept optional `vol` kwarg). Commission = `max(commission_pct, commission_min_eur / order_eur)`.
Tax only on sell. Vectorized helpers welcome (`*_vec` taking numpy arrays) for the backtest.

## guard.py  (agent: «Φύλακας»)

```python
@dataclass class Check: id: str; passed: bool; label_el: str; detail_el: str; blocking: bool = True
def stock_checks(row: pd.Series, cfg) -> list[Check]     # liquidity (adv20_eur ≥ min_adv_eur), price ≥ min_price_eur,
                                                         # atr_pct ≤ max_atr_pct, history_days ≥ min_history_days,
                                                         # market crash (mkt_drawdown_10 > −crash_index_drop), features available
def decision_checks(ev: dict, p_tp: float, cfg) -> list[Check]    # ev_net ≥ min_ev_net, p_tp ≥ min_p_tp
def portfolio_checks(ticker, sector, open_positions: list[dict], cfg) -> list[Check]  # max_positions, max_per_sector, not already held
def position_size_eur(cash_eur, adv_eur, cfg) -> float   # min(position_eur_max, cash, max_adv_fraction × adv)
def passed(checks) -> bool                                # all blocking checks passed
```

## executor.py  (agent: «Εκτελεστής»)

```python
def run_backtest(prices, panel, preds, labels, cfg, selector="model", rng=None) -> dict
```
Day loop over prediction dates. At close of `t`: candidates = rows with predictions at `t` passing
`stock_checks` and `decision_checks`; ranked by `ev_net` desc (`selector="model"`) or randomly
(`selector="random"`, same candidate set *before* decision_checks but taking the same count the
model would on that day — see benchmarks). Respect `portfolio_checks`, size with `position_size_eur`
(skip if < €300). Enter at `o[t+1]`, exit using `labels.simulate_exit` (same barriers), pay
`side_cost_pct` on each side (cash accounting in EUR, shares may be fractional = order/price).
Mark‑to‑market equity daily using close. Returns
`{"equity": pd.Series (date → EUR), "trades": pd.DataFrame [ticker, sector, signal_date, entry_date, entry_price,
exit_date, exit_price, label, size_eur, gross_ret, cost_eur, net_pnl_eur, net_ret, p_tp, p_sl, p_to, ev_net],
"decisions": pd.DataFrame [date, ticker, taken(bool), reason_el], "metrics": dict}`.
Positions still open at the end are closed at the last close (label −1, marked).

## benchmarks.py

```python
def metrics(equity: pd.Series, trades: pd.DataFrame | None, capital: float) -> dict
    # total_return, cagr, max_drawdown, sharpe, volatility, n_trades, win_rate, avg_win, avg_loss,
    # profit_factor, avg_bars_held, total_costs_eur, exposure (fraction of days with ≥1 position)
def buy_and_hold(index_df, start, end, capital) -> pd.Series
def random_benchmark(prices, panel, preds, labels, cfg, runs) -> dict
    # {"equity_median": Series, "equity_p05": Series, "equity_p95": Series, "final_returns": list[float],
    #  "run_metrics": {metric: [...]}, "metrics_median": {metric: median across runs}, ...}
def skill(preds, labels) -> dict   # out-of-sample log-loss/Brier vs constant base rates, block bootstrap by date
def calibration(preds, labels, bins=10) -> list[dict]      # [{"bin_lo","bin_hi","predicted","realized","n"}] for p_tp vs label==TP (resolved only)
def verdict(strategy_metrics, random_final_returns, bh_metrics, *, bh_universe_metrics=None, skill_result=None) -> dict
    # {"edge": bool, "percentile": float (share of random runs strategy beats), "p_value", "text_el": str,
    #  "level": "good"|"weak"|"none", ...}
```
`edge` is true only if p = (1 + #{random ≥ strategy}) / (1 + runs) ≤ 0.05 with ≥ 100 random runs **and** the
strategy beats buy&hold of the index **and** of an equal-weight portfolio of the same (adjusted) universe **and**
the out-of-sample `skill` test passes **and** it is profitable with ≥ 30 trades. "weak" also requires skill.

## journal.py  (paper trading memory)

SQLite at `data/journal.sqlite`, table `decisions(id, signal_date, ticker, decision ('take'|'skip'),
reason_el, p_tp, p_sl, p_to, ev_net, entry_ref (last close), tp_ret, sl_ret, size_eur,
status ('pending'|'open'|'tp'|'sl'|'timeout'|'void'), entry_date, entry_price, exit_date, exit_price,
gross_ret, net_ret, created_at, UNIQUE(signal_date, ticker))`.
```python
def record(cfg, signal_date, rows: list[dict]) -> int      # upsert today's decisions (both take & skip);
                                                           # frozen from 10:00 Athens of the next business day;
                                                           # extra column `blocked` (skip by the Guard)
def resolve(cfg, prices) -> int                            # fill outcomes using labels.simulate_exit once data is available
def list_entries(cfg, limit=500) -> list[dict]
def summary(cfg) -> dict                                   # counts, hit rates, avg net ret for 'take' vs 'skip', calibration on resolved
```

## pipeline.py

```python
def state_dir(cfg) -> Path                 # data/state
def run_refresh(cfg, progress=None, demo=False) -> dict
    # 1 fetch/update data (or demo data)  2 panel + labels  3 walk_forward  4 latest signals for last date
    # 5 journal.record + journal.resolve  6 write state files  → returns summary
def run_backtest_job(cfg, progress=None) -> dict   # executor + benchmarks → state/backtest.json
def load_state(cfg, name) -> dict | None   # reads state/<name>.json
```
The official index is used only if it lags the stocks by ≤ 2 sessions (`usable_index`); otherwise an equal-weight
index is used, `index_ok = False` and `meta.index_note_el` explains why. Only `run_refresh` writes journal
decisions; the fast `recompute_signals` (settings change, add/remove stock) never does. A demo refresh clears the
demo-mode journal.

State files (JSON, UTF‑8, `ensure_ascii=False`, NaN → null):
* `state/universe.json` — the `/api/universe` payload minus `favorite` flags.
* `state/stocks/<TICKER>.json` — the `/api/stock/{ticker}` payload minus `favorite`.
* `state/signals.json` — the `/api/signals` payload.
* `state/backtest.json` — the `/api/backtest` payload.
* `state/meta.json` — model/training metadata for `/api/status`.
Panel/labels/preds may be cached as pickles in `state/` for speed; never committed.

## explain.py

```python
def agent_report(ticker, panel, labels, preds_row_or_None, model, stock, cfg, open_positions=()) -> dict
```
Returns the `agents` object of `/api/stock/{ticker}` (see API). Greek text, concise, numbers formatted by UI
(send raw numbers + `fmt` hints).

## HTTP API (FastAPI, all JSON, UTF‑8). Served at `http://127.0.0.1:8000`, static UI at `/`.

Number conventions: fractions for percentages (0.0123 = 1.23%), EUR as floats, dates `YYYY-MM-DD`.

* `GET /api/status` →
  `{"mode": "demo"|"live"|"empty", "data_last_date", "data_updated_at", "model_trained_until", "n_tickers",
    "n_with_data", "horizon_days", "k_tp", "k_sl", "jobs": [job...], "disclaimer_el"}`
* `GET /api/universe` →
  `{"as_of": date, "items": [Item]}` with
  `Item = {"ticker","name","sector","custom": bool,"has_data": bool,"last_close","change_1d","change_5d",
   "spark": [last 60 closes], "adv20_eur", "orb": {"p_tp","p_sl","p_to","ev_net","dominant":"tp"|"sl"|"to"} | null,
   "decision": "take"|"skip"|"blocked"|"none", "favorite": bool}`
* `GET /api/search?q=` → `{"items": [Item], "suggest": {"ticker": "XYZ.AT", "valid": bool} | null}`
  (`suggest` present when q looks like a ticker not in the universe → UI offers "Προσθήκη").
* `POST /api/universe` body `{"ticker": "...", "name"?: "...", "sector"?: "..."}` → `{"ok", "item"?, "error_el"?}`
  (409 while a refresh/demo job runs; the index ticker is refused).
  Normalizes ticker, adds to custom universe, fetches its data (or synthesizes in demo mode), recomputes that
  stock's state with the current model (fast path; full retrain happens on next refresh).
* `DELETE /api/universe/{ticker}` → custom only.
* `GET /api/favorites` → `{"tickers": [...]}` · `PUT /api/favorites/{ticker}` · `DELETE /api/favorites/{ticker}`
  (persisted in `data/favorites.json`, order preserved).
* `GET /api/stock/{ticker}?days=250` →
  `{"ticker","name","sector","custom","favorite","has_data","as_of",
    "candles": [{"d","o","h","l","c","v"}], "orb_history": [{"d","p_tp","p_sl","p_to"}],
    "orb": {"p_tp","p_sl","p_to","ev_net","dominant","breakeven_p_tp"} | null,
    "plan": {"entry_ref","tp_price","sl_price","horizon_days","size_eur","shares"} | null,
    "decision": "take"|"skip"|"blocked"|"none", "decision_reason_el",
    "agents": [Agent x6], "outcomes": [{"signal_date","label","gross_ret"}] (last 20 resolved, for this stock)}`
  `Agent = {"id": "scout"|"market"|"judge"|"quant"|"guard"|"executor", "name_el", "role_el",
            "status": "ok"|"warn"|"block"|"info", "headline_el",
            "metrics": [{"label_el","value": number|string|null,"fmt": "pct"|"eur"|"num"|"int"|"text"|"date","hint_el"?}],
            "notes_el": [str]}`
  Guard agent additionally has `"checks": [{"id","passed","label_el","detail_el","blocking"}]`.
  Judge agent additionally has `"drivers": [{"feature","label_el","contribution"}]` (top ±, may be empty).
* `GET /api/signals` → `{"as_of","items":[{"ticker","name","sector","decision","reason_el","p_tp","p_sl","p_to","ev_net",
   "entry_ref","tp_price","sl_price","size_eur"}]}` (take first, then by ev_net desc).
* `GET /api/backtest` → `{"available": bool, "generated_at", "period": {"start","end"},
   "params": {...}, "metrics": {"strategy": M, "buy_hold": M, "buy_hold_universe": M, "random_median": M},
   "skill": {...}, "equity": [{"d","strategy","buy_hold","buy_hold_universe","random_median","random_p05","random_p95"}],
   "drawdown": [{"d","strategy"}], "trades": [... last 300 ...], "calibration": [...],
   "by_label": {"tp": n, "sl": n, "to": n}, "verdict": {...}}`
* `GET /api/journal` → `{"items": [...], "summary": {...}}`
* `GET /api/settings` → editable subset `{"portfolio": {...}, "barriers": {...}, "guard": {...}, "costs": {...}, "model": {"kind"}}`;
  `PUT /api/settings` same shape (partial ok) → saved to `data/settings.json`; response says if retrain needed.
* `POST /api/jobs` body `{"type": "refresh"|"backtest"|"demo"}` → `{"job_id"}` (409 if same type running).
  `GET /api/jobs/{id}` → `{"id","type","status":"queued"|"running"|"done"|"error","progress": 0..1,"message_el","started_at","finished_at","error"}`.
  `GET /api/jobs` → recent jobs. Jobs run in a background thread; one at a time.
* Errors: HTTP 4xx/5xx with `{"detail": "<Greek message>"}`.

## CLI

`python -m athex_orb demo | refresh | backtest | serve [--port] | signals | journal`.
`demo` = generate synthetic data + refresh + backtest. `serve` starts uvicorn and prints the URL.
