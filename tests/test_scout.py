"""«Ανιχνευτής»: καθαρισμός, cache CSV, λειτουργία demo/live και (ψεύτικη) λήψη από Yahoo."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from athex_orb import scout, synthetic
from athex_orb.universe import Stock


# ---------------------------------------------------------------- clean_ohlcv

def test_clean_ohlcv_fixes_messy_input():
    raw = pd.DataFrame({
        "Date": ["2024-01-04", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-05", "2024-01-08", "bad"],
        "Open": [10.0, 9.5, 0.0, 9.8, 10.2, np.nan, 1.0],
        "High": [10.5, 9.4, 10.1, 10.0, 10.3, 10.6, 1.0],      # 2024‑01‑02: high < open
        "Low": [9.9, 9.3, 9.6, 9.9, 10.4, 10.1, 1.0],           # 2024‑01‑05: low > open/close
        "Close": [10.2, 9.45, 9.9, 9.95, 10.25, np.nan, 1.0],   # 2024‑01‑08: χωρίς κλείσιμο
        "Volume": [1000, 2000, -5, 1500, np.nan, 100, 1],
        "Dividends": [0, 0, 0, 0, 0, 0, 0],
    })
    df = scout.clean_ohlcv(raw)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert (df.dtypes == np.float64).all()
    assert df.index.name == "date" and df.index.is_monotonic_increasing and df.index.is_unique
    assert list(df.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    # διπλή ημερομηνία ⇒ κρατιέται η τελευταία εγγραφή
    assert df.loc["2024-01-03", "close"] == 9.95 and df.loc["2024-01-03", "volume"] == 1500
    # high/low συνεπή
    assert df.loc["2024-01-02", "high"] == 9.5
    assert df.loc["2024-01-05", "low"] == 10.2
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    # αρνητικός/κενός όγκος ⇒ 0
    assert df.loc["2024-01-05", "volume"] == 0.0


def test_clean_ohlcv_zero_open_and_tz_index():
    idx = pd.DatetimeIndex(["2024-03-01 00:00", "2024-03-04 00:00"]).tz_localize("Europe/Athens")
    raw = pd.DataFrame({"Open": [0.0, 5.0], "High": [0.0, 5.2], "Low": [0.0, 4.9], "Close": [5.1, 5.0],
                        "Volume": [0, 300]}, index=idx)
    df = scout.clean_ohlcv(raw)
    assert df.index.tz is None
    assert list(df.index.strftime("%Y-%m-%d")) == ["2024-03-01", "2024-03-04"]   # τοπική ημερομηνία, όχι UTC
    first = df.iloc[0]
    assert first.open == first.high == first.low == first.close == 5.1


def test_clean_ohlcv_empty_and_invalid():
    assert scout.clean_ohlcv(None).empty
    assert list(scout.clean_ohlcv(pd.DataFrame()).columns) == ["open", "high", "low", "close", "volume"]
    with pytest.raises(ValueError):
        scout.clean_ohlcv(pd.DataFrame({"date": ["2024-01-02"], "open": [1.0]}))


def test_clean_ohlcv_adj_close_fallback():
    raw = pd.DataFrame({"date": ["2024-01-02"], "Adj Close": [3.0], "volume": [10]})
    df = scout.clean_ohlcv(raw)
    assert df.iloc[0].tolist() == [3.0, 3.0, 3.0, 3.0, 10.0]


# ---------------------------------------------------------------- demo / cache

def test_empty_state(cfg):
    assert scout.data_mode(cfg) == "empty"
    meta = scout.data_meta(cfg)
    assert meta["mode"] == "empty" and meta["tickers"] == 0 and meta["last_date"] is None
    assert meta["failed"] == [] and meta["updated_at"] is None
    prices, index_df = scout.load_prices(cfg)
    assert prices == {} and index_df is None


def test_write_demo_data_roundtrip(cfg, small_universe):
    scout.write_demo_data(cfg, small_universe, seed=3)
    assert scout.prices_dir(cfg) == cfg.data_dir / "prices"
    files = sorted(p.name for p in scout.prices_dir(cfg).glob("*.csv"))
    assert f"{cfg.data.index_ticker}.csv" in files and len(files) == len(small_universe) + 1
    header = (scout.prices_dir(cfg) / "OPAP.AT.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "date,open,high,low,close,volume"

    meta = scout.data_meta(cfg)
    assert scout.data_mode(cfg) == "demo" and meta["mode"] == "demo"
    assert meta["tickers"] == len(small_universe) and meta["failed"] == []
    assert meta["last_date"] == synthetic.default_end().strftime("%Y-%m-%d")
    assert meta["updated_at"]

    prices, index_df = scout.load_prices(cfg)
    assert set(prices) == {s.ticker for s in small_universe}
    expected, exp_index = synthetic.generate_universe(small_universe, start=meta["start_date"],
                                                      end=meta["last_date"], seed=3)
    for t, df in prices.items():
        assert df.index.equals(expected[t].index)
        np.testing.assert_allclose(df.to_numpy(), expected[t].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(index_df.to_numpy(), exp_index.to_numpy(), rtol=1e-12)
    years = (index_df.index[-1] - index_df.index[0]).days / 365.25
    assert abs(years - cfg.data.history_years) < 0.05

    subset, _ = scout.load_prices(cfg, ["OPAP.AT", "NOPE.AT"])
    assert list(subset) == ["OPAP.AT"]


def test_load_prices_skips_corrupt_files(cfg, small_universe):
    scout.write_demo_data(cfg, small_universe[:2])
    (scout.prices_dir(cfg) / "BROKEN.AT.csv").write_text("not,a,price\nfile", encoding="utf-8")
    prices, _ = scout.load_prices(cfg)
    assert set(prices) == {s.ticker for s in small_universe[:2]}


def test_add_stock_data_in_demo_matches_universe(cfg, small_universe):
    scout.write_demo_data(cfg, small_universe[:3], seed=9)
    new = Stock("NEWCO.AT", "Νέα Εταιρεία", "Βιομηχανία", custom=True)
    assert scout.add_stock_data(cfg, new)
    prices, _ = scout.load_prices(cfg)
    meta = scout.data_meta(cfg)
    assert meta["tickers"] == 4 and meta["mode"] == "demo"
    together, _ = synthetic.generate_universe(small_universe[:3] + [new], start=meta["start_date"],
                                              end=meta["last_date"], seed=9)
    np.testing.assert_allclose(prices["NEWCO.AT"].to_numpy(), together["NEWCO.AT"].to_numpy(), rtol=1e-12)


# ---------------------------------------------------------------- update_cache (χωρίς δίκτυο)

def _fake_fetch(frames, calls):
    def fake(tickers, years):
        calls.append((list(tickers), years))
        return {t: frames[t] for t in tickers if t in frames}
    return fake


def test_update_cache_live_replaces_demo(cfg, small_universe, monkeypatch):
    scout.write_demo_data(cfg, small_universe)
    real, real_index = synthetic.generate_universe(small_universe, start="2020-01-01", end="2024-06-28", seed=99)
    missing = small_universe[-1].ticker
    frames = {t: df for t, df in real.items() if t != missing}
    frames[cfg.data.index_ticker] = real_index
    calls, progress = [], []
    monkeypatch.setattr(scout, "fetch_yahoo", _fake_fetch(frames, calls))

    res = scout.update_cache(cfg, small_universe, progress=lambda f, m: progress.append((f, m)))

    assert res["failed"] == [missing] and len(res["ok"]) == len(small_universe) - 1 and res["index_ok"]
    requested = [t for chunk, _ in calls for t in chunk]
    assert set(requested) == {s.ticker for s in small_universe} | {cfg.data.index_ticker}
    assert all(years == cfg.data.history_years for _, years in calls)

    meta = scout.data_meta(cfg)
    assert meta["mode"] == "live" and scout.data_mode(cfg) == "live"
    assert meta["failed"] == [missing] and meta["last_date"] == "2024-06-28"
    assert meta["tickers"] == len(small_universe) - 1
    # τα συνθετικά δεδομένα σβήστηκαν — καμία ανάμειξη demo με πραγματικά
    assert not (scout.prices_dir(cfg) / f"{missing}.csv").exists()
    prices, index_df = scout.load_prices(cfg)
    assert missing not in prices
    t0 = small_universe[0].ticker
    np.testing.assert_allclose(prices[t0].to_numpy(), real[t0].to_numpy(), rtol=1e-12)
    assert index_df.index[-1] == pd.Timestamp("2024-06-28")

    fracs = [f for f, _ in progress]
    assert fracs == sorted(fracs) and fracs[0] == 0.0 and fracs[-1] == 1.0
    assert all(isinstance(m, str) and m for _, m in progress)


def test_update_cache_keeps_old_live_data_on_failure(cfg, small_universe, monkeypatch):
    real, real_index = synthetic.generate_universe(small_universe[:3], start="2021-01-01", end="2023-12-29", seed=5)
    frames = dict(real)
    monkeypatch.setattr(scout, "fetch_yahoo", _fake_fetch(frames, []))
    scout.update_cache(cfg, small_universe[:3])
    # δεύτερη ενημέρωση: μία μετοχή αποτυγχάνει — τα παλιά πραγματικά της δεδομένα μένουν
    gone = small_universe[0].ticker
    frames.pop(gone)
    res = scout.update_cache(cfg, small_universe[:3])
    assert res["failed"] == [gone] and not res["index_ok"]
    prices, index_df = scout.load_prices(cfg)
    assert gone in prices and index_df is None
    assert scout.data_meta(cfg)["failed"] == [gone]


def test_update_cache_nothing_downloaded_keeps_demo(cfg, small_universe, monkeypatch):
    scout.write_demo_data(cfg, small_universe[:2])
    before = sorted(p.name for p in scout.prices_dir(cfg).glob("*.csv"))
    monkeypatch.setattr(scout, "fetch_yahoo", lambda tickers, years: {})
    res = scout.update_cache(cfg, small_universe[:2])
    assert res["ok"] == [] and len(res["failed"]) == 2 and res["mode"] == "demo"
    assert scout.data_mode(cfg) == "demo"
    assert sorted(p.name for p in scout.prices_dir(cfg).glob("*.csv")) == before


def test_update_cache_from_user_csv(cfg, small_universe, monkeypatch):
    cfg.data.source = "csv"
    monkeypatch.setattr(scout, "fetch_yahoo", lambda *a, **k: pytest.fail("δεν πρέπει να καλείται το Yahoo"))
    real, _ = synthetic.generate_universe(small_universe[:2], start="2022-01-03", end="2023-12-29", seed=4)
    csv_dir = scout.user_csv_dir(cfg)
    csv_dir.mkdir(parents=True)
    a, b = (s.ticker for s in small_universe[:2])
    real[a].to_csv(csv_dir / f"{a}.csv", index_label="date")
    real[b].rename(columns=str.capitalize).to_csv(csv_dir / f"{b.removesuffix('.AT')}.csv", index_label="Date")
    res = scout.update_cache(cfg, small_universe[:3])
    assert sorted(res["ok"]) == sorted([a, b]) and res["failed"] == [small_universe[2].ticker]
    prices, _ = scout.load_prices(cfg)
    np.testing.assert_allclose(prices[b].to_numpy(), real[b].to_numpy(), rtol=1e-12)
    assert scout.data_meta(cfg)["source"] == "csv"


def test_add_stock_data_live(cfg, small_universe, monkeypatch):
    real, _ = synthetic.generate_universe(small_universe[:2], start="2022-01-03", end="2023-12-29", seed=4)
    monkeypatch.setattr(scout, "fetch_yahoo", _fake_fetch(dict(real), []))
    scout.update_cache(cfg, small_universe[:1])
    assert scout.add_stock_data(cfg, small_universe[1])
    assert scout.data_meta(cfg)["tickers"] == 2
    assert not scout.add_stock_data(cfg, Stock("NONE.AT", "Καμία", "Άλλο"))


# ---------------------------------------------------------------- fetch_yahoo με ψεύτικο yfinance

class _FakeYF:
    def __init__(self, raw, history=None):
        self.raw, self.history_frames, self.download_calls = raw, history or {}, []

    def download(self, tickers, **kwargs):
        self.download_calls.append((tickers, kwargs))
        return self.raw

    def Ticker(self, t):  # noqa: N802 — όπως στο yfinance
        frames = self.history_frames

        class _T:
            def history(self, **kwargs):
                if t not in frames:
                    raise RuntimeError("no data")
                return frames[t]
        return _T()


def _yahoo_like(df):
    out = df.rename(columns=str.capitalize)
    out.index = out.index.rename("Date")
    return out


def test_fetch_yahoo_parses_grouped_download(small_universe, monkeypatch):
    real, _ = synthetic.generate_universe(small_universe[:2], start="2023-01-02", end="2023-03-31", seed=2)
    a, b = list(real)
    raw = pd.concat({a: _yahoo_like(real[a]), b: _yahoo_like(real[b]).iloc[:0]}, axis=1)  # το b ήρθε κενό
    hist = _yahoo_like(real[b])
    hist.index = hist.index.tz_localize("Europe/Athens")
    fake = _FakeYF(raw, history={b: hist})
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    out = scout.fetch_yahoo([a, b, "MISSING.AT"], years=2)
    assert set(out) == {a, b}
    np.testing.assert_allclose(out[a].to_numpy(), real[a].to_numpy())
    assert out[b].index.equals(real[b].index)                 # από το Ticker.history, χωρίς ζώνη ώρας
    assert fake.download_calls[0][1]["auto_adjust"] is False      # προσαρμογή για μερίσματα από εμάς


def test_fetch_yahoo_without_network_returns_empty():
    # το autouse fixture έχει «μπλοκάρει» το yfinance
    assert scout.fetch_yahoo(["OPAP.AT"], years=1) == {}
    assert scout.fetch_yahoo([], years=1) == {}


def test_meta_file_is_utf8_json(cfg, small_universe):
    scout.write_demo_data(cfg, small_universe[:1])
    meta = json.loads((scout.prices_dir(cfg) / "_meta.json").read_text(encoding="utf-8"))
    assert meta["mode"] == "demo" and isinstance(meta["seed"], int)


# ---------------------------------------------------------------- μερίσματα χωρίς ματιά στο μέλλον

def test_dividend_adjust_keeps_traded_value_as_of():
    # Close = τιμή της ημέρας· Adj Close = προσαρμοσμένη για μέρισμα 5% που πληρώνεται αργότερα (στη 3η ημέρα)
    idx = pd.bdate_range("2024-06-03", periods=4, name="Date")
    raw = pd.DataFrame({"Open": [10.0, 10.0, 9.5, 9.6], "High": [10.2, 10.1, 9.7, 9.8], "Low": [9.9, 9.8, 9.4, 9.5],
                        "Close": [10.0, 10.0, 9.5, 9.6], "Adj Close": [9.5, 9.5, 9.5, 9.6],
                        "Volume": [1000.0, 2000.0, 1500.0, 1200.0]}, index=idx)
    df = scout.clean_ohlcv(scout.dividend_adjust(raw))
    # αποδόσεις/ετικέτες με τις προσαρμοσμένες τιμές (χωρίς ψεύτικη πτώση την ημέρα αποκοπής)
    np.testing.assert_allclose(df["close"].to_numpy(), [9.5, 9.5, 9.5, 9.6])
    np.testing.assert_allclose(df["open"].iloc[0], 9.5)
    # τζίρος (κλείσιμο × όγκος) = ο πραγματικός της ημέρας, ανεξάρτητος από το μελλοντικό μέρισμα
    np.testing.assert_allclose((df["close"] * df["volume"]).to_numpy(), [10000.0, 20000.0, 14250.0, 11520.0])
    # χωρίς Adj Close (π.χ. δείκτης): ως έχει
    no_adj = raw.drop(columns=["Adj Close"])
    np.testing.assert_allclose(scout.clean_ohlcv(scout.dividend_adjust(no_adj))["close"].to_numpy(),
                               no_adj["Close"].to_numpy())


def test_drop_partial_bar_during_athens_session():
    idx = pd.bdate_range("2026-10-12", periods=2)          # Δευτέρα 12/10, Τρίτη 13/10
    df = pd.DataFrame({c: [1.0, 1.1] for c in scout.OHLCV}, index=idx)
    during = pd.Timestamp("2026-10-13 12:30", tz="Europe/Athens")
    after = pd.Timestamp("2026-10-13 18:00", tz="Europe/Athens")
    assert scout.drop_partial_bar(df, during).index[-1] == pd.Timestamp("2026-10-12")
    assert scout.drop_partial_bar(df, after).index[-1] == pd.Timestamp("2026-10-13")
    # την επόμενη μέρα η χθεσινή μπάρα είναι οριστική
    assert len(scout.drop_partial_bar(df, pd.Timestamp("2026-10-14 11:00", tz="Europe/Athens"))) == 2


# ---------------------------------------------------------------- CSV του χρήστη (ελληνικές μορφές)

def _true_frame():
    idx = pd.bdate_range("2024-01-02", periods=30)
    c = 10 + np.arange(30) * 0.1
    return pd.DataFrame({"open": c, "high": c + 0.2, "low": c - 0.2, "close": c, "volume": 1000.0 + np.arange(30)},
                        index=idx)


def test_user_csv_day_first_dates(cfg):
    real = _true_frame()
    d = scout.user_csv_dir(cfg)
    d.mkdir(parents=True)
    out = real.copy()
    out.index = out.index.strftime("%d/%m/%Y")
    out.to_csv(d / "TEST.csv", index_label="date")
    df, note = scout._read_user_csv(d / "TEST.csv")
    assert note is None and len(df) == 30
    assert df.index.equals(real.index.as_unit("us"))
    np.testing.assert_allclose(df["close"].to_numpy(), real["close"].to_numpy())


def test_user_csv_greek_excel_export(cfg):
    real = _true_frame()
    d = scout.user_csv_dir(cfg)
    d.mkdir(parents=True)
    lines = ["Ημερομηνία;Άνοιγμα;Υψηλό;Χαμηλό;Κλείσιμο;Όγκος"]
    for t, r in real.iterrows():
        def g(x, nd=2):
            return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",").replace(" ", ".")
        lines.append(f"{t:%d.%m.%Y};{g(r.open)};{g(r.high)};{g(r.low)};{g(r.close)};{g(r.volume * 10, 0)}")
    (d / "TEST.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    df, note = scout._read_user_csv(d / "TEST.csv")
    assert note is None and len(df) == 30 and df.index[0] == pd.Timestamp("2024-01-02")
    np.testing.assert_allclose(df["close"].to_numpy(), real["close"].to_numpy())
    np.testing.assert_allclose(df["volume"].to_numpy(), real["volume"].to_numpy() * 10)


def test_user_csv_reports_bad_rows_and_missing_close(cfg):
    d = scout.user_csv_dir(cfg)
    d.mkdir(parents=True)
    (d / "BAD.csv").write_text("date,open,high,low,close,volume\n2024-01-02,1,1,1,1,10\nσκουπίδια,1,1,1,1,1\n"
                               "2024-01-04,1,1,1,,1\n", encoding="utf-8")
    df, note = scout._read_user_csv(d / "BAD.csv")
    assert len(df) == 1 and "αγνοήθηκαν 2 από 3" in note
    (d / "NOCLOSE.csv").write_text("date;price\n2024-01-02;1\n", encoding="utf-8")
    df, note = scout._read_user_csv(d / "NOCLOSE.csv")
    assert df is None and "close" in note


def test_update_cache_csv_problem_reaches_the_user(cfg, small_universe):
    from athex_orb import pipeline

    cfg.data.source = "csv"
    d = scout.user_csv_dir(cfg)
    d.mkdir(parents=True)
    (d / "ETE.csv").write_text("date;price\n2024-01-02;1\n", encoding="utf-8")
    with pytest.raises(pipeline.PipelineError, match="close"):
        pipeline.run_refresh(cfg, stocks=small_universe[:1])


# ---------------------------------------------------------------- «κομμένη» λήψη, meta, σημάδι επίδειξης

def test_truncated_download_never_wipes_history(cfg, small_universe, monkeypatch):
    real, _ = synthetic.generate_universe(small_universe[:2], start=pd.Timestamp.today().normalize()
                                          - pd.DateOffset(years=cfg.data.history_years), end=synthetic.default_end(),
                                          seed=3)
    frames = dict(real)
    monkeypatch.setattr(scout, "fetch_yahoo", _fake_fetch(frames, []))
    scout.update_cache(cfg, small_universe[:2])
    a, b = list(real)
    n_full = len(real[a])
    # δεύτερη λήψη: για το a μόνο οι 3 τελευταίες ημέρες, με νέα βάση προσαρμογής (μέρισμα: × 0,98)
    tail = real[a].tail(3).copy()
    tail[["open", "high", "low", "close"]] *= 0.98
    frames[a] = tail
    res = scout.update_cache(cfg, small_universe[:2])
    assert a in res["spliced"] and a in res["problems"]
    prices, _ = scout.load_prices(cfg)
    assert len(prices[a]) == n_full and len(prices[b]) == n_full
    # το παλιό ιστορικό αναγμένο στη νέα βάση: καμία ψεύτικη «πτώση» 2% στη ραφή
    np.testing.assert_allclose(prices[a]["close"].iloc[:-3].to_numpy(), real[a]["close"].iloc[:-3].to_numpy() * 0.98,
                               rtol=1e-9)
    np.testing.assert_allclose((prices[a]["close"] * prices[a]["volume"]).to_numpy(),
                               (real[a]["close"] * real[a]["volume"]).to_numpy() * np.r_[np.ones(n_full - 3), np.full(3, 0.98)],
                               rtol=1e-6)


def test_meta_is_atomic_and_demo_fails_closed(cfg, small_universe, monkeypatch):
    scout.write_demo_data(cfg, small_universe[:3])
    assert not list(scout.prices_dir(cfg).glob(".*.tmp"))
    # κατεστραμμένο _meta.json: τα συνθετικά δεδομένα δεν περνιούνται για πραγματικά
    (scout.prices_dir(cfg) / "_meta.json").write_text("{\"mode\": \"de", encoding="utf-8")
    assert scout.data_mode(cfg) == "demo"
    # μια μερική πραγματική λήψη σβήνει ΟΛΑ τα συνθετικά (καμία ανάμειξη)
    real, _ = synthetic.generate_universe(small_universe[:1], start="2022-01-03", end="2023-12-29", seed=4)
    monkeypatch.setattr(scout, "fetch_yahoo", _fake_fetch(dict(real), []))
    scout.update_cache(cfg, small_universe[:3])
    assert scout.data_mode(cfg) == "live"
    assert sorted(p.stem for p in scout._stock_files(cfg)) == [small_universe[0].ticker]


def test_add_stock_data_live_is_trimmed_to_cache_date(cfg, small_universe, monkeypatch):
    real, _ = synthetic.generate_universe(small_universe[:2], start="2022-01-03", end="2023-12-29", seed=4)
    a, b = list(real)
    frames = {a: real[a][real[a].index <= "2023-12-22"], b: real[b]}       # νέα μετοχή: 5 μέρες «νεότερη»
    monkeypatch.setattr(scout, "fetch_yahoo", _fake_fetch(frames, []))
    scout.update_cache(cfg, [s for s in small_universe if s.ticker == a])
    assert scout.add_stock_data(cfg, next(s for s in small_universe if s.ticker == b))
    prices, _ = scout.load_prices(cfg)
    assert prices[b].index[-1] == prices[a].index[-1] == pd.Timestamp("2023-12-22")
    assert not scout.add_stock_data(cfg, Stock(cfg.data.index_ticker, "Γενικός Δείκτης", "Άλλο"))
