"""«Ανιχνευτής»: λήψη, καθαρισμός, αποθήκευση και φόρτωση τιμών.

Αποθήκευση: ένα CSV ανά σύμβολο στο data/prices/<TICKER>.csv (κεφαλίδα date,open,high,low,close,volume)
και μεταδεδομένα στο data/prices/_meta.json (λειτουργία demo/live, ώρα ενημέρωσης κ.λπ.).
Το yfinance φορτώνεται μόνο όταν χρειάζεται (οι δοκιμές δεν το καλούν ποτέ).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import synthetic
from .universe import Stock, fold

OHLCV = ["open", "high", "low", "close", "volume"]
META_FILE = "_meta.json"
DEMO_MARKER = "_DEMO"
YAHOO_CHUNK = 8                 # σύμβολα ανά αίτημα λήψης (για ενδείξεις προόδου)
ATHENS_TZ = "Europe/Athens"
SESSION_CLOSE_LOCAL = (17, 35)  # μετά τις 17:30 (δημοπρασία κλεισίματος) η ημερήσια τιμή είναι οριστική
TRUNCATED_DAYS = 10             # νέα λήψη που ξεκινά > 10 ημέρες μετά την παλιά ⇒ «κομμένη» λήψη


# ---------------------------------------------------------------- διαδρομές

def prices_dir(cfg) -> Path:
    return Path(cfg.data_dir) / "prices"


def user_csv_dir(cfg) -> Path:
    """Φάκελος με αρχεία CSV του χρήστη όταν data.source = "csv"."""
    return Path(cfg.data_dir) / "csv"


def _safe_name(ticker: str) -> str:
    return re.sub(r"[^A-Za-z0-9.\-_^]", "_", ticker)


def _csv_path(cfg, ticker: str) -> Path:
    return prices_dir(cfg) / f"{_safe_name(ticker)}.csv"


def _meta_path(cfg) -> Path:
    return prices_dir(cfg) / META_FILE


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _report(progress, fraction: float, message_el: str) -> None:
    if progress is not None:
        try:
            progress(float(min(max(fraction, 0.0), 1.0)), message_el)
        except Exception:  # η ένδειξη προόδου δεν πρέπει ποτέ να σταματά τη λήψη
            pass


# ---------------------------------------------------------------- καθαρισμός

def _empty_frame() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series([], dtype="float64") for c in OHLCV},
                      index=pd.DatetimeIndex([], name="date").as_unit("us"))
    return df


DAYFIRST_RE = re.compile(r"^\d{1,2}([/.\-])\d{1,2}\1\d{4}(\s|$)")   # 02/01/2024 (ελληνική σύμβαση)


def _parse_text_dates(values) -> pd.DatetimeIndex:
    """Αυστηρή ανάγνωση ημερομηνιών-κειμένου: ISO (2024-01-02) ή ημέρα/μήνας/έτος (02/01/2024).

    Ποτέ «μαντεψιά» ανά γραμμή: όλη η στήλη διαβάζεται με μία μορφή, ώστε να μη μπερδευτούν ημέρα και
    μήνας. Οι ημερομηνίες ΗΗ/ΜΜ/ΕΕΕΕ ερμηνεύονται ως ημέρα‑πρώτα (ελληνική σύμβαση)· μόνο αν το δεύτερο
    μέρος ξεπερνά το 12 σε κάποια γραμμή ενώ το πρώτο ποτέ, το αρχείο θεωρείται αμερικανικό (ΜΜ/ΗΗ/ΕΕΕΕ).
    """
    s = pd.Series(list(values), dtype="object").astype("str").str.strip()
    sample = s[(s.str.len() > 0) & (s.str.lower() != "nan")]
    if len(sample) and sample.map(lambda x: bool(DAYFIRST_RE.match(x))).all():
        sep = DAYFIRST_RE.match(sample.iloc[0]).group(1)
        head = s.str.split(r"\s", n=1, regex=True).str[0]
        parts = head.str.split(sep, expand=True)
        first = pd.to_numeric(parts[0], errors="coerce")
        second = pd.to_numeric(parts[1], errors="coerce")
        monthfirst = bool((second > 12).any()) and not bool((first > 12).any())
        fmt = f"%m{sep}%d{sep}%Y" if monthfirst else f"%d{sep}%m{sep}%Y"
        return pd.DatetimeIndex(pd.to_datetime(head, format=fmt, errors="coerce"))
    try:
        return pd.DatetimeIndex(pd.to_datetime(s, format="ISO8601", errors="coerce"))
    except (ValueError, TypeError):
        # μικτές ζώνες ώρας → ώρα Αθήνας
        return pd.DatetimeIndex(pd.to_datetime(s, format="ISO8601", errors="coerce", utc=True))


def _to_dates(values) -> pd.DatetimeIndex:
    arr = values if isinstance(values, pd.Index) else pd.Index(values)
    if isinstance(arr, pd.DatetimeIndex):
        idx = arr
    elif arr.dtype.kind == "M" or isinstance(arr.dtype, pd.DatetimeTZDtype):
        idx = pd.DatetimeIndex(arr)
    elif arr.dtype == object and len(arr) and all(isinstance(v, (pd.Timestamp, datetime)) for v in arr[:50]):
        try:
            idx = pd.DatetimeIndex(pd.to_datetime(arr, errors="coerce"))
        except (ValueError, TypeError):   # μικτές ζώνες ώρας
            idx = pd.DatetimeIndex(pd.to_datetime(arr, errors="coerce", utc=True))
    else:
        idx = _parse_text_dates(arr)
    if idx.tz is not None:
        idx = idx.tz_convert("Europe/Athens").tz_localize(None)   # η τοπική ημερομηνία της συνεδρίασης
    return idx.normalize().as_unit("us")


def clean_ohlcv(df: pd.DataFrame | None, report: dict | None = None) -> pd.DataFrame:
    """Επιβάλλει το σχήμα date→open,high,low,close,volume (float), ταξινομεί, αφαιρεί διπλές/κενές
    ημερομηνίες και γραμμές χωρίς κλείσιμο, και διορθώνει τη συνέπεια high/low.

    Μηδενικές/αρνητικές ή κενές τιμές open/high/low (συνηθισμένες στο Yahoo για ημέρες χωρίς
    συναλλαγές) αντικαθίστανται από το κλείσιμο· κενός ή αρνητικός όγκος γίνεται 0.
    `report` (προαιρετικό dict): συμπληρώνεται με rows_in, bad_dates, bad_close, duplicates.
    """
    if df is None or len(df) == 0:
        if report is not None:
            report.update(rows_in=0, bad_dates=0, bad_close=0, duplicates=0)
        return _empty_frame()
    d = df.copy()
    d.columns = [str(c).strip().lower().replace(" ", "_") for c in d.columns]
    for date_col in ("date", "datetime", "timestamp"):
        if date_col in d.columns:
            d = d.set_index(date_col)
            break
    if "close" not in d.columns and "adj_close" in d.columns:
        d["close"] = d["adj_close"]
    if "close" not in d.columns:
        raise ValueError("Λείπει η στήλη close από τα δεδομένα τιμών")

    out = pd.DataFrame(index=_to_dates(d.index))
    for col in OHLCV:
        vals = pd.to_numeric(d[col], errors="coerce").to_numpy(dtype="float64") if col in d.columns \
            else np.full(len(d), np.nan)
        out[col] = vals
    out.index.name = "date"

    n_in = len(out)
    out = out[out.index.notna()]
    n_dates = len(out)
    close = out["close"].where(np.isfinite(out["close"]) & (out["close"] > 0))
    out = out.assign(close=close).dropna(subset=["close"])
    n_close = len(out)
    out = out.sort_index(kind="mergesort")
    out = out[~out.index.duplicated(keep="last")]
    if report is not None:
        report.update(rows_in=n_in, bad_dates=n_in - n_dates, bad_close=n_dates - n_close,
                      duplicates=n_close - len(out))

    c = out["close"]
    fix = {}
    for col in ("open", "high", "low"):
        v = out[col]
        fix[col] = v.where(np.isfinite(v) & (v > 0), c)
    vol = out["volume"]
    fix["volume"] = vol.where(np.isfinite(vol) & (vol >= 0), 0.0)
    out = out.assign(**fix)
    out["high"] = out[["high", "open", "close"]].max(axis=1)
    out["low"] = out[["low", "open", "close"]].min(axis=1)
    return out[OHLCV].astype("float64")


# ---------------------------------------------------------------- αρχεία

def _write_csv(cfg, ticker: str, df: pd.DataFrame) -> None:
    path = _csv_path(cfg, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index_label="date", date_format="%Y-%m-%d")
    tmp.replace(path)


def _read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return clean_ohlcv(pd.read_csv(path))
    except (OSError, ValueError, KeyError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None


def _sniff_csv(path: Path) -> tuple[str, bool]:
    """(διαχωριστικό, δεκαδικό κόμμα;) ενός CSV χρήστη — π.χ. εξαγωγή Excel με ελληνικές ρυθμίσεις:
    «;» ως διαχωριστικό και «12,34» ως αριθμός."""
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        head = [fh.readline() for _ in range(6)]
    first = head[0] if head else ""
    counts = {sep: first.count(sep) for sep in (";", "\t", ",")}
    sep = max(counts, key=counts.get) if any(counts.values()) else ","
    comma = sep != "," and bool(re.search(r"\d,\d", "".join(head[1:])))
    return sep, comma


# ελληνικές/εναλλακτικές κεφαλίδες → σχήμα της εφαρμογής (η σύγκριση γίνεται χωρίς τόνους/κεφαλαία)
HEADER_ALIASES = {
    "ημερομηνια": "date", "ημ/νια": "date", "time": "date",
    "ανοιγμα": "open", "υψηλο": "high", "μεγιστο": "high", "χαμηλο": "low", "ελαχιστο": "low",
    "κλεισιμο": "close", "τιμη κλεισιματος": "close", "τελευταια": "close", "ογκος": "volume",
    "adj close": "adj_close", "adj_close": "adj_close",
}
_HEADER_ALIASES = {fold(k): v for k, v in HEADER_ALIASES.items()}


def _to_number(s: pd.Series, decimal_comma: bool) -> pd.Series:
    t = s.astype("str").str.strip().str.replace(" ", "", regex=False).str.replace(" ", "", regex=False)
    if decimal_comma:
        t = t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(t, errors="coerce")


def _read_user_csv(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    """CSV του χρήστη → (καθαρό frame ή None, ελληνικό μήνυμα προβλήματος ή None).

    Δέχεται «,» / «;» / tab ως διαχωριστικό, δεκαδικό κόμμα («12,34» ή «1.234,56»), ημερομηνίες ISO
    (2024-01-02) ή ημέρα/μήνας/έτος (02/01/2024) και ελληνικές κεφαλίδες (Ημερομηνία, Κλείσιμο, …).
    Γραμμές με μη αναγνώσιμη ημερομηνία ή κλείσιμο αγνοούνται και αναφέρονται στο μήνυμα.
    """
    name = path.name
    try:
        sep, comma = _sniff_csv(path)
        raw = pd.read_csv(path, encoding="utf-8-sig", sep=sep, dtype="str", keep_default_na=False)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as e:
        return None, f"{name}: δεν διαβάζεται ως CSV ({type(e).__name__})."
    cols = []
    for c in raw.columns:
        k = fold(str(c)).strip().replace("_", " ") if str(c).strip() else str(c)
        k = _HEADER_ALIASES.get(k, _HEADER_ALIASES.get(k.replace(" ", "_"), k))
        cols.append(k)
    raw.columns = cols
    date_col = next((c for c in cols if c in ("date", "datetime", "timestamp")), cols[0] if cols else None)
    if date_col is None:
        return None, f"{name}: το αρχείο είναι κενό."
    raw = raw.rename(columns={date_col: "date"})
    for c in raw.columns:
        if c != "date":
            raw[c] = _to_number(raw[c], comma)
    report: dict = {}
    try:
        df = clean_ohlcv(raw, report=report)
    except ValueError:
        cols = ", ".join(str(c) for c in raw.columns[:8])
        return None, (f"{name}: λείπει η στήλη «close» — αναμένεται κεφαλίδα date,open,high,low,close,volume "
                      f"(βρέθηκαν: {cols}).")
    if not len(df):
        return None, f"{name}: καμία γραμμή με έγκυρη ημερομηνία και τιμή κλεισίματος."
    bad = report.get("bad_dates", 0) + report.get("bad_close", 0)
    note = None
    if bad:
        note = (f"{name}: αγνοήθηκαν {bad} από {report.get('rows_in', len(raw))} γραμμές "
                f"({report.get('bad_dates', 0)} με μη έγκυρη ημερομηνία, {report.get('bad_close', 0)} χωρίς κλείσιμο).")
    return df, note


def _stock_files(cfg) -> list[Path]:
    """Τα CSV μετοχών στην cache (χωρίς τον δείκτη)."""
    pdir = prices_dir(cfg)
    if not pdir.exists():
        return []
    index_name = _safe_name(cfg.data.index_ticker or "")
    return sorted(p for p in pdir.glob("*.csv") if not p.name.startswith("_") and p.stem != index_name)


def _demo_marker(cfg) -> Path:
    """Σημάδι «συνθετικά δεδομένα» ανεξάρτητο από το _meta.json: αν το meta χαθεί ή καταστραφεί,
    τα δεδομένα επίδειξης δεν περνιούνται ποτέ για πραγματικά."""
    return prices_dir(cfg) / DEMO_MARKER


def _wipe_prices(cfg) -> None:
    for p in prices_dir(cfg).glob("*.csv"):
        p.unlink(missing_ok=True)
    _demo_marker(cfg).unlink(missing_ok=True)


def _read_meta(cfg) -> dict:
    p = _meta_path(cfg)
    if p.exists():
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                return meta
        except (ValueError, OSError, UnicodeError):
            pass
    return {}


def _write_meta(cfg, meta: dict) -> None:
    """Ατομική εγγραφή (προσωρινό αρχείο + μετονομασία): ποτέ μισογραμμένο _meta.json."""
    p = _meta_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def _last_date(frames) -> str | None:
    last = [df.index[-1] for df in frames if df is not None and len(df)]
    return max(last).strftime("%Y-%m-%d") if last else None


# ---------------------------------------------------------------- λήψη

def _extract_ticker(raw: pd.DataFrame, ticker: str, single: bool) -> pd.DataFrame | None:
    cols = raw.columns
    if isinstance(cols, pd.MultiIndex):
        for level in range(cols.nlevels):
            if ticker in cols.get_level_values(level):
                return raw.xs(ticker, axis=1, level=level)
        return None
    return raw if single else None


def dividend_adjust(raw: pd.DataFrame | None) -> pd.DataFrame | None:
    """Ημερήσια σειρά του Yahoo *χωρίς* αυτόματη προσαρμογή (auto_adjust=False) → τιμές προσαρμοσμένες
    για μερίσματα και splits, με όγκο τέτοιο ώστε το κλείσιμο × όγκος να είναι ο πραγματικός τζίρος της ημέρας.

    Το Yahoo δίνει Open/High/Low/Close προσαρμοσμένα μόνο για splits και το «Adj Close» προσαρμοσμένο και για
    μερίσματα. Με συντελεστή f = Adj Close / Close (≤ 1 για παλιές ημέρες): OHLC × f για αποδόσεις, ετικέτες
    και χαρακτηριστικά, και όγκος / f — έτσι ο μέσος τζίρος (adv20_eur), ο έλεγχος ρευστότητας, το μέγεθος
    θέσης και τα κόστη στην ημέρα t δεν εξαρτώνται από μερίσματα που πληρώθηκαν αργότερα (καμία ματιά στο
    μέλλον). Χωρίς «Adj Close» (π.χ. δείκτης) η σειρά μένει ως έχει.
    """
    if raw is None or not len(raw):
        return raw
    d = raw.copy()
    norm = {c: str(c).strip().lower().replace(" ", "_") for c in d.columns}
    d = d.rename(columns=norm)
    if "adj_close" not in d.columns or "close" not in d.columns:
        return d
    close = pd.to_numeric(d["close"], errors="coerce")
    adj = pd.to_numeric(d["adj_close"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (adj / close).where((close > 0) & (adj > 0))
    f = f.where(np.isfinite(f)).ffill().bfill().fillna(1.0).clip(lower=1e-6, upper=1e3)
    for k in ("open", "high", "low", "close"):
        if k in d.columns:
            d[k] = pd.to_numeric(d[k], errors="coerce") * f
    if "volume" in d.columns:
        d["volume"] = pd.to_numeric(d["volume"], errors="coerce") / f
    return d.drop(columns=["adj_close"])


def drop_partial_bar(df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Αφαιρεί τη σημερινή ημερήσια μπάρα όσο η συνεδρίαση του ΧΑ δεν έχει κλείσει (ώρα Αθήνας).

    Το Yahoo επιστρέφει κατά τη διάρκεια της συνεδρίασης μια «μισή» μπάρα για σήμερα (με τρέχουσα τιμή και
    μέρος του όγκου). Μια απόφαση «στο κλείσιμο» δεν πρέπει να παίρνεται πάνω της.
    """
    if df is None or not len(df):
        return df
    now_ath = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    now_ath = (now_ath.tz_localize("UTC") if now_ath.tz is None else now_ath).tz_convert(ATHENS_TZ)
    today = pd.Timestamp(now_ath.date())
    closed = (now_ath.hour, now_ath.minute) >= SESSION_CLOSE_LOCAL
    if df.index[-1] >= today and not closed:
        return df[df.index < today]
    return df


def _clean_yahoo(sub) -> pd.DataFrame | None:
    try:
        return clean_ohlcv(dividend_adjust(sub)) if sub is not None else None
    except (ValueError, KeyError, TypeError):
        return None


def fetch_yahoo(tickers: list[str], years: int) -> dict[str, pd.DataFrame]:
    """Λήψη ημερήσιων τιμών από Yahoo Finance μέσω yfinance (auto_adjust=False + `dividend_adjust`).

    Σύμβολα που αποτυγχάνουν απλώς παραλείπονται. Αν η μαζική λήψη δεν φέρει τίποτα, γίνεται μία μόνο
    δοκιμή ένα‑ένα: αν αποτύχει κι αυτή, πιθανότατα δεν υπάρχει σύνδεση και δεν περιμένουμε για κάθε σύμβολο.
    """
    tickers = list(dict.fromkeys(t for t in tickers if t))
    if not tickers:
        return {}
    try:
        import yfinance as yf  # τοπική εισαγωγή: δεν απαιτείται για demo/tests
    except Exception:  # λείπει ή δεν φορτώνεται ⇒ καμία λήψη
        return {}
    start = (pd.Timestamp.today().normalize() - pd.DateOffset(years=max(1, int(years)))).strftime("%Y-%m-%d")
    out: dict[str, pd.DataFrame] = {}
    try:
        raw = yf.download(tickers, start=start, interval="1d", auto_adjust=False, actions=False,
                          group_by="ticker", progress=False, threads=True)
    except Exception:
        raw = None
    if raw is not None and len(raw):
        for t in tickers:
            try:
                sub = _extract_ticker(raw, t, single=len(tickers) == 1)
            except (ValueError, KeyError, TypeError):
                sub = None
            df = _clean_yahoo(sub)
            if df is not None and len(df):
                out[t] = df
    for t in tickers:  # δεύτερη προσπάθεια, ένα‑ένα, για όσα έλειψαν
        if t in out:
            continue
        try:
            df = _clean_yahoo(yf.Ticker(t).history(start=start, interval="1d", auto_adjust=False, actions=False))
        except Exception:
            df = None
        if df is not None and len(df):
            out[t] = df
        elif not out:
            break           # τίποτα από τη μαζική λήψη και τίποτα εδώ ⇒ χωρίς σύνδεση· όχι αναμονή για όλα
    return out


def _read_user_csvs(cfg, tickers: list[str], problems: dict | None = None) -> dict[str, pd.DataFrame]:
    """Αρχεία data/csv/<TICKER>.csv (ή χωρίς «.AT»). `problems`: {ticker: ελληνικό μήνυμα} για όσα έχουν θέμα."""
    out = {}
    base = user_csv_dir(cfg)
    for t in tickers:
        for name in (_safe_name(t), _safe_name(t).removesuffix(".AT")):
            p = base / f"{name}.csv"
            if p.exists():
                df, note = _read_user_csv(p)
                if df is not None and len(df):
                    out[t] = df
                if note and problems is not None:
                    problems[t] = note
                break
    return out


def _download(cfg, tickers: list[str], progress=None, lo: float = 0.0, hi: float = 1.0,
              problems: dict | None = None) -> dict[str, pd.DataFrame]:
    """Λήψη από την πηγή των ρυθμίσεων (yahoo ή csv), σε κομμάτια για ενδείξεις προόδου.

    Από το Yahoo αφαιρείται η «μισή» μπάρα της τρέχουσας συνεδρίασης (`drop_partial_bar`). Αν το πρώτο
    κομμάτι δεν φέρει απολύτως τίποτα, τα υπόλοιπα παραλείπονται (χωρίς σύνδεση δεν έχει νόημα η αναμονή).
    """
    if cfg.data.source == "csv":
        _report(progress, lo, "Ανάγνωση αρχείων CSV…")
        return _read_user_csvs(cfg, tickers, problems)
    frames: dict[str, pd.DataFrame] = {}
    chunks = [tickers[i:i + YAHOO_CHUNK] for i in range(0, len(tickers), YAHOO_CHUNK)] or [[]]
    for i, chunk in enumerate(chunks):
        _report(progress, lo + (hi - lo) * i / len(chunks),
                f"Λήψη τιμών από Yahoo Finance ({min(i * YAHOO_CHUNK, len(tickers))}/{len(tickers)})…")
        if not chunk:
            continue
        got = fetch_yahoo(chunk, cfg.data.history_years) or {}
        frames.update({t: drop_partial_bar(df) for t, df in got.items() if t in chunk})
        if i == 0 and not got and len(chunks) > 1:
            _report(progress, hi, "Το Yahoo Finance δεν απαντά — η λήψη σταμάτησε")
            break
    return frames


def _splice_truncated(old: pd.DataFrame | None, new: pd.DataFrame,
                      expected_start=None) -> tuple[pd.DataFrame, str]:
    """Συγχώνευση νέας λήψης με την αποθηκευμένη σειρά. Επιστρέφει (σειρά, "full"|"spliced"|"kept").

    * Κανονική λήψη (ξεκινά το πολύ TRUNCATED_DAYS μετά την αρχή που ζητήθηκε, `expected_start`, ή μετά την
      αρχή της παλιάς σειράς): αντικαθιστά την παλιά — έτσι ενσωματώνονται και νέες προσαρμογές για
      μερίσματα σε όλο το ιστορικό. Χωρίς `expected_start` (π.χ. αρχεία CSV του χρήστη) πάντα αντικατάσταση.
    * «Κομμένη» λήψη (μόνο οι τελευταίες ημέρες) με επικάλυψη: το παλιό ιστορικό πριν από τη νέα σειρά κρατιέται,
      αναγμένο στη βάση προσαρμογής της νέας (λόγος κλεισιμάτων στις κοινές ημέρες) ώστε να μην υπάρχει άλμα.
    * «Κομμένη» λήψη χωρίς επικάλυψη: κρατιέται η παλιά σειρά (η νέα δεν μπορεί να ενωθεί με ασφάλεια).
    """
    if old is None or not len(old) or not len(new) or expected_start is None:
        return new, "full"
    slack = pd.Timedelta(days=TRUNCATED_DAYS)
    if new.index[0] <= max(pd.Timestamp(expected_start), old.index[0]) + slack:
        return new, "full"
    common = old.index.intersection(new.index)
    if not len(common):
        return old, "kept"
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (new.loc[common, "close"] / old.loc[common, "close"]).to_numpy("float64")
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    r = float(np.median(ratio)) if len(ratio) else 1.0
    head = old[old.index < new.index[0]].copy()
    for k in ("open", "high", "low", "close"):
        head[k] = head[k] * r
    head["volume"] = head["volume"] / r
    return pd.concat([head, new]).sort_index(), "spliced"


def update_cache(cfg, stocks: list[Stock], progress=None) -> dict:
    """Κατεβάζει μετοχές + δείκτη, γράφει τα CSV και επιστρέφει {"ok": [...], "failed": [...], ...}.

    Αν η cache περιείχε συνθετικά δεδομένα (demo), διαγράφονται μόλις έρθει έστω και μία πραγματική
    σειρά, ώστε να μη μπερδευτούν ποτέ συνθετικά με πραγματικά δεδομένα. Αν δεν έρθει τίποτα, η cache
    μένει όπως ήταν. Παλιά πραγματικά δεδομένα μετοχής που απέτυχε τώρα διατηρούνται· μια «κομμένη» λήψη
    (μόνο οι τελευταίες ημέρες) δεν σβήνει ποτέ το αποθηκευμένο ιστορικό (`_splice_truncated`).
    Επιπλέον κλειδιά: "index_ok", "index_last_date", "problems" ({ticker: ελληνικό μήνυμα}), "spliced", "mode".
    """
    tickers = list(dict.fromkeys(s.ticker for s in stocks))
    index_ticker = cfg.data.index_ticker
    want = tickers + ([index_ticker] if index_ticker and index_ticker not in tickers else [])
    _report(progress, 0.0, "Έναρξη λήψης δεδομένων…")
    problems: dict[str, str] = {}
    frames = {t: clean_ohlcv(df) for t, df in _download(cfg, want, progress, 0.0, 0.9, problems).items()}
    frames = {t: df for t, df in frames.items() if len(df)}

    was_demo = data_mode(cfg) == "demo"
    ok = [t for t in tickers if t in frames]
    failed = [t for t in tickers if t not in frames]
    index_ok = bool(index_ticker) and index_ticker in frames
    result = {"ok": ok, "failed": failed, "index_ok": index_ok, "mode": data_mode(cfg), "problems": problems,
              "spliced": []}
    if not ok:
        _report(progress, 1.0, "Δεν ήρθαν δεδομένα — η αποθήκη έμεινε ως είχε")
        return result

    _report(progress, 0.92, "Αποθήκευση δεδομένων…")
    prices_dir(cfg).mkdir(parents=True, exist_ok=True)
    expected_start = None
    if cfg.data.source != "csv":
        expected_start = pd.Timestamp.today().normalize() - pd.DateOffset(years=max(1, int(cfg.data.history_years)))
    if was_demo:
        _wipe_prices(cfg)
    for t in ok + ([index_ticker] if index_ok else []):
        old = None if was_demo else _read_csv(_csv_path(cfg, t)) if _csv_path(cfg, t).exists() else None
        df, how = _splice_truncated(old, frames[t], expected_start)
        if how != "full":
            result["spliced"].append(t)
            problems.setdefault(t, "Η λήψη έφερε μόνο τις τελευταίες ημέρες — κρατήθηκε το αποθηκευμένο ιστορικό.")
        frames[t] = df
        if how != "kept":
            _write_csv(cfg, t, df)
    index_last = frames[index_ticker].index[-1].strftime("%Y-%m-%d") if index_ok else None
    if not index_ok and index_ticker and _csv_path(cfg, index_ticker).exists():
        old_idx = _read_csv(_csv_path(cfg, index_ticker))
        index_last = old_idx.index[-1].strftime("%Y-%m-%d") if old_idx is not None and len(old_idx) else None

    meta = {
        "mode": "live",
        "source": cfg.data.source,
        "updated_at": _now_iso(),
        "last_date": _last_date(frames[t] for t in ok),
        "tickers": len(_stock_files(cfg)),
        "failed": failed,
        "index_ok": index_ok,
        "index_last_date": index_last,
        "problems": problems,
    }
    _write_meta(cfg, meta)
    result.update(mode="live", index_last_date=index_last)
    _report(progress, 1.0, f"Ολοκληρώθηκε: {len(ok)} μετοχές, {len(failed)} αποτυχίες")
    return result


def write_demo_data(cfg, stocks: list[Stock], seed: int = 7) -> None:
    """Γράφει συνθετικά δεδομένα στην cache και σημειώνει τη λειτουργία ως «demo» (meta + σημάδι _DEMO)."""
    end = synthetic.default_end()
    start = (end - pd.DateOffset(years=int(cfg.data.history_years))).normalize()
    prices, index_df = synthetic.generate_universe(stocks, start=start, end=end, seed=seed)
    prices_dir(cfg).mkdir(parents=True, exist_ok=True)
    _wipe_prices(cfg)
    _demo_marker(cfg).write_text("Συνθετικά δεδομένα επίδειξης — όχι πραγματικές τιμές.\n", encoding="utf-8")
    for t, df in prices.items():
        _write_csv(cfg, t, df)
    if cfg.data.index_ticker:
        _write_csv(cfg, cfg.data.index_ticker, index_df)
    _write_meta(cfg, {
        "mode": "demo",
        "source": "synthetic",
        "updated_at": _now_iso(),
        "last_date": end.strftime("%Y-%m-%d"),
        "start_date": pd.Timestamp(start).strftime("%Y-%m-%d"),
        "seed": int(seed),
        "tickers": len(prices),
        "failed": [],
        "index_ok": bool(cfg.data.index_ticker),
    })


def add_stock_data(cfg, stock: Stock) -> bool:
    """Δεδομένα για μία (νέα) μετοχή: συνθετικά σε demo, αλλιώς από την πηγή. True αν γράφτηκαν.

    Σε demo η σειρά είναι ίδια με αυτή που θα έβγαζε το write_demo_data με τον ίδιο σπόρο. Με πραγματικά
    δεδομένα η νέα σειρά κόβεται στην τελευταία ημερομηνία της αποθήκης: αλλιώς μια μετοχή που προστέθηκε
    μία μέρα μετά την τελευταία ενημέρωση θα «μετέφερε» την ημερομηνία των σημάτων και όλες οι άλλες θα
    έμεναν χωρίς τρέχον σήμα. Την επεκτείνει η επόμενη πλήρης ενημέρωση.
    """
    if stock.ticker == cfg.data.index_ticker:
        return False
    meta = _read_meta(cfg)
    if data_mode(cfg) == "demo":
        prices, _ = synthetic.generate_universe([stock], start=meta.get("start_date", "2017-01-02"),
                                                end=meta.get("last_date"), seed=int(meta.get("seed", 7)))
        df = prices.get(stock.ticker)
    else:
        df = _download(cfg, [stock.ticker]).get(stock.ticker)
        df = clean_ohlcv(df) if df is not None else None
        last = meta.get("last_date") if _stock_files(cfg) else None
        if df is not None and last:
            df = df[df.index <= pd.Timestamp(last)]
    if df is None or not len(df):
        return False
    _write_csv(cfg, stock.ticker, df)
    if meta:
        meta["tickers"] = len(_stock_files(cfg))
        meta["failed"] = [t for t in meta.get("failed", []) if t != stock.ticker]
        _write_meta(cfg, meta)
    else:
        _write_meta(cfg, {"mode": "live", "source": cfg.data.source, "updated_at": _now_iso(),
                          "last_date": _last_date([df]), "tickers": len(_stock_files(cfg)), "failed": []})
    return True


# ---------------------------------------------------------------- ανάγνωση

def load_prices(cfg, tickers: list[str] | None = None) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    """Φορτώνει τιμές από την cache. `tickers=None` ⇒ όλα τα διαθέσιμα. Επιστρέφει (prices, index_df ή None)."""
    prices: dict[str, pd.DataFrame] = {}
    if tickers is None:
        paths = [(p.stem, p) for p in _stock_files(cfg)]
    else:
        paths = [(t, _csv_path(cfg, t)) for t in dict.fromkeys(tickers)]
    for t, p in paths:
        if not p.exists():
            continue
        df = _read_csv(p)
        if df is not None and len(df):
            prices[t] = df
    index_df = None
    if cfg.data.index_ticker:
        p = _csv_path(cfg, cfg.data.index_ticker)
        if p.exists():
            df = _read_csv(p)
            index_df = df if df is not None and len(df) else None
    return prices, index_df


def data_mode(cfg) -> str:
    """"demo" (συνθετικά), "live" (πραγματικά) ή "empty" (καθόλου δεδομένα).

    Κλειστό σε σφάλμα: αν το _meta.json λείπει ή είναι κατεστραμμένο αλλά υπάρχει το σημάδι _DEMO,
    η λειτουργία είναι «demo» — ποτέ δεν παρουσιάζονται συνθετικά δεδομένα ως πραγματικά.
    """
    if not _stock_files(cfg):
        return "empty"
    mode = _read_meta(cfg).get("mode")
    if _demo_marker(cfg).exists() or mode == "demo":
        return "demo"
    return "live"


def data_meta(cfg) -> dict:
    """{"mode", "updated_at", "last_date", "tickers", "failed"} (+ ό,τι άλλο είναι αποθηκευμένο)."""
    meta = _read_meta(cfg)
    files = _stock_files(cfg)
    out = dict(meta)
    out["mode"] = data_mode(cfg)
    out["tickers"] = len(files)
    out.setdefault("updated_at", None)
    out.setdefault("failed", [])
    if "last_date" not in out:
        frames = [_read_csv(p) for p in files]
        out["last_date"] = _last_date(frames)
    if out["mode"] == "empty":
        out["last_date"] = None
    return out
