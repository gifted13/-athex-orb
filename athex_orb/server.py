"""Διακομιστής FastAPI: το HTTP API του ARCHITECTURE.md, εργασίες στο παρασκήνιο και το UI (web/).

* `create_app(cfg)` φτιάχνει την εφαρμογή· το `app` του module (για `uvicorn athex_orb.server:app`)
  δημιουργείται «τεμπέλικα» με τις προεπιλεγμένες ρυθμίσεις.
* Τα δεδομένα των σελίδων διαβάζονται από τα αρχεία του data/state/ (τα γράφει το pipeline). Αν λείπουν,
  κάθε endpoint απαντά κανονικά με κενές λίστες / `has_data: false` / mode "empty" (το UI δείχνει onboarding).
* Εργασίες (ενημέρωση, backtest, επίδειξη) τρέχουν σε ένα νήμα‑εργάτη, μία κάθε φορά, με πρόοδο και
  ελληνικό μήνυμα. Δεύτερη εργασία ίδιου τύπου όσο η πρώτη περιμένει/τρέχει ⇒ 409.
* Σφάλματα: HTTP 4xx/5xx με {"detail": "<ελληνικό μήνυμα>"}.
"""
from __future__ import annotations

import copy
import json
import math
import queue
import sys
import threading
import traceback
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__, journal, pipeline, scout
from .guard import fmt_num
from .config import ROOT, USER_SETTINGS_SPEC, load_config, save_user_settings
from .universe import Stock, fold, load_universe, normalize_ticker, remove_custom
from .universe import search as universe_search

WEB_DIR = ROOT / "web"
JOB_TYPES = ("refresh", "backtest", "demo")
JOB_TYPES_EL = {"refresh": "Ενημέρωση", "backtest": "Backtest", "demo": "Δεδομένα επίδειξης"}
MAX_JOBS_KEPT = 30
MAX_DAYS = pipeline.N_CANDLES
DEFAULT_SECTOR = "Άλλο"

# Επεξεργάσιμες ρυθμίσεις: (τύπος, ελάχιστο, μέγιστο, ετικέτα) — κοινές με τη φόρτωση του settings.json.
SETTINGS_SPEC: dict[str, dict[str, tuple]] = USER_SETTINGS_SPEC
RETRAIN_SECTIONS = ("barriers", "model")
PCT_KEYS = {"max_adv_fraction", "max_atr_pct", "crash_index_drop", "min_ev_net", "min_p_tp", "commission_pct",
            "exchange_fees_pct", "sell_tax_pct"}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _err(status: int, detail: str, **extra) -> JSONResponse:
    return JSONResponse({"detail": detail, **extra}, status_code=status)


# ---------------------------------------------------------------- ρυθμίσεις

def settings_subset(cfg) -> dict:
    d = cfg.to_dict()
    return {"portfolio": d["portfolio"], "barriers": d["barriers"], "guard": d["guard"], "costs": d["costs"],
            "model": {"kind": d["model"]["kind"]}}


def _fmt_bound(key: str, v: float, kind: str) -> str:
    """Όριο ρύθμισης για ελληνικό μήνυμα: «1.000.000.000 €», «0,3%», «0,2» — ποτέ «1e+09»."""
    if kind == "int":
        return fmt_num(int(v), 0)
    if key in PCT_KEYS:
        x = v * 100
        nd = 0 if float(x).is_integer() else (1 if float(x * 10).is_integer() else 2)
        return f"{fmt_num(x, nd)}%".replace("-", "−")
    nd = 0 if float(v).is_integer() else (1 if float(v * 10).is_integer() else 2)
    unit = " €" if key.endswith("_eur") else ""
    return f"{fmt_num(v, nd)}{unit}".replace("-", "−")


def validate_settings(body) -> tuple[dict, list[dict]]:
    """Έλεγχος του σώματος PUT /api/settings. Επιστρέφει (καθαρές τιμές, σφάλματα)."""
    clean: dict[str, dict] = {}
    errors: list[dict] = []
    if not isinstance(body, dict):
        return {}, [{"field": None, "message_el": "Οι ρυθμίσεις πρέπει να είναι αντικείμενο JSON."}]
    for section, values in body.items():
        spec = SETTINGS_SPEC.get(section)
        if spec is None:
            errors.append({"field": section, "message_el": f"Άγνωστη ενότητα ρυθμίσεων «{section}»."})
            continue
        if not isinstance(values, dict):
            errors.append({"field": section, "message_el": f"Η ενότητα «{section}» πρέπει να είναι αντικείμενο."})
            continue
        for key, raw in values.items():
            field = f"{section}.{key}"
            if key not in spec:
                errors.append({"field": field, "message_el": f"Άγνωστη ρύθμιση «{field}»."})
                continue
            kind, lo, hi, label = spec[key]
            if kind == "choice":
                if raw not in lo:
                    errors.append({"field": field, "message_el": f"«{label}»: επιτρέπονται μόνο {', '.join(lo)}."})
                else:
                    clean.setdefault(section, {})[key] = raw
                continue
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
                errors.append({"field": field, "message_el": f"«{label}»: χρειάζεται αριθμός."})
                continue
            v = float(raw)
            if kind == "int":
                if v != int(v):
                    errors.append({"field": field, "message_el": f"«{label}»: χρειάζεται ακέραιος αριθμός."})
                    continue
                v = int(v)
            if v < lo or v > hi:
                errors.append({"field": field, "message_el": (f"«{label}»: η τιμή πρέπει να είναι από "
                                                              f"{_fmt_bound(key, lo, kind)} έως {_fmt_bound(key, hi, kind)}.")})
                continue
            clean.setdefault(section, {})[key] = v
    return clean, errors


# ---------------------------------------------------------------- αγαπημένα

def favorites_path(cfg) -> Path:
    return Path(cfg.data_dir) / "favorites.json"


def load_favorites(cfg) -> list[str]:
    try:
        data = json.loads(favorites_path(cfg).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    items = data.get("tickers", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    # φιλτράρισμα πριν από την αφαίρεση διπλών: μια φωλιασμένη λίστα/λεξικό δεν είναι hashable
    return list(dict.fromkeys(t for t in items if isinstance(t, str) and t))


def save_favorites(cfg, tickers: list[str]) -> None:
    p = favorites_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"tickers": list(dict.fromkeys(tickers))}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(p)


# ---------------------------------------------------------------- ανάγνωση κατάστασης (με cache ανά mtime)

class StateReader:
    def __init__(self, cfg):
        self.cfg = cfg
        self._cache: dict[str, tuple] = {}
        self._lock = threading.Lock()

    def _read(self, path: Path):
        try:
            st = path.stat()
        except OSError:
            return None
        key = (st.st_mtime_ns, st.st_size)
        with self._lock:
            hit = self._cache.get(str(path))
            if hit and hit[0] == key:
                return hit[1]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        with self._lock:
            self._cache[str(path)] = (key, data)
        return data

    def get(self, name: str):
        return self._read(pipeline.state_dir(self.cfg) / f"{name}.json")

    def stock(self, ticker: str):
        return self._read(pipeline._stock_file(self.cfg, ticker))


# ---------------------------------------------------------------- εργασίες

class JobConflict(Exception):
    pass


class JobManager:
    """Μητρώο εργασιών + ένα νήμα‑εργάτης (μία εργασία κάθε φορά, με σειρά υποβολής)."""

    def __init__(self, cfg_getter, runners: dict):
        self._cfg_getter = cfg_getter
        self._runners = runners
        self._jobs: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None

    def _ensure_worker(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._work, name="athex-orb-jobs", daemon=True)
            self._thread.start()

    def submit(self, typ: str) -> dict:
        with self._lock:
            for j in self._jobs.values():
                if j["type"] == typ and j["status"] in ("queued", "running"):
                    raise JobConflict(typ)
            jid = uuid.uuid4().hex[:12]
            job = {"id": jid, "type": typ, "type_el": JOB_TYPES_EL.get(typ, typ), "status": "queued",
                   "progress": 0.0, "message_el": "Σε αναμονή…", "created_at": _now_iso(), "started_at": None,
                   "finished_at": None, "error": None, "result": None}
            self._jobs[jid] = job
            while len(self._jobs) > MAX_JOBS_KEPT:
                old = next(iter(self._jobs))
                if self._jobs[old]["status"] in ("queued", "running"):
                    break
                self._jobs.pop(old)
            self._ensure_worker()
        self._queue.put(jid)
        return dict(job)

    def get(self, jid: str) -> dict | None:
        with self._lock:
            j = self._jobs.get(jid)
            return dict(j) if j else None

    def list(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return [dict(j) for j in list(self._jobs.values())[::-1][:limit]]

    def active(self) -> list[dict]:
        with self._lock:
            return [dict(j) for j in self._jobs.values() if j["status"] in ("queued", "running")]

    def busy(self, types=JOB_TYPES) -> bool:
        return any(j["type"] in types for j in self.active())

    def _update(self, jid: str, **kw) -> None:
        with self._lock:
            j = self._jobs.get(jid)
            if j:
                j.update(kw)

    def _work(self) -> None:
        while True:
            jid = self._queue.get()
            job = self.get(jid)
            if job is None:
                continue
            self._update(jid, status="running", started_at=_now_iso(), message_el="Έναρξη…")

            def progress(fraction: float, message_el: str, _jid=jid) -> None:
                with self._lock:
                    j = self._jobs.get(_jid)
                    if j:
                        j["progress"] = round(max(j["progress"], min(max(float(fraction), 0.0), 0.999)), 4)
                        j["message_el"] = str(message_el)

            try:
                cfg = copy.deepcopy(self._cfg_getter())     # οι ρυθμίσεις τη στιγμή που ξεκινά η εργασία
                result = self._runners[job["type"]](cfg, progress)
                self._update(jid, status="done", progress=1.0, message_el="Ολοκληρώθηκε",
                             finished_at=_now_iso(), result=_summarize(job["type"], result))
            except pipeline.PipelineError as e:
                self._update(jid, status="error", message_el=str(e), error=str(e), finished_at=_now_iso())
            except Exception as e:  # απρόσμενο: καταγραφή + μήνυμα στον χρήστη
                traceback.print_exc(file=sys.stderr)
                msg = f"Απρόσμενο σφάλμα: {type(e).__name__}: {e}"
                self._update(jid, status="error", message_el=msg, error=msg, finished_at=_now_iso())


def _summarize(typ: str, result) -> dict | None:
    """Μικρή, ασφαλής για JSON σύνοψη αποτελέσματος εργασίας."""
    if not isinstance(result, dict):
        return None
    try:
        if typ == "backtest":
            return pipeline._clean({"verdict": result.get("verdict"), "n_trades": result.get("n_trades")})
        if typ == "demo":
            ref, bt = result.get("refresh") or {}, result.get("backtest") or {}
            return pipeline._clean({"as_of": ref.get("as_of"), "counts": ref.get("counts"),
                                    "verdict": bt.get("verdict")})
        return pipeline._clean({k: result.get(k) for k in ("as_of", "counts", "failed", "mode", "n_with_data")})
    except Exception:
        return None


def _default_runners() -> dict:
    # οι συναρτήσεις αναζητούνται τη στιγμή της εκτέλεσης (ώστε να αντικαθίστανται σε tests)
    return {
        "refresh": lambda cfg, progress: pipeline.run_refresh(cfg, progress=progress),
        "backtest": lambda cfg, progress: pipeline.run_backtest_job(cfg, progress=progress),
        "demo": lambda cfg, progress: pipeline.run_demo(cfg, progress=progress),
    }


# ---------------------------------------------------------------- στατικά αρχεία (UI)

ASSET_SUFFIXES = (".js", ".mjs", ".css", ".map", ".svg", ".png", ".jpg", ".jpeg", ".webp", ".ico", ".gif",
                  ".woff", ".woff2", ".ttf", ".json", ".txt", ".webmanifest")


class SPAStaticFiles(StaticFiles):
    """Στατικά αρχεία με επιστροφή στο index.html για άγνωστες διαδρομές σελίδων (όχι /api, όχι αρχεία)."""

    @staticmethod
    def _fallback(path: str) -> bool:
        return not path.startswith("api") and not path.lower().endswith(ASSET_SUFFIXES)

    async def get_response(self, path: str, scope):
        if path == "dev" or path.startswith(("dev/", "dev\\")):
            # web/dev/ = εργαλεία ανάπτυξης (mock διακομιστής, έλεγχοι UI)· δεν σερβίρονται από την εφαρμογή
            raise StarletteHTTPException(status_code=404)
        try:
            resp = await super().get_response(path, scope)
        except StarletteHTTPException as e:
            if e.status_code != 404 or not self._fallback(path):
                raise
            resp = await super().get_response("index.html", scope)
        else:
            if resp.status_code == 404 and self._fallback(path):
                resp = await super().get_response("index.html", scope)
        return resp


# ---------------------------------------------------------------- εφαρμογή

def create_app(cfg=None, web_dir: Path | str | None = None) -> FastAPI:
    """Η εφαρμογή FastAPI. `cfg`: ρυθμίσεις (προεπιλογή: load_config()), `web_dir`: φάκελος του UI."""
    cfg = cfg if cfg is not None else load_config()
    app = FastAPI(title="ATHEX Orb", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json",
                  redoc_url=None)
    state = StateReader(cfg)
    jobs = JobManager(lambda: cfg, _default_runners())
    fav_lock = threading.Lock()
    app.state.cfg = cfg
    app.state.jobs = jobs
    app.state.reader = state

    # ------------------------------------------------------------ σφάλματα στα ελληνικά
    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        detail = exc.detail if isinstance(exc.detail, str) and exc.detail else {
            404: "Δεν βρέθηκε.", 405: "Η μέθοδος δεν επιτρέπεται.", 409: "Υπάρχει σύγκρουση με εκτελούμενη εργασία.",
        }.get(exc.status_code, "Σφάλμα αιτήματος.")
        if exc.status_code == 404 and detail == "Not Found":
            detail = "Δεν βρέθηκε."
        return _err(exc.status_code, detail)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return _err(422, "Μη έγκυρο αίτημα: ελέγξτε τα δεδομένα που στάλθηκαν (JSON).")

    @app.exception_handler(Exception)
    async def server_error(request: Request, exc: Exception):
        traceback.print_exception(exc, file=sys.stderr)
        return _err(500, f"Σφάλμα διακομιστή: {type(exc).__name__}.")

    @app.middleware("http")
    async def no_cache(request: Request, call_next):
        resp = await call_next(request)
        if request.url.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        else:
            resp.headers.setdefault("Cache-Control", "no-cache")
        return resp

    # ------------------------------------------------------------ βοηθητικά
    def stocks() -> list[Stock]:
        return load_universe(cfg.data_dir)

    def find(raw: str, universe: list[Stock] | None = None) -> Stock | None:
        universe = universe if universe is not None else stocks()
        by = {s.ticker: s for s in universe}
        if raw in by:
            return by[raw]
        t = normalize_ticker(raw or "")
        return by.get(t) if t else None

    def data_mode() -> str:
        return scout.data_mode(cfg)

    def mode() -> str:
        """«empty» αν δεν υπάρχουν δεδομένα ή δεν έχει γίνει ακόμη υπολογισμός."""
        if state.get("universe") is None:
            return "empty"
        return data_mode()

    def journal_mode() -> str | None:
        m = data_mode()
        return m if m in ("demo", "live") else None

    def default_item(s: Stock) -> dict:
        return {"ticker": s.ticker, "name": s.name, "sector": s.sector, "custom": bool(s.custom), "has_data": False,
                "last_close": None, "change_1d": None, "change_5d": None, "spark": [], "adv20_eur": None,
                "orb": None, "decision": "none", "as_of": None}

    def items_for(universe: list[Stock], favs: list[str]) -> list[dict]:
        uni = state.get("universe") or {}
        by = {it.get("ticker"): it for it in uni.get("items", []) if isinstance(it, dict)}
        fav = set(favs)
        out = []
        for s in universe:
            it = dict(by.get(s.ticker) or default_item(s))
            it.update({"name": s.name, "sector": s.sector, "custom": bool(s.custom), "favorite": s.ticker in fav})
            out.append(it)
        return out

    # ------------------------------------------------------------ κατάσταση
    @app.get("/api/status")
    def api_status():
        meta = state.get("meta") or {}
        uni = state.get("universe")
        dm = scout.data_meta(cfg)
        universe = stocks()
        tickers = {s.ticker for s in universe}
        n_with = sum(1 for it in (uni or {}).get("items", []) if it.get("has_data") and it.get("ticker") in tickers)
        fp = meta.get("fingerprint") or {}
        bar = fp.get("barriers") or {}
        bt = state.get("backtest")
        model = meta.get("model") or {}
        return pipeline._clean({
            "mode": mode(),
            "data_mode": dm.get("mode"),
            "data_last_date": dm.get("last_date") if dm.get("mode") != "empty" else None,
            "data_updated_at": dm.get("updated_at") if dm.get("mode") != "empty" else None,
            "as_of": (uni or {}).get("as_of"),
            "model_trained_until": model.get("trained_until"),
            "model_kind": model.get("kind") or cfg.model.kind,
            "n_tickers": len(universe),
            "n_with_data": n_with,
            "horizon_days": bar.get("horizon_days", cfg.barriers.horizon_days),
            "k_tp": bar.get("k_tp", cfg.barriers.k_tp),
            "k_sl": bar.get("k_sl", cfg.barriers.k_sl),
            "counts": meta.get("counts"),
            "regime": (meta.get("market") or {}).get("regime"),
            "regime_el": (meta.get("market") or {}).get("regime_el"),
            "market": meta.get("market"),
            "failed": dm.get("failed", []) if dm.get("mode") == "live" else [],
            "index_ok": meta.get("index_ok"),
            "generated_at": meta.get("generated_at"),
            "retrain_needed": pipeline.needs_retrain(cfg, meta),
            "settings_changed": bool(meta) and meta.get("settings_hash") != pipeline.settings_hash(cfg),
            "backtest_available": bool(bt and bt.get("available")),
            "jobs": jobs.active(),
            "disclaimer_el": pipeline.DISCLAIMER_EL,
            "version": __version__,
        })

    # ------------------------------------------------------------ λίστα μετοχών & αναζήτηση
    @app.get("/api/universe")
    def api_universe():
        uni = state.get("universe") or {}
        return {"as_of": uni.get("as_of"), "items": items_for(stocks(), load_favorites(cfg))}

    @app.get("/api/search")
    def api_search(q: str = "", limit: int = 20):
        universe = stocks()
        found = universe_search(universe, q, limit=max(1, min(int(limit), 100)))
        by = {it["ticker"]: it for it in items_for(universe, load_favorites(cfg))}
        suggest = None
        qs = (q or "").strip()
        t = normalize_ticker(qs)
        if qs and t and qs.isascii() and t not in by and t != cfg.data.index_ticker:
            # όχι πρόταση αν ο χρήστης έγραψε ακριβώς όνομα/ψευδώνυμο γνωστής μετοχής (π.χ. «allwyn»)
            fq = fold(qs)
            exact = any(fq == fold(x) for s in found for x in [s.name, *s.aliases])
            if not exact:
                suggest = {"ticker": t, "valid": True}
        return {"items": [by[s.ticker] for s in found], "suggest": suggest}

    @app.post("/api/universe")
    def api_add(body: dict = Body(...)):
        raw = str(body.get("ticker") or "")
        t = normalize_ticker(raw)
        if not t:
            return {"ok": False, "error_el": "Μη έγκυρο σύμβολο μετοχής. Χρησιμοποιήστε λατινικούς χαρακτήρες, "
                                             "π.χ. «INTEK» ή «INTEK.AT»."}
        if t == cfg.data.index_ticker:
            return {"ok": False, "error_el": f"Το {t} είναι ο Γενικός Δείκτης — δεν προστίθεται ως μετοχή."}
        if jobs.busy(("refresh", "demo")):
            # η εργασία δημοσιεύει στο τέλος τη λίστα που διάβασε όταν ξεκίνησε: μια προσθήκη τώρα θα χανόταν
            raise HTTPException(409, "Τρέχει ενημέρωση δεδομένων — περιμένετε να ολοκληρωθεί και δοκιμάστε ξανά.")
        universe = stocks()
        if any(s.ticker == t for s in universe):
            return {"ok": False, "error_el": f"Η μετοχή {t} υπάρχει ήδη στη λίστα."}
        name = str(body.get("name") or "").strip() or t.removesuffix(".AT")
        sector = str(body.get("sector") or "").strip() or DEFAULT_SECTOR
        stock = Stock(t, name[:80], sector[:40], [], custom=True)
        res = pipeline.add_stock(cfg, stock)
        if not res.get("ok"):
            return {"ok": False, "error_el": res.get("error_el") or f"Δεν ήταν δυνατή η προσθήκη του {t}."}
        item = dict(res.get("item") or default_item(stock))
        item.update({"name": stock.name, "sector": stock.sector, "custom": True,
                     "favorite": t in load_favorites(cfg)})
        out = {"ok": True, "item": pipeline._clean(item)}
        if res.get("note_el"):
            out["note_el"] = res["note_el"]
        return out

    @app.delete("/api/universe/{ticker}")
    def api_remove(ticker: str):
        s = find(ticker)
        if s is None or not s.custom:
            raise HTTPException(404, "Μόνο μετοχές που προσθέσατε εσείς μπορούν να αφαιρεθούν.")
        sig = state.get("signals") or {}
        was_taken = any(it.get("ticker") == s.ticker and it.get("decision") == "take" for it in sig.get("items", []))
        remove_custom(cfg.data_dir, s.ticker)
        pipeline.remove_stock(cfg, s.ticker)
        with fav_lock:
            favs = load_favorites(cfg)
            if s.ticker in favs:
                save_favorites(cfg, [f for f in favs if f != s.ticker])
        if was_taken and not jobs.busy(("refresh", "demo")):
            # μια θέση ελευθερώθηκε: ξαναμοιράζονται οι σημερινές υποψήφιες
            try:
                pipeline.recompute_signals(cfg)
            except Exception:
                traceback.print_exc(file=sys.stderr)
        return {"ok": True, "ticker": s.ticker}

    # ------------------------------------------------------------ αγαπημένα
    @app.get("/api/favorites")
    def api_favorites():
        known = {s.ticker for s in stocks()}
        return {"tickers": [t for t in load_favorites(cfg) if t in known]}

    @app.put("/api/favorites/{ticker}")
    def api_fav_add(ticker: str):
        s = find(ticker)
        if s is None:
            raise HTTPException(404, f"Η μετοχή {ticker} δεν υπάρχει στη λίστα.")
        with fav_lock:
            favs = load_favorites(cfg)
            if s.ticker not in favs:
                favs.append(s.ticker)
                save_favorites(cfg, favs)
        return {"tickers": favs}

    @app.delete("/api/favorites/{ticker}")
    def api_fav_remove(ticker: str):
        s = find(ticker)
        t = s.ticker if s is not None else ticker
        with fav_lock:
            favs = load_favorites(cfg)
            if t in favs:
                favs = [f for f in favs if f != t]
                save_favorites(cfg, favs)
        return {"tickers": favs}

    # ------------------------------------------------------------ μετοχή
    @app.get("/api/stock/{ticker}")
    def api_stock(ticker: str, days: int = 250):
        s = find(ticker)
        if s is None:
            raise HTTPException(404, f"Η μετοχή {ticker} δεν βρέθηκε στη λίστα.")
        days = max(20, min(int(days), MAX_DAYS))
        payload = state.stock(s.ticker)
        if payload is None:
            reason = None
            if data_mode() != "empty" and state.get("universe") is not None:
                reason = "Δεν υπάρχουν δεδομένα για αυτή τη μετοχή — τρέξτε «Ενημέρωση»."
            payload = pipeline.empty_stock_payload(cfg, s, reason, {"mode": data_mode()})
        out = dict(payload)
        out["candles"] = list(payload.get("candles") or [])[-days:]
        out["orb_history"] = list(payload.get("orb_history") or [])[-days:]
        out.update({"name": s.name, "sector": s.sector, "custom": bool(s.custom),
                    "favorite": s.ticker in load_favorites(cfg)})
        return pipeline._clean(out)

    # ------------------------------------------------------------ σήματα, backtest, ημερολόγιο
    @app.get("/api/signals")
    def api_signals():
        sig = state.get("signals") or {}
        known = {s.ticker for s in stocks()}
        fav = set(load_favorites(cfg))
        items = [dict(it, favorite=it.get("ticker") in fav) for it in sig.get("items", [])
                 if it.get("ticker") in known]
        return {"as_of": sig.get("as_of"), "items": items, "counts": sig.get("counts")}

    @app.get("/api/backtest")
    def api_backtest():
        bt = state.get("backtest")
        if not bt or not bt.get("available"):
            return {"available": False,
                    "reason_el": "Δεν έχει τρέξει ακόμη backtest — πατήστε «Τρέξε backtest» μετά από μια ενημέρωση."}
        meta = state.get("meta") or {}
        reasons = []
        if bt.get("preds_id") and meta.get("preds_id") and bt["preds_id"] != meta["preds_id"]:
            reasons.append("το μοντέλο ξαναεκπαιδεύτηκε μετά το backtest")
        if bt.get("settings_hash") and bt["settings_hash"] != pipeline.settings_hash(cfg):
            reasons.append("οι ρυθμίσεις άλλαξαν")
        out = dict(bt)
        out["stale"] = bool(reasons)
        out["stale_reason_el"] = ("Το backtest δεν είναι ενημερωμένο: " + " και ".join(reasons) + ".") if reasons else None
        return out

    @app.get("/api/journal")
    def api_journal(limit: int = 500):
        jm = journal_mode()
        names = {s.ticker: s.name for s in stocks()}
        items = journal.list_entries(cfg, limit=max(1, min(int(limit), 5000)), mode=jm)
        for it in items:
            it["name"] = names.get(it["ticker"], it["ticker"].removesuffix(".AT"))
        return pipeline._clean({"items": items, "summary": journal.summary(cfg, mode=jm), "mode": jm})

    # ------------------------------------------------------------ ρυθμίσεις
    @app.get("/api/settings")
    def api_settings():
        return settings_subset(cfg)

    @app.put("/api/settings")
    def api_settings_put(body: dict = Body(...)):
        clean, errors = validate_settings(body)
        if errors:
            return _err(422, " · ".join(e["message_el"] for e in errors), errors=errors)
        before = settings_subset(cfg)
        changed = {sec: {k: v for k, v in vals.items() if before.get(sec, {}).get(k) != v}
                   for sec, vals in clean.items()}
        changed = {sec: vals for sec, vals in changed.items() if vals}
        if clean:
            save_user_settings(cfg, clean)
        meta = state.get("meta")
        # χρειάζεται νέα εκπαίδευση όταν τα όρια/μοντέλο διαφέρουν από αυτά του εκπαιδευμένου μοντέλου
        retrain = pipeline.needs_retrain(cfg, meta) if meta else any(sec in RETRAIN_SECTIONS for sec in changed)
        updated = False
        if any(sec not in RETRAIN_SECTIONS for sec in changed) and meta is not None \
                and not jobs.busy(("refresh", "demo")):
            try:
                updated = pipeline.recompute_signals(cfg) is not None
            except Exception:
                traceback.print_exc(file=sys.stderr)
        return {"ok": True, "retrain_needed": bool(retrain), "signals_updated": updated,
                "changed": [f"{sec}.{k}" for sec, vals in changed.items() for k in vals],
                "settings": settings_subset(cfg)}

    # ------------------------------------------------------------ εργασίες
    @app.post("/api/jobs")
    def api_jobs_post(body: dict = Body(...)):
        typ = body.get("type")
        if typ not in JOB_TYPES:
            raise HTTPException(400, "Άγνωστος τύπος εργασίας (επιτρέπονται: refresh, backtest, demo).")
        try:
            job = jobs.submit(typ)
        except JobConflict:
            raise HTTPException(409, f"Η εργασία «{JOB_TYPES_EL[typ]}» εκτελείται ήδη.") from None
        return {"job_id": job["id"], "job": job}

    @app.get("/api/jobs")
    def api_jobs(limit: int = 20):
        return {"items": jobs.list(max(1, min(int(limit), MAX_JOBS_KEPT)))}

    @app.get("/api/jobs/{jid}")
    def api_job(jid: str):
        j = jobs.get(jid)
        if j is None:
            raise HTTPException(404, "Η εργασία δεν βρέθηκε.")
        return j

    @app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    def api_unknown(rest: str):
        raise HTTPException(404, "Άγνωστο endpoint.")

    # ------------------------------------------------------------ UI
    wd = Path(web_dir) if web_dir is not None else WEB_DIR
    if wd.is_dir() and (wd / "index.html").exists():
        app.mount("/", SPAStaticFiles(directory=str(wd), html=True), name="web")
    return app


_APP: FastAPI | None = None


def __getattr__(name: str):
    """`athex_orb.server.app` δημιουργείται μόνο όταν ζητηθεί (π.χ. από το uvicorn)."""
    global _APP
    if name == "app":
        if _APP is None:
            _APP = create_app()
        return _APP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
