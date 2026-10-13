"""Φόρτωση ρυθμίσεων από config.toml (+ προαιρετικές αλλαγές χρήστη από data/settings.json)."""
from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.toml"
DEFAULT_DATA_DIR = ROOT / "data"


@dataclass
class DataCfg:
    source: str = "yahoo"
    history_years: int = 8
    index_ticker: str = "GD.AT"


@dataclass
class BarrierCfg:
    k_tp: float = 2.0
    k_sl: float = 1.0
    horizon_days: int = 10
    atr_days: int = 14


@dataclass
class ModelCfg:
    kind: str = "logreg"
    min_train_days: int = 500
    retrain_every_days: int = 21
    train_window_days: int = 1500


@dataclass
class CostCfg:
    commission_pct: float = 0.0010
    commission_min_eur: float = 5.0
    exchange_fees_pct: float = 0.0003
    sell_tax_pct: float = 0.0
    spread_bps_base: float = 10
    impact_coef: float = 0.10


@dataclass
class PortfolioCfg:
    capital_eur: float = 10000
    position_eur_max: float = 2000
    max_positions: int = 5
    max_per_sector: int = 2
    max_adv_fraction: float = 0.01


@dataclass
class GuardCfg:
    min_adv_eur: float = 300000
    min_price_eur: float = 0.50
    max_atr_pct: float = 0.08
    min_history_days: int = 260
    crash_index_drop: float = 0.08
    min_ev_net: float = 0.003
    min_p_tp: float = 0.0


@dataclass
class BacktestCfg:
    random_runs: int = 200
    seed: int = 42


@dataclass
class ServerCfg:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class Config:
    data: DataCfg = field(default_factory=DataCfg)
    barriers: BarrierCfg = field(default_factory=BarrierCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    costs: CostCfg = field(default_factory=CostCfg)
    portfolio: PortfolioCfg = field(default_factory=PortfolioCfg)
    guard: GuardCfg = field(default_factory=GuardCfg)
    backtest: BacktestCfg = field(default_factory=BacktestCfg)
    server: ServerCfg = field(default_factory=ServerCfg)
    data_dir: Path = DEFAULT_DATA_DIR

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_dir"] = str(self.data_dir)
        return d


# Ρυθμίσεις που αλλάζει ο χρήστης από το UI: (τύπος, ελάχιστο, μέγιστο, ετικέτα). Τα όρια είναι «λογικά»,
# όχι αυστηρά οικονομικά. Ελέγχονται στο PUT /api/settings και ξανά όταν διαβάζεται το data/settings.json.
USER_SETTINGS_SPEC: dict[str, dict[str, tuple]] = {
    "portfolio": {
        "capital_eur": ("float", 100.0, 1e9, "Κεφάλαιο"),
        "position_eur_max": ("float", 50.0, 1e8, "Μέγιστο ποσό ανά θέση"),
        "max_positions": ("int", 1, 50, "Μέγιστες ταυτόχρονες θέσεις"),
        "max_per_sector": ("int", 1, 50, "Μέγιστες θέσεις ανά κλάδο"),
        "max_adv_fraction": ("float", 0.0001, 0.5, "Μέγιστο μέρος του μέσου τζίρου"),
    },
    "barriers": {
        "k_tp": ("float", 0.2, 10.0, "Στόχος (× ATR)"),
        "k_sl": ("float", 0.2, 10.0, "Stop (× ATR)"),
        "horizon_days": ("int", 1, 60, "Ορίζοντας (συνεδριάσεις)"),
        "atr_days": ("int", 2, 100, "Ημέρες ATR"),
    },
    "guard": {
        "min_adv_eur": ("float", 0.0, 1e9, "Ελάχιστος μέσος τζίρος 20 ημερών"),
        "min_price_eur": ("float", 0.0, 1e4, "Ελάχιστη τιμή μετοχής"),
        "max_atr_pct": ("float", 0.001, 1.0, "Μέγιστη μεταβλητότητα (ATR / τιμή)"),
        "min_history_days": ("int", 0, 5000, "Ελάχιστο ιστορικό"),
        "crash_index_drop": ("float", 0.005, 1.0, "Πτώση δείκτη για παύση"),
        "min_ev_net": ("float", -0.1, 0.2, "Ελάχιστο καθαρό EV"),
        "min_p_tp": ("float", 0.0, 1.0, "Ελάχιστη πιθανότητα στόχου"),
    },
    "costs": {
        "commission_pct": ("float", 0.0, 0.05, "Προμήθεια ανά πλευρά"),
        "commission_min_eur": ("float", 0.0, 1000.0, "Ελάχιστη προμήθεια"),
        "exchange_fees_pct": ("float", 0.0, 0.05, "Χρεώσεις χρηματιστηρίου"),
        "sell_tax_pct": ("float", 0.0, 0.05, "Φόρος πώλησης"),
        "spread_bps_base": ("float", 0.0, 500.0, "Βασικό μισό spread (bps)"),
        "impact_coef": ("float", 0.0, 5.0, "Συντελεστής επίδρασης στην τιμή"),
    },
    "model": {
        "kind": ("choice", ("logreg", "hgb"), None, "Τύπος μοντέλου"),
    },
}


def _convert(ftype: str, val):
    """Μετατροπή στον δηλωμένο τύπο του πεδίου· TypeError/ValueError για τιμές που δεν ταιριάζουν
    (π.χ. null, λίστα, NaN, 2.5 για ακέραιο)."""
    if ftype == "bool":
        if not isinstance(val, (bool, int)):
            raise TypeError("bool")
        return bool(val)
    if ftype in ("int", "float"):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(ftype)
        v = float(val)
        if not math.isfinite(v):
            raise ValueError("μη πεπερασμένη τιμή")
        if ftype == "int":
            if v != int(v):
                raise ValueError("όχι ακέραιος")
            return int(v)
        return v
    if ftype == "str":
        if not isinstance(val, str):
            raise TypeError("str")
        return val
    return val


def _apply(section_obj, values: dict, spec: dict | None = None) -> None:
    """Εφαρμόζει τιμές σε μια ενότητα. Κάθε κλειδί ελέγχεται χωριστά: μια λάθος τιμή αγνοείται (και κρατά
    την προηγούμενη) αντί να «ρίξει» όλη τη φόρτωση. `spec`: όρια της ενότητας (USER_SETTINGS_SPEC)."""
    valid = {f.name: f for f in fields(section_obj)}
    for key, val in values.items():
        if key not in valid:
            continue
        # κρατάμε τον δηλωμένο τύπο του πεδίου (int/float/str) — όχι τον τύπο της τρέχουσας τιμής,
        # αφού π.χ. το TOML δίνει `capital_eur = 10000` ως int ενώ το πεδίο είναι float
        ftype = valid[key].type
        ftype = ftype if isinstance(ftype, str) else getattr(ftype, "__name__", "")
        try:
            val = _convert(ftype, val)
        except (TypeError, ValueError, OverflowError):
            continue
        rule = (spec or {}).get(key)
        if rule is not None:
            kind, lo, hi, _ = rule
            if kind == "choice":
                if val not in lo:
                    continue
            elif not (lo <= val <= hi):
                continue
        setattr(section_obj, key, val)


def apply_overrides(cfg: Config, overrides: dict, validate: bool = False) -> Config:
    """Εφαρμόζει {section: {key: value}} πάνω σε ένα Config (in place) και το επιστρέφει.
    `validate=True`: οι ρυθμίσεις του USER_SETTINGS_SPEC πρέπει να είναι μέσα στα όριά τους."""
    if not isinstance(overrides, dict):
        return cfg
    for section, values in overrides.items():
        if section == "data_dir" or not isinstance(values, dict):
            continue
        obj = getattr(cfg, section, None)
        if obj is not None and hasattr(obj, "__dataclass_fields__"):
            _apply(obj, values, USER_SETTINGS_SPEC.get(section) if validate else None)
    return cfg


def settings_path(data_dir: Path) -> Path:
    return Path(data_dir) / "settings.json"


def load_config(path: Path | str | None = None, data_dir: Path | str | None = None,
                use_user_settings: bool = True) -> Config:
    cfg = Config()
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if p.exists():
        with open(p, "rb") as fh:
            apply_overrides(cfg, tomllib.load(fh))
    cfg.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    if use_user_settings:
        sp = settings_path(cfg.data_dir)
        if sp.exists():
            try:
                apply_overrides(cfg, json.loads(sp.read_text(encoding="utf-8")), validate=True)
            except (ValueError, OSError, UnicodeError, TypeError, AttributeError):
                pass               # κατεστραμμένο settings.json ⇒ οι προεπιλογές του config.toml
    return cfg


def save_user_settings(cfg: Config, overrides: dict) -> None:
    """Αποθηκεύει τις αλλαγές του χρήστη (συγχωνεύοντας με τις υπάρχουσες)."""
    sp = settings_path(cfg.data_dir)
    sp.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if sp.exists():
        try:
            current = json.loads(sp.read_text(encoding="utf-8"))
        except (ValueError, OSError, UnicodeError):
            current = {}
    if not isinstance(current, dict):
        current = {}
    for section, values in overrides.items():
        if isinstance(values, dict):
            if not isinstance(current.get(section), dict):
                current[section] = {}
            current[section].update(values)
    tmp = sp.with_name(f".{sp.name}.tmp")
    tmp.write_text(json.dumps(current, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(sp)
    apply_overrides(cfg, overrides)
