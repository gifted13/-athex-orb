"""Φόρτωση ρυθμίσεων από config.toml (+ προαιρετικές αλλαγές χρήστη από data/settings.json)."""
from __future__ import annotations

import json
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


def _apply(section_obj, values: dict) -> None:
    valid = {f.name: f for f in fields(section_obj)}
    for key, val in values.items():
        if key not in valid:
            continue
        current = getattr(section_obj, key)
        # κρατάμε τον τύπο του default (int/float/str)
        if isinstance(current, bool):
            val = bool(val)
        elif isinstance(current, int) and not isinstance(val, bool):
            val = int(val)
        elif isinstance(current, float):
            val = float(val)
        setattr(section_obj, key, val)


def apply_overrides(cfg: Config, overrides: dict) -> Config:
    """Εφαρμόζει {section: {key: value}} πάνω σε ένα Config (in place) και το επιστρέφει."""
    for section, values in (overrides or {}).items():
        if section == "data_dir" or not isinstance(values, dict):
            continue
        obj = getattr(cfg, section, None)
        if obj is not None:
            _apply(obj, values)
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
                apply_overrides(cfg, json.loads(sp.read_text(encoding="utf-8")))
            except (ValueError, OSError):
                pass
    return cfg


def save_user_settings(cfg: Config, overrides: dict) -> None:
    """Αποθηκεύει τις αλλαγές του χρήστη (συγχωνεύοντας με τις υπάρχουσες)."""
    sp = settings_path(cfg.data_dir)
    sp.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if sp.exists():
        try:
            current = json.loads(sp.read_text(encoding="utf-8"))
        except ValueError:
            current = {}
    for section, values in overrides.items():
        if isinstance(values, dict):
            current.setdefault(section, {}).update(values)
    sp.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    apply_overrides(cfg, overrides)
