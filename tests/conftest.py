"""Κοινά fixtures για τα tests.

* `cfg`             — ρυθμίσεις με data_dir = προσωρινός φάκελος (χωρίς settings.json χρήστη).
* `session_cfg`     — το ίδιο, ένα για όλη τη συνεδρία (για ακριβά fixtures).
* `small_universe`  — 8 μετοχές από το DEFAULT_UNIVERSE (3 τράπεζες, 2 ενέργεια, 3 άλλοι κλάδοι).
* `synth`           — (prices, index_df) συνθετικά, ~4 χρόνια (2019–2022), χωρίς σήμα· `synth_range` = (start, end, seed).
* `synth_planted`   — όπως το `synth` αλλά με planted_signal = 1 (μαθήσιμο σήμα επαναφοράς στο ret_5).
* `panel`, `labels_df` — panel χαρακτηριστικών και ετικέτες triple barrier για το `synth`.

Το δίκτυο δεν χρησιμοποιείται ποτέ: ένα autouse fixture αντικαθιστά το yfinance με «μπλοκαρισμένο» module.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from athex_orb import features, labels, synthetic  # noqa: E402
from athex_orb.config import load_config  # noqa: E402
from athex_orb.universe import DEFAULT_UNIVERSE, Stock  # noqa: E402

SMALL_TICKERS = ["ETE.AT", "EUROB.AT", "TPEIR.AT", "PPC.AT", "MOH.AT", "OPAP.AT", "HTO.AT", "BELA.AT"]
SYNTH_START = "2019-01-02"
SYNTH_END = "2022-12-30"
SYNTH_SEED = 11


class _BlockedYFinance:
    """Ψεύτικο yfinance: κάθε χρήση αποτυγχάνει (τα tests δεν έχουν πρόσβαση στο δίκτυο)."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise RuntimeError("Το yfinance είναι απενεργοποιημένο στα tests")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", _BlockedYFinance())


def _copy(stocks):
    return [Stock(**s.to_dict()) for s in stocks]


@pytest.fixture
def cfg(tmp_path):
    return load_config(data_dir=tmp_path, use_user_settings=False)


@pytest.fixture(scope="session")
def session_cfg(tmp_path_factory):
    return load_config(data_dir=tmp_path_factory.mktemp("data"), use_user_settings=False)


@pytest.fixture(scope="session")
def small_universe():
    return _copy([s for s in DEFAULT_UNIVERSE if s.ticker in SMALL_TICKERS])


@pytest.fixture(scope="session")
def full_universe():
    return _copy(DEFAULT_UNIVERSE)


@pytest.fixture(scope="session")
def synth_range():
    """(start, end, seed) των fixtures `synth` / `synth_planted`."""
    return SYNTH_START, SYNTH_END, SYNTH_SEED


@pytest.fixture(scope="session")
def synth(small_universe):
    return synthetic.generate_universe(small_universe, start=SYNTH_START, end=SYNTH_END, seed=SYNTH_SEED)


@pytest.fixture(scope="session")
def synth_planted(small_universe):
    return synthetic.generate_universe(small_universe, start=SYNTH_START, end=SYNTH_END, seed=SYNTH_SEED,
                                       planted_signal=1.0)


@pytest.fixture(scope="session")
def panel(synth, small_universe, session_cfg):
    prices, index_df = synth
    return features.build_panel(prices, index_df, small_universe, session_cfg)


@pytest.fixture(scope="session")
def labels_df(synth, panel, session_cfg):
    return labels.triple_barrier(synth[0], panel, session_cfg)
