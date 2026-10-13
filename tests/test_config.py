"""Φόρτωση ρυθμίσεων: ένα κατεστραμμένο data/settings.json δεν «ρίχνει» ποτέ τον διακομιστή ή το CLI."""
from __future__ import annotations

import json

import pytest

from athex_orb import cli
from athex_orb.config import Config, load_config, save_user_settings, settings_path


@pytest.mark.parametrize("content", [
    '{"portfolio": {"capital_eur": null}}',
    "[1, 2]",
    '{"portfolio": {"max_positions": [3]}}',
    '{"guard": {"min_ev_net": NaN}}',
    '{"portfolio": {"max_positions": -3}, "barriers": {"horizon_days": 0}}',
    '{"model": {"kind": "xyz"}}',
    '{"portfolio": 5}',
    "{σκουπίδια",
    '{"barriers": {"horizon_days": 2.5}}',
])
def test_corrupt_user_settings_fall_back_to_defaults(tmp_path, content):
    settings_path(tmp_path).write_text(content, encoding="utf-8")
    cfg = load_config(data_dir=tmp_path)
    d = Config()
    assert cfg.portfolio.capital_eur == d.portfolio.capital_eur
    assert cfg.portfolio.max_positions == d.portfolio.max_positions
    assert cfg.barriers.horizon_days == d.barriers.horizon_days
    assert cfg.guard.min_ev_net == d.guard.min_ev_net
    assert cfg.model.kind == d.model.kind
    json.dumps(cfg.to_dict(), allow_nan=False)         # GET /api/settings: πάντα έγκυρο JSON


def test_good_keys_survive_next_to_bad_ones(tmp_path):
    settings_path(tmp_path).write_text('{"portfolio": {"capital_eur": 25000, "max_positions": null}, '
                                       '"guard": {"min_ev_net": 0.01}}', encoding="utf-8")
    cfg = load_config(data_dir=tmp_path)
    assert cfg.portfolio.capital_eur == 25000.0 and isinstance(cfg.portfolio.capital_eur, float)
    assert cfg.portfolio.max_positions == Config().portfolio.max_positions
    assert cfg.guard.min_ev_net == 0.01


def test_save_over_corrupt_file(tmp_path):
    settings_path(tmp_path).write_text("[1, 2]", encoding="utf-8")
    cfg = load_config(data_dir=tmp_path)
    save_user_settings(cfg, {"portfolio": {"capital_eur": 5000.0}})
    assert json.loads(settings_path(tmp_path).read_text(encoding="utf-8")) == {"portfolio": {"capital_eur": 5000.0}}
    assert load_config(data_dir=tmp_path).portfolio.capital_eur == 5000.0


def test_cli_signals_survives_corrupt_settings(tmp_path, capsys):
    settings_path(tmp_path).write_text('{"portfolio": {"capital_eur": null}}', encoding="utf-8")
    assert cli.main(["--data-dir", str(tmp_path), "signals"]) == 1      # «δεν υπάρχουν σήματα», όχι traceback
    assert "Δεν υπάρχουν σήματα" in capsys.readouterr().out


def test_serve_reports_busy_port_in_greek(tmp_path, capsys):
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        assert cli.main(["--data-dir", str(tmp_path), "serve", "--port", str(port)]) == 1
    err = capsys.readouterr()
    assert "χρησιμοποιείται ήδη" in err.err and "Άνοιξε" not in err.out
