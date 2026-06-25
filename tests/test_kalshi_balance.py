from __future__ import annotations

from kalshi_bridge import KalshiClient


def test_get_balance_parses_top_level_cents(monkeypatch):
    client = KalshiClient.__new__(KalshiClient)

    def fake_get(path, params=None):
        assert path == "/portfolio/balance"
        return {"balance": 16000, "portfolio_value": 20000}

    monkeypatch.setattr(client, "_get", fake_get)
    assert client.get_balance() == 160.0


def test_get_balance_parses_balance_dollars(monkeypatch):
    client = KalshiClient.__new__(KalshiClient)
    monkeypatch.setattr(client, "_get", lambda path, params=None: {"balance_dollars": "160.50", "balance": 16050})
    assert client.get_balance() == 160.5


def test_get_balance_legacy_nested(monkeypatch):
    client = KalshiClient.__new__(KalshiClient)
    monkeypatch.setattr(
        client,
        "_get",
        lambda path, params=None: {"balance": {"available_balance": 12345}},
    )
    assert client.get_balance() == 123.45
