"""Exchange adapter tests: mock ``get_json`` so these run with zero
network access and verify each adapter's parsing/normalization logic
against a small hand-built payload shaped exactly like the real
Binance/OKX public endpoints.

These intentionally do NOT hit the network (this sandbox's egress
policy blocks api.binance.com/www.okx.com outright, and even where
network access is available, adapter tests should never depend on a
live third-party API being reachable or stable).
"""
from __future__ import annotations

import pandas as pd
import pytest

from accscan.exchanges import binance as binance_mod
from accscan.exchanges import okx as okx_mod
from accscan.exchanges.binance import BinanceClient
from accscan.exchanges.okx import OKXClient


# --- Binance -----------------------------------------------------------

_BINANCE_EXCHANGE_INFO = {
    "symbols": [
        {"symbol": "BTCUSDT", "status": "TRADING", "baseAsset": "BTC", "quoteAsset": "USDT"},
        {"symbol": "ETHFIUSDT", "status": "TRADING", "baseAsset": "ETHFI", "quoteAsset": "USDT"},
        {"symbol": "ETHBUSD", "status": "TRADING", "baseAsset": "ETH", "quoteAsset": "BUSD"},
        {"symbol": "DELISTEDUSDT", "status": "BREAK", "baseAsset": "DELISTED", "quoteAsset": "USDT"},
    ]
}

_BINANCE_TICKERS_24H = [
    {"symbol": "BTCUSDT", "lastPrice": "65000.0", "priceChangePercent": "1.5", "quoteVolume": "900000000.0"},
    {"symbol": "ETHFIUSDT", "lastPrice": "0.69", "priceChangePercent": "-0.8", "quoteVolume": "36000000.0"},
    {"symbol": "ETHBUSD", "lastPrice": "3000.0", "priceChangePercent": "2.0", "quoteVolume": "5000000.0"},
    {"symbol": "DELISTEDUSDT", "lastPrice": "1.0", "priceChangePercent": "0.0", "quoteVolume": "1000.0"},
]


def _binance_kline_row(close_time_ms, o, h, l, c, v):
    return [
        close_time_ms - 3_600_000, str(o), str(h), str(l), str(c),
        str(v), close_time_ms, "0", 100, "0", "0", "0",
    ]


def test_binance_get_tickers_24h_parses_and_filters_by_quote(monkeypatch):
    calls = []

    def fake_get_json(url, params=None, **kwargs):
        calls.append(url)
        if url.endswith("/api/v3/exchangeInfo"):
            return _BINANCE_EXCHANGE_INFO
        if url.endswith("/api/v3/ticker/24hr"):
            return _BINANCE_TICKERS_24H
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(binance_mod, "get_json", fake_get_json)
    client = BinanceClient()
    tickers = client.get_tickers_24h(quote_asset="USDT")

    symbols = {t.symbol for t in tickers}
    # ETHBUSD is quoted in BUSD, not USDT -> excluded. DELISTEDUSDT is not
    # TRADING in exchangeInfo -> no base/quote mapping -> excluded too.
    assert symbols == {"BTCUSDT", "ETHFIUSDT"}

    ethfi = next(t for t in tickers if t.symbol == "ETHFIUSDT")
    assert ethfi.base_asset == "ETHFI"
    assert ethfi.quote_asset == "USDT"
    assert ethfi.last_price == pytest.approx(0.69)
    assert ethfi.price_change_pct_24h == pytest.approx(-0.8)
    assert ethfi.quote_volume_24h == pytest.approx(36_000_000.0)


def test_binance_exchange_info_is_cached_across_calls(monkeypatch):
    info_calls = []

    def fake_get_json(url, params=None, **kwargs):
        if url.endswith("/api/v3/exchangeInfo"):
            info_calls.append(1)
            return _BINANCE_EXCHANGE_INFO
        if url.endswith("/api/v3/ticker/24hr"):
            return _BINANCE_TICKERS_24H
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(binance_mod, "get_json", fake_get_json)
    client = BinanceClient()
    client.get_tickers_24h()
    client.get_tickers_24h()
    assert len(info_calls) == 1


def test_binance_get_klines_parses_ohlcv_and_sets_close_time_index(monkeypatch):
    rows = [
        _binance_kline_row(1_700_000_000_000 + i * 3_600_000, 10 + i, 11 + i, 9 + i, 10.5 + i, 1000 + i)
        for i in range(5)
    ]

    def fake_get_json(url, params=None, **kwargs):
        assert params["symbol"] == "ETHFIUSDT"
        assert params["interval"] == "1h"
        return rows

    monkeypatch.setattr(binance_mod, "get_json", fake_get_json)
    client = BinanceClient()
    df = client.get_klines("ETHFIUSDT", "1h", limit=5)

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 5
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df["close"].iloc[0] == pytest.approx(10.5)
    assert df["volume"].iloc[-1] == pytest.approx(1004.0)


def test_binance_get_klines_raises_on_empty_response(monkeypatch):
    monkeypatch.setattr(binance_mod, "get_json", lambda *a, **k: [])
    client = BinanceClient()
    with pytest.raises(RuntimeError):
        client.get_klines("ETHFIUSDT", "1h")


def test_binance_get_klines_rejects_unknown_interval():
    client = BinanceClient()
    with pytest.raises(ValueError):
        client.get_klines("ETHFIUSDT", "3h")


def test_binance_normalize_symbol():
    client = BinanceClient()
    assert client.normalize_symbol("ethfi", "usdt") == "ETHFIUSDT"


def test_binance_market_wide_snapshot_is_volume_weighted(monkeypatch):
    def fake_get_json(url, params=None, **kwargs):
        if url.endswith("/api/v3/exchangeInfo"):
            return _BINANCE_EXCHANGE_INFO
        if url.endswith("/api/v3/ticker/24hr"):
            return _BINANCE_TICKERS_24H
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(binance_mod, "get_json", fake_get_json)
    client = BinanceClient()
    snap = client.get_market_wide_snapshot(quote_asset="USDT")

    total = 900_000_000.0 + 36_000_000.0
    assert snap.exchange == "binance"
    assert snap.pair_count == 2
    # BTC holds >50% of volume, so the volume-weighted median is BTC's move.
    assert snap.volume_weighted_change_pct_24h == pytest.approx(1.5)
    assert snap.btc_quote_volume_share_pct == pytest.approx(900_000_000.0 / total * 100.0)


# --- OKX -----------------------------------------------------------------

_OKX_INSTRUMENTS = {
    "code": "0",
    "msg": "",
    "data": [
        {"instId": "BTC-USDT", "baseCcy": "BTC", "quoteCcy": "USDT", "state": "live"},
        {"instId": "ETHFI-USDT", "baseCcy": "ETHFI", "quoteCcy": "USDT", "state": "live"},
        {"instId": "ETH-USDC", "baseCcy": "ETH", "quoteCcy": "USDC", "state": "live"},
        {"instId": "OLD-USDT", "baseCcy": "OLD", "quoteCcy": "USDT", "state": "suspend"},
    ],
}

_OKX_TICKERS = {
    "code": "0",
    "msg": "",
    "data": [
        {"instId": "BTC-USDT", "last": "65000.0", "open24h": "64000.0", "volCcy24h": "800000000.0"},
        {"instId": "ETHFI-USDT", "last": "0.69", "open24h": "0.70", "volCcy24h": "20000000.0"},
        {"instId": "ETH-USDC", "last": "3000.0", "open24h": "2950.0", "volCcy24h": "4000000.0"},
        {"instId": "OLD-USDT", "last": "1.0", "open24h": "1.0", "volCcy24h": "500.0"},
    ],
}


def test_okx_get_tickers_24h_parses_and_filters_by_quote(monkeypatch):
    def fake_get_json(url, params=None, **kwargs):
        if url.endswith("/api/v5/public/instruments"):
            return _OKX_INSTRUMENTS
        if url.endswith("/api/v5/market/tickers"):
            return _OKX_TICKERS
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(okx_mod, "get_json", fake_get_json)
    client = OKXClient()
    tickers = client.get_tickers_24h(quote_asset="USDT")

    symbols = {t.symbol for t in tickers}
    # ETH-USDC is quoted in USDC, not USDT. OLD-USDT is "suspend" in the
    # instruments list -> no mapping -> excluded.
    assert symbols == {"BTC-USDT", "ETHFI-USDT"}

    ethfi = next(t for t in tickers if t.symbol == "ETHFI-USDT")
    assert ethfi.base_asset == "ETHFI"
    assert ethfi.last_price == pytest.approx(0.69)
    # OKX ticker rows don't carry a ready-made % change -- computed from
    # last vs open24h: (0.69 - 0.70) / 0.70 * 100
    assert ethfi.price_change_pct_24h == pytest.approx((0.69 - 0.70) / 0.70 * 100.0)
    assert ethfi.quote_volume_24h == pytest.approx(20_000_000.0)


def test_okx_api_error_code_raises(monkeypatch):
    def fake_get_json(url, params=None, **kwargs):
        return {"code": "50011", "msg": "rate limit exceeded", "data": []}

    monkeypatch.setattr(okx_mod, "get_json", fake_get_json)
    client = OKXClient()
    with pytest.raises(RuntimeError, match="rate limit exceeded"):
        client.get_tickers_24h()


def test_okx_get_klines_reverses_newest_first_order_and_renames_volume(monkeypatch):
    # OKX returns candles newest-first; build three bars 1h apart where
    # that ordering is the only way to tell them apart (ascending ts).
    newest_first = {
        "code": "0",
        "msg": "",
        "data": [
            ["1700007200000", "12", "13", "11", "12.5", "0", "1200", "0", "1"],
            ["1700003600000", "11", "12", "10", "11.5", "0", "1100", "0", "1"],
            ["1700000000000", "10", "11", "9", "10.5", "0", "1000", "0", "1"],
        ],
    }

    def fake_get_json(url, params=None, **kwargs):
        assert params["instId"] == "ETHFI-USDT"
        assert params["bar"] == "1H"
        return newest_first

    monkeypatch.setattr(okx_mod, "get_json", fake_get_json)
    client = OKXClient()
    df = client.get_klines("ETHFI-USDT", "1h")

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 3
    # Chronological order after reversal: oldest close (10.5) first.
    assert df["close"].iloc[0] == pytest.approx(10.5)
    assert df["close"].iloc[-1] == pytest.approx(12.5)
    assert df.index.is_monotonic_increasing


def test_okx_normalize_symbol():
    client = OKXClient()
    assert client.normalize_symbol("ethfi", "usdt") == "ETHFI-USDT"


# --- Closed-candle handling and volume units ------------------------------

def test_binance_get_klines_drops_the_still_forming_candle(monkeypatch):
    base = 1_700_000_000_000
    rows = [_binance_kline_row(base + i * 3_600_000, 10, 11, 9, 10 + i, 100) for i in range(4)]
    # "now" sits inside the last bar -> its close_time is in the future.
    monkeypatch.setattr(binance_mod, "_now_ms", lambda: base + 2 * 3_600_000 + 1)
    monkeypatch.setattr(binance_mod, "get_json", lambda *a, **k: rows)
    df = BinanceClient().get_klines("ETHFIUSDT", "1h")
    assert len(df) == 3
    assert df["close"].iloc[-1] == pytest.approx(12.0)


def test_okx_get_klines_drops_unconfirmed_candle_and_uses_base_volume(monkeypatch):
    payload = {
        "code": "0", "msg": "",
        "data": [
            ["1700007200000", "12", "13", "11", "12.5", "7", "1200", "0", "0"],  # forming
            ["1700003600000", "11", "12", "10", "11.5", "6", "1100", "0", "1"],
            ["1700000000000", "10", "11", "9", "10.5", "5", "1000", "0", "1"],
        ],
    }
    monkeypatch.setattr(okx_mod, "get_json", lambda *a, **k: payload)
    df = OKXClient().get_klines("ETHFI-USDT", "1h")
    assert len(df) == 2
    assert df["close"].iloc[-1] == pytest.approx(11.5)
    assert list(df["volume"]) == [pytest.approx(5.0), pytest.approx(6.0)]


def test_okx_daily_bars_are_utc_aligned(monkeypatch):
    seen = {}

    def fake_get_json(url, params=None, **kwargs):
        seen.update(params)
        return {"code": "0", "msg": "", "data": [["1700000000000", "1", "1", "1", "1", "1", "1", "0", "1"]]}

    monkeypatch.setattr(okx_mod, "get_json", fake_get_json)
    OKXClient().get_klines("ETHFI-USDT", "1D")
    assert seen["bar"] == "1Dutc"


def test_market_wide_snapshot_excludes_stablecoin_pairs(monkeypatch):
    info = {"symbols": _BINANCE_EXCHANGE_INFO["symbols"] + [
        {"symbol": "USDCUSDT", "status": "TRADING", "baseAsset": "USDC", "quoteAsset": "USDT"},
    ]}
    tickers = _BINANCE_TICKERS_24H + [
        {"symbol": "USDCUSDT", "lastPrice": "1.0", "priceChangePercent": "0.0", "quoteVolume": "5000000000.0"},
    ]

    def fake_get_json(url, params=None, **kwargs):
        return info if url.endswith("/api/v3/exchangeInfo") else tickers

    monkeypatch.setattr(binance_mod, "get_json", fake_get_json)
    snap = BinanceClient().get_market_wide_snapshot("USDT", exclude_bases=["USDC"])
    total = 900_000_000.0 + 36_000_000.0
    assert snap.pair_count == 2
    assert snap.volume_weighted_change_pct_24h == pytest.approx(1.5)
