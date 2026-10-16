"""Step 1 (market-wide regime timing) tests.

A tiny fake ExchangeClient stands in for Binance/OKX here -- these
tests are about the regime-classification arithmetic in
market_timing.py, not about any real exchange's wire format (that is
covered in test_exchanges.py).
"""
from __future__ import annotations

from typing import List

import pandas as pd
import pytest

from accscan.config import ScreenConfig
from accscan.exchanges.base import ExchangeClient, Ticker24h
from accscan.market_timing import MarketRegime, assess_market_timing


class _FakeClient(ExchangeClient):
    name = "fake"

    def __init__(self, tickers: List[Ticker24h]):
        self._tickers = tickers

    def get_tickers_24h(self, quote_asset: str = "USDT") -> List[Ticker24h]:
        return [t for t in self._tickers if t.quote_asset == quote_asset]

    def get_klines(self, symbol: str, interval: str, limit: int = 240) -> pd.DataFrame:
        raise NotImplementedError

    def normalize_symbol(self, base_asset: str, quote_asset: str = "USDT") -> str:
        return f"{base_asset}{quote_asset}"


def _ticker(symbol, base, change_pct, volume, quote="USDT"):
    return Ticker24h(
        symbol=symbol, base_asset=base, quote_asset=quote,
        last_price=1.0, price_change_pct_24h=change_pct, quote_volume_24h=volume,
    )


def test_sharp_trend_when_weighted_move_exceeds_threshold():
    cfg = ScreenConfig()
    client = _FakeClient([
        _ticker("BTCUSDT", "BTC", change_pct=8.0, volume=1_000_000_000.0),
        _ticker("ETHUSDT", "ETH", change_pct=6.0, volume=400_000_000.0),
    ])
    result = assess_market_timing(client, cfg)
    assert result.regime is MarketRegime.SHARP_TREND
    assert result.strict_mode is True
    assert result.snapshot.volume_weighted_change_pct_24h > cfg.sharp_market_move_pct


def test_active_range_when_move_is_moderate():
    cfg = ScreenConfig()
    client = _FakeClient([
        _ticker("BTCUSDT", "BTC", change_pct=1.0, volume=1_000_000_000.0),
        _ticker("ETHFIUSDT", "ETHFI", change_pct=-1.5, volume=30_000_000.0),
    ])
    result = assess_market_timing(client, cfg)
    assert result.regime is MarketRegime.ACTIVE_RANGE
    assert result.strict_mode is False
    assert abs(result.snapshot.volume_weighted_change_pct_24h) <= cfg.sharp_market_move_pct


def test_btc_dominance_share_is_btc_volume_over_total():
    cfg = ScreenConfig()
    client = _FakeClient([
        _ticker("BTCUSDT", "BTC", change_pct=0.5, volume=750_000_000.0),
        _ticker("ETHFIUSDT", "ETHFI", change_pct=0.5, volume=250_000_000.0),
    ])
    result = assess_market_timing(client, cfg)
    assert result.snapshot.btc_quote_volume_share_pct == pytest.approx(75.0)


def test_raises_when_exchange_returns_no_tickers():
    cfg = ScreenConfig()
    client = _FakeClient([])
    with pytest.raises(RuntimeError):
        assess_market_timing(client, cfg)


def test_quiet_regime_when_market_barely_moves():
    cfg = ScreenConfig()
    client = _FakeClient([
        _ticker("BTCUSDT", "BTC", change_pct=0.2, volume=1_000_000_000.0),
        _ticker("ETHFIUSDT", "ETHFI", change_pct=-0.1, volume=30_000_000.0),
    ])
    result = assess_market_timing(client, cfg)
    assert result.regime is MarketRegime.QUIET
    assert result.strict_mode is False


def test_stablecoin_volume_no_longer_masks_a_sharp_market():
    # Alts +6% but a giant ~0% USDC pair: previously diluted to ~1.6%.
    cfg = ScreenConfig()
    client = _FakeClient([
        _ticker("BTCUSDT", "BTC", change_pct=6.0, volume=1_000_000_000.0),
        _ticker("USDCUSDT", "USDC", change_pct=0.0, volume=3_000_000_000.0),
    ])
    result = assess_market_timing(client, cfg)
    assert result.regime is MarketRegime.SHARP_TREND


def test_config_only_accepts_usdt_quote():
    assert ScreenConfig().quote_asset == "USDT"
    assert ScreenConfig(quote_asset="usdt").quote_asset == "USDT"
    with pytest.raises(ValueError):
        ScreenConfig(quote_asset="USDC")


def test_one_pumping_high_volume_pair_does_not_fake_a_sharp_market():
    # Majors flat-to-down, one pair +80% on big volume. The old weighted
    # mean read ~+15% (SHARP_TREND); the weighted median stays near flat.
    cfg = ScreenConfig()
    client = _FakeClient([
        _ticker("BTCUSDT", "BTC", change_pct=-0.9, volume=1_000_000_000.0),
        _ticker("ETHUSDT", "ETH", change_pct=0.2, volume=600_000_000.0),
        _ticker("SOLUSDT", "SOL", change_pct=-1.2, volume=300_000_000.0),
        _ticker("PUMPUSDT", "PUMP", change_pct=80.0, volume=400_000_000.0),
    ])
    result = assess_market_timing(client, cfg)
    assert result.regime is not MarketRegime.SHARP_TREND
    assert abs(result.snapshot.volume_weighted_change_pct_24h) < 1.0
