"""Step 2 (universe + accumulation shortlist) tests.

A tiny fake ExchangeClient stands in for Binance/OKX (wire-format
parsing is covered in test_exchanges.py) so these tests focus purely
on the screener's own filtering/ranking logic: liquidity and
major/stable exclusion, the universe size cap, honest market-cap
enforcement, and the RSI/performance shortlist band.
"""
from __future__ import annotations

import math
from typing import Dict, List

import pandas as pd
import pytest

from accscan.config import ScreenConfig
from accscan.exchanges.base import ExchangeClient, Ticker24h
from accscan.market_cap import MarketCapProvider
from accscan.screener import (
    Candidate,
    _is_probably_pegged,
    build_universe,
    score_candidates,
    shortlist_accumulation_candidates,
)


def _flat_df(n=100, price=1.0):
    idx = pd.date_range("2026-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame(
        {"open": [price] * n, "high": [price] * n, "low": [price] * n,
         "close": [price] * n, "volume": [1000.0] * n},
        index=idx,
    )


def _oscillating_df(n=100, mid=50.0, amplitude=3.0):
    idx = pd.date_range("2026-01-01", periods=n, freq="1D", tz="UTC")
    closes = [mid + amplitude * math.sin(i / 6.0) for i in range(n)]
    return pd.DataFrame(
        {"open": closes, "high": [c + 0.2 for c in closes], "low": [c - 0.2 for c in closes],
         "close": closes, "volume": [1000.0] * n},
        index=idx,
    )


class _FakeClient(ExchangeClient):
    name = "fake"

    def __init__(self, tickers: List[Ticker24h], klines: Dict[str, pd.DataFrame]):
        self._tickers = tickers
        self._klines = klines

    def get_tickers_24h(self, quote_asset: str = "USDT") -> List[Ticker24h]:
        return [t for t in self._tickers if t.quote_asset == quote_asset]

    def get_klines(self, symbol: str, interval: str, limit: int = 240) -> pd.DataFrame:
        if symbol not in self._klines:
            raise RuntimeError(f"no fixture klines for {symbol}")
        value = self._klines[symbol]
        if isinstance(value, Exception):
            raise value
        return value

    def normalize_symbol(self, base_asset: str, quote_asset: str = "USDT") -> str:
        return f"{base_asset}{quote_asset}"


class _FakeMarketCapProvider(MarketCapProvider):
    def __init__(self, caps: Dict[str, float]):
        self._caps = caps

    def get_market_caps_usd(self, base_assets):
        wanted = {a.upper() for a in base_assets}
        return {b.upper(): v for b, v in self._caps.items() if b.upper() in wanted}


def _ticker(symbol, base, volume, quote="USDT", change=0.0, price=1.0):
    return Ticker24h(
        symbol=symbol, base_asset=base, quote_asset=quote,
        last_price=price, price_change_pct_24h=change, quote_volume_24h=volume,
    )


# --- _is_probably_pegged -------------------------------------------------

def test_is_probably_pegged_true_for_flat_price():
    cfg = ScreenConfig()
    assert _is_probably_pegged(_flat_df(), cfg)


def test_is_probably_pegged_false_for_normal_swing():
    cfg = ScreenConfig()
    assert not _is_probably_pegged(_oscillating_df(), cfg)


# --- build_universe -------------------------------------------------------

def test_build_universe_filters_by_volume_quote_and_major_exclusion():
    cfg = ScreenConfig()
    tickers = [
        _ticker("BTCUSDT", "BTC", volume=2_000_000_000.0),       # excluded: major
        _ticker("USDCUSDT", "USDC", volume=500_000_000.0),       # excluded: known stable
        _ticker("ETHFIUSDT", "ETHFI", volume=36_000_000.0),      # kept
        _ticker("LOWVOLUSDT", "LOWVOL", volume=1_000_000.0),     # excluded: below volume floor
        _ticker("OTHERUSDC", "OTHER", volume=100_000_000.0, quote="USDC"),  # excluded: wrong quote
    ]
    client = _FakeClient(tickers, klines={})
    universe = build_universe(client, cfg)
    assert {t.symbol for t in universe} == {"ETHFIUSDT"}


def test_build_universe_sorts_by_volume_descending():
    cfg = ScreenConfig()
    tickers = [
        _ticker("AAAUSDT", "AAA", volume=20_000_000.0),
        _ticker("BBBUSDT", "BBB", volume=80_000_000.0),
        _ticker("CCCUSDT", "CCC", volume=50_000_000.0),
    ]
    client = _FakeClient(tickers, klines={})
    universe = build_universe(client, cfg)
    assert [t.symbol for t in universe] == ["BBBUSDT", "CCCUSDT", "AAAUSDT"]


def test_build_universe_caps_at_max_universe_size_keeping_top_by_volume():
    cfg = ScreenConfig(max_universe_size=2, min_24h_quote_volume_usd=0.0)
    tickers = [
        _ticker("AUSDT", "A", volume=10.0),
        _ticker("BUSDT", "B", volume=30.0),
        _ticker("CUSDT", "C", volume=20.0),
    ]
    client = _FakeClient(tickers, klines={})
    universe = build_universe(client, cfg)
    assert len(universe) == 2
    assert [t.symbol for t in universe] == ["BUSDT", "CUSDT"]


# --- score_candidates -------------------------------------------------------

def test_score_candidates_skips_pegged_and_failed_fetches_but_keeps_normal():
    cfg = ScreenConfig()
    tickers = [
        _ticker("ETHFIUSDT", "ETHFI", volume=36_000_000.0),
        _ticker("PEGGEDUSDT", "PEGGED", volume=40_000_000.0),
        _ticker("BROKENUSDT", "BROKEN", volume=50_000_000.0),
    ]
    client = _FakeClient(
        tickers,
        klines={
            "ETHFIUSDT": _oscillating_df(),
            "PEGGEDUSDT": _flat_df(),
            "BROKENUSDT": RuntimeError("connection reset"),
        },
    )
    universe = [t for t in tickers]  # bypass build_universe filters for this test
    scored = score_candidates(client, universe, cfg)
    assert {c.symbol for c in scored} == {"ETHFIUSDT"}


def test_score_candidates_enforces_known_market_cap_floor_only_when_known():
    cfg = ScreenConfig()
    tickers = [
        _ticker("ETHFIUSDT", "ETHFI", volume=36_000_000.0),
        _ticker("SMALLCAPUSDT", "SMALLCAP", volume=36_000_000.0),
        _ticker("UNKNOWNCAPUSDT", "UNKNOWNCAP", volume=36_000_000.0),
    ]
    client = _FakeClient(
        tickers,
        klines={
            "ETHFIUSDT": _oscillating_df(),
            "SMALLCAPUSDT": _oscillating_df(mid=20.0),
            "UNKNOWNCAPUSDT": _oscillating_df(mid=80.0),
        },
    )
    provider = _FakeMarketCapProvider({
        "ETHFI": cfg.min_market_cap_usd * 2,   # well above floor -> kept
        "SMALLCAP": cfg.min_market_cap_usd / 2,  # below floor -> dropped
        # UNKNOWNCAP deliberately absent -> unknown cap, not enforced -> kept
    })
    scored = score_candidates(client, tickers, cfg, market_cap_provider=provider)
    symbols = {c.symbol for c in scored}
    assert symbols == {"ETHFIUSDT", "UNKNOWNCAPUSDT"}

    ethfi = next(c for c in scored if c.symbol == "ETHFIUSDT")
    assert ethfi.market_cap_usd == pytest.approx(cfg.min_market_cap_usd * 2)
    unknown = next(c for c in scored if c.symbol == "UNKNOWNCAPUSDT")
    assert unknown.market_cap_usd is None


# --- shortlist_accumulation_candidates --------------------------------------

def _candidate(symbol, rsi, perf):
    return Candidate(
        symbol=symbol, base_asset=symbol.replace("USDT", ""), quote_asset="USDT",
        last_price=1.0, quote_volume_24h=50_000_000.0, market_cap_usd=None,
        rsi14_1d=rsi, perf_30d_pct=perf, price_range_90bar_pct=30.0,
    )


def test_shortlist_keeps_only_candidates_in_rsi_and_perf_band():
    cfg = ScreenConfig()
    scored = [
        _candidate("INBANDUSDT", rsi=52.0, perf=2.0),      # in band
        _candidate("HOTRSIUSDT", rsi=75.0, perf=2.0),       # RSI out of band
        _candidate("BIGMOVEUSDT", rsi=52.0, perf=25.0),     # perf out of band
    ]
    shortlist = shortlist_accumulation_candidates(scored, cfg)
    assert [c.symbol for c in shortlist] == ["INBANDUSDT"]


def test_shortlist_ranks_flattest_and_most_neutral_first():
    cfg = ScreenConfig()
    mid_rsi = sum(cfg.rsi_band) / 2.0
    scored = [
        _candidate("NOISYUSDT", rsi=mid_rsi + 5.0, perf=10.0),
        _candidate("FLATUSDT", rsi=mid_rsi, perf=0.5),
    ]
    shortlist = shortlist_accumulation_candidates(scored, cfg)
    assert [c.symbol for c in shortlist] == ["FLATUSDT", "NOISYUSDT"]


def test_shortlist_respects_shortlist_max_cap():
    cfg = ScreenConfig(shortlist_max=2)
    scored = [_candidate(f"SYM{i}USDT", rsi=50.0, perf=float(i)) for i in range(5)]
    shortlist = shortlist_accumulation_candidates(scored, cfg)
    assert len(shortlist) == 2


def test_shortlist_strict_mode_uses_tighter_rsi_band():
    cfg = ScreenConfig()
    # 58 sits inside the normal band (45-60) but outside the strict band
    # (48-56) used when the market is in a sharp trend.
    scored = [_candidate("BORDERLINEUSDT", rsi=58.0, perf=1.0)]
    assert len(shortlist_accumulation_candidates(scored, cfg, strict_mode=False)) == 1
    assert len(shortlist_accumulation_candidates(scored, cfg, strict_mode=True)) == 0
