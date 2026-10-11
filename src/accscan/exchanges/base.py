"""Common interface every exchange adapter implements.

Keeping this tiny and exchange-agnostic is what lets ``screener.py``,
``multi_timeframe.py`` etc. run unmodified against either Binance or
OKX (or a third exchange added later): they only ever call these four
methods, never touch a raw REST response.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd


@dataclass(frozen=True)
class Ticker24h:
    """One symbol's rolling 24h snapshot, normalized across exchanges."""

    symbol: str          # exchange-native symbol, e.g. "BTCUSDT" / "BTC-USDT"
    base_asset: str      # "BTC"
    quote_asset: str     # "USDT"
    last_price: float
    price_change_pct_24h: float
    quote_volume_24h: float   # in quote-asset units (~USD for *USDT pairs)


class ExchangeClient(ABC):
    """Abstract public-market-data client. No API key required."""

    name: str

    @abstractmethod
    def get_tickers_24h(self, quote_asset: str = "USDT") -> List[Ticker24h]:
        """Return a 24h snapshot for every symbol quoted in ``quote_asset``."""

    @abstractmethod
    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 240,
    ) -> pd.DataFrame:
        """Return OHLCV bars as a DataFrame indexed by UTC close time.

        Columns: open, high, low, close, volume (all float64).
        ``interval`` uses this project's own vocabulary: "1h", "4h", "1D".
        Each adapter maps that to its exchange's native interval string.
        """

    @abstractmethod
    def normalize_symbol(self, base_asset: str, quote_asset: str = "USDT") -> str:
        """Build this exchange's native symbol string for a base/quote pair."""

    def get_market_wide_snapshot(
        self,
        quote_asset: str = "USDT",
        exclude_bases: Iterable[str] = (),
    ) -> "MarketSnapshot":
        """Aggregate step-1 "market timing" stats from the tickers endpoint.

        This is a liquidity-weighted proxy for total-market-cap change and
        BTC dominance, built only from exchange data (no CoinGecko
        dependency): quote-volume-weighted average 24h change across all
        pairs, and BTC's share of total quote volume as a dominance proxy.
        It is deliberately named differently from "market cap change" in
        the report layer, since it is not the same metric -- see
        ``market_timing.py`` for the honest framing.
        """
        excluded = {b.upper() for b in exclude_bases}
        # Stablecoin pairs (USDC/USDT, FDUSD/USDT...) carry huge volume and
        # ~0% change; left in, they dilute the weighted move and hide a
        # genuinely sharp market from the regime gate.
        tickers = [
            t for t in self.get_tickers_24h(quote_asset=quote_asset)
            if t.base_asset.upper() not in excluded
        ]
        if not tickers:
            raise RuntimeError(f"{self.name}: no tickers returned for {quote_asset}")

        total_volume = sum(t.quote_volume_24h for t in tickers)
        if total_volume <= 0:
            raise RuntimeError(f"{self.name}: zero aggregate volume, cannot compute snapshot")

        weighted_change = sum(
            t.price_change_pct_24h * t.quote_volume_24h for t in tickers
        ) / total_volume

        btc = next((t for t in tickers if t.base_asset == "BTC"), None)
        btc_dominance_pct = (
            (btc.quote_volume_24h / total_volume) * 100.0 if btc is not None else float("nan")
        )

        return MarketSnapshot(
            exchange=self.name,
            quote_asset=quote_asset,
            pair_count=len(tickers),
            total_quote_volume_24h=total_volume,
            volume_weighted_change_pct_24h=weighted_change,
            btc_quote_volume_share_pct=btc_dominance_pct,
        )


@dataclass(frozen=True)
class MarketSnapshot:
    exchange: str
    quote_asset: str
    pair_count: int
    total_quote_volume_24h: float
    volume_weighted_change_pct_24h: float
    btc_quote_volume_share_pct: float
