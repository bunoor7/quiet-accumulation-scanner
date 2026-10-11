"""Binance public-market-data adapter (Spot). No API key required.

Endpoints used (all public, documented at
https://developers.binance.com/docs/binance-spot-api-docs):

- GET /api/v3/ticker/24hr  -> 24h rolling stats for every symbol
- GET /api/v3/klines       -> OHLCV candles for one symbol/interval
- GET /api/v3/exchangeInfo -> symbol metadata (base/quote asset split)
"""
from __future__ import annotations

import time
from typing import Dict, List

import pandas as pd

from .base import ExchangeClient, Ticker24h
from ._http import get_json

BASE_URL = "https://api.binance.com"

# This project's own interval vocabulary -> Binance's native kline interval.
_INTERVAL_MAP = {
    "1h": "1h",
    "4h": "4h",
    "1D": "1d",
    "1W": "1w",
}


def _now_ms() -> int:
    """Current UTC time in ms. A function (not inline) so tests can pin it."""
    return int(time.time() * 1000)


class BinanceClient(ExchangeClient):
    name = "binance"

    def __init__(self, timeout_s: float = 10.0, retries: int = 3, retry_wait_s: float = 2.0):
        self.timeout_s = timeout_s
        self.retries = retries
        self.retry_wait_s = retry_wait_s
        self._symbol_asset_cache: Dict[str, tuple] | None = None

    def _get(self, path: str, params: dict | None = None):
        return get_json(
            f"{BASE_URL}{path}",
            params=params,
            timeout_s=self.timeout_s,
            retries=self.retries,
            retry_wait_s=self.retry_wait_s,
        )

    def _symbol_to_assets(self) -> Dict[str, tuple]:
        """Map e.g. "ETHFIUSDT" -> ("ETHFI", "USDT") using exchangeInfo.

        Binance ticker/24hr does not split base/quote itself, so this one
        extra (cached) call is what lets the rest of the pipeline reason
        about base assets without guessing where the quote suffix starts
        (ambiguous for bases that themselves end in "USD"-like strings).
        """
        if self._symbol_asset_cache is not None:
            return self._symbol_asset_cache
        info = self._get("/api/v3/exchangeInfo")
        mapping: Dict[str, tuple] = {}
        for s in info.get("symbols", []):
            if s.get("status") != "TRADING":
                continue
            mapping[s["symbol"]] = (s["baseAsset"], s["quoteAsset"])
        self._symbol_asset_cache = mapping
        return mapping

    def normalize_symbol(self, base_asset: str, quote_asset: str = "USDT") -> str:
        return f"{base_asset.upper()}{quote_asset.upper()}"

    def get_tickers_24h(self, quote_asset: str = "USDT") -> List[Ticker24h]:
        asset_map = self._symbol_to_assets()
        raw = self._get("/api/v3/ticker/24hr")
        out: List[Ticker24h] = []
        for row in raw:
            sym = row["symbol"]
            assets = asset_map.get(sym)
            if assets is None:
                continue
            base, quote = assets
            if quote != quote_asset:
                continue
            try:
                out.append(
                    Ticker24h(
                        symbol=sym,
                        base_asset=base,
                        quote_asset=quote,
                        last_price=float(row["lastPrice"]),
                        price_change_pct_24h=float(row["priceChangePercent"]),
                        quote_volume_24h=float(row["quoteVolume"]),
                    )
                )
            except (KeyError, ValueError):
                continue
        return out

    def get_klines(self, symbol: str, interval: str, limit: int = 240) -> pd.DataFrame:
        native = _INTERVAL_MAP.get(interval)
        if native is None:
            raise ValueError(f"Unsupported interval '{interval}' for Binance")
        raw = self._get(
            "/api/v3/klines",
            params={"symbol": symbol, "interval": native, "limit": limit},
        )
        if not raw:
            raise RuntimeError(f"binance: empty klines for {symbol} {interval}")
        df = pd.DataFrame(
            raw,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades",
                "taker_base_vol", "taker_quote_vol", "ignore",
            ],
        )
        # Binance always returns the still-forming candle as the last row.
        # Its close_time lies in the future; every indicator and decision
        # must run on closed bars only, otherwise a signal can appear
        # mid-candle and vanish at the close.
        df = df[df["close_time"].astype("int64") < _now_ms()]
        if df.empty:
            raise RuntimeError(f"binance: no closed klines for {symbol} {interval}")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        df = df.set_index("close_time")[["open", "high", "low", "close", "volume"]]
        return df
