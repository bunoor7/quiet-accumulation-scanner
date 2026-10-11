"""OKX public-market-data adapter (Spot). No API key required.

Endpoints used (all public, documented at
https://www.okx.com/docs-v5/en/#order-book-trading-market-data):

- GET /api/v5/market/tickers?instType=SPOT -> 24h snapshot per instrument
- GET /api/v5/market/candles                -> OHLCV candles
- GET /api/v5/public/instruments?instType=SPOT -> base/quote ccy split

OKX returns candles newest-bar-first and as strings; both quirks are
normalized away here so callers see the exact same shape Binance's
adapter returns.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .base import ExchangeClient, Ticker24h
from ._http import get_json

BASE_URL = "https://www.okx.com"

_INTERVAL_MAP = {
    "1h": "1H",
    "4h": "4H",
    "1D": "1Dutc",  # UTC-aligned daily bars, same boundaries as Binance
    "1W": "1Wutc",
}


class OKXClient(ExchangeClient):
    name = "okx"

    def __init__(self, timeout_s: float = 10.0, retries: int = 3, retry_wait_s: float = 2.0):
        self.timeout_s = timeout_s
        self.retries = retries
        self.retry_wait_s = retry_wait_s
        self._instrument_cache: Dict[str, tuple] | None = None

    def _get(self, path: str, params: dict | None = None):
        payload = get_json(
            f"{BASE_URL}{path}",
            params=params,
            timeout_s=self.timeout_s,
            retries=self.retries,
            retry_wait_s=self.retry_wait_s,
        )
        if payload.get("code") != "0":
            raise RuntimeError(f"okx: API error {payload.get('code')}: {payload.get('msg')}")
        return payload["data"]

    def _instruments(self) -> Dict[str, tuple]:
        if self._instrument_cache is not None:
            return self._instrument_cache
        data = self._get("/api/v5/public/instruments", params={"instType": "SPOT"})
        mapping: Dict[str, tuple] = {}
        for row in data:
            if row.get("state") != "live":
                continue
            mapping[row["instId"]] = (row["baseCcy"], row["quoteCcy"])
        self._instrument_cache = mapping
        return mapping

    def normalize_symbol(self, base_asset: str, quote_asset: str = "USDT") -> str:
        return f"{base_asset.upper()}-{quote_asset.upper()}"

    def get_tickers_24h(self, quote_asset: str = "USDT") -> List[Ticker24h]:
        instruments = self._instruments()
        data = self._get("/api/v5/market/tickers", params={"instType": "SPOT"})
        out: List[Ticker24h] = []
        for row in data:
            inst_id = row["instId"]
            assets = instruments.get(inst_id)
            if assets is None:
                continue
            base, quote = assets
            if quote != quote_asset:
                continue
            try:
                last = float(row["last"])
                open24h = float(row["open24h"])
                quote_vol = float(row["volCcy24h"])
                pct = ((last - open24h) / open24h) * 100.0 if open24h > 0 else 0.0
                out.append(
                    Ticker24h(
                        symbol=inst_id,
                        base_asset=base,
                        quote_asset=quote,
                        last_price=last,
                        price_change_pct_24h=pct,
                        quote_volume_24h=quote_vol,
                    )
                )
            except (KeyError, ValueError, ZeroDivisionError):
                continue
        return out

    def get_klines(self, symbol: str, interval: str, limit: int = 240) -> pd.DataFrame:
        native = _INTERVAL_MAP.get(interval)
        if native is None:
            raise ValueError(f"Unsupported interval '{interval}' for OKX")
        raw = self._get(
            "/api/v5/market/candles",
            params={"instId": symbol, "bar": native, "limit": limit},
        )
        if not raw:
            raise RuntimeError(f"okx: empty candles for {symbol} {interval}")
        # OKX candle row: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
        # newest first -- reverse to chronological order to match Binance.
        rows = list(reversed(raw))
        df = pd.DataFrame(
            rows,
            columns=[
                "ts", "open", "high", "low", "close",
                "vol", "volCcy", "volCcyQuote", "confirm",
            ],
        )
        # confirm == "0" marks the still-forming candle. Decisions run on
        # closed bars only, same as the Binance adapter.
        df = df[df["confirm"].astype(str) == "1"]
        if df.empty:
            raise RuntimeError(f"okx: no closed candles for {symbol} {interval}")
        df["close_time"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
        # For SPOT, "vol" is in base currency (same unit as Binance's
        # kline volume); "volCcy" is quote currency. Use base so OBV is
        # comparable across both exchanges.
        for col in ("open", "high", "low", "close", "vol"):
            df[col] = df[col].astype(float)
        df = df.rename(columns={"vol": "volume"})
        df = df.set_index("close_time")[["open", "high", "low", "close", "volume"]]
        return df
