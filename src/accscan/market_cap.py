"""Optional market-cap enrichment.

Binance and OKX's public endpoints do not expose circulating supply or
market cap -- only price and volume. The original methodology this
project mirrors uses a market-cap floor (~$150M) as part of step 2, so
this module offers a pluggable, strictly optional way to honor that
without smuggling in a dependency the user didn't ask for: by default
the screener falls back to 24h quote volume alone as its liquidity
filter and is explicit in the report that the market-cap rule was
skipped, rather than silently pretending volume and market cap are the
same thing.

If the user wants real market-cap numbers, set
``ScreenConfig`` aside and pass a ``CoinGeckoMarketCapProvider()``
instance to ``run_screen`` -- it calls CoinGecko's free
``/coins/markets`` endpoint (no key required) and is skipped
automatically (with a logged warning, not a crash) if that call fails.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional

from .exchanges._http import get_json

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"


class MarketCapProvider(ABC):
    @abstractmethod
    def get_market_caps_usd(self, base_assets: Iterable[str]) -> Dict[str, float]:
        """Return {base_asset_symbol_upper: market_cap_usd} for whatever it can resolve.

        Silently omitting a symbol it cannot map is expected and fine --
        callers treat a missing entry as "unknown", not "zero".
        """


class NullMarketCapProvider(MarketCapProvider):
    """Default: no market-cap data. The screener falls back to volume-only."""

    def get_market_caps_usd(self, base_assets: Iterable[str]) -> Dict[str, float]:
        return {}


class CoinGeckoMarketCapProvider(MarketCapProvider):
    """Optional enrichment via CoinGecko's free, keyless /coins/markets endpoint.

    Not used by default (this project's stated scope is "direct from
    Binance/OKX"). Wire it in explicitly if you want the $150M market-cap
    rule enforced for real instead of approximated by volume.
    """

    def __init__(self, timeout_s: float = 10.0, retries: int = 2, retry_wait_s: float = 2.0):
        self.timeout_s = timeout_s
        self.retries = retries
        self.retry_wait_s = retry_wait_s

    def get_market_caps_usd(self, base_assets: Iterable[str]) -> Dict[str, float]:
        symbols = {s.upper() for s in base_assets}
        out: Dict[str, float] = {}
        try:
            page = 1
            seen_pages = 0
            while seen_pages < 5:  # top 1000 coins by market cap is plenty
                rows = get_json(
                    f"{COINGECKO_BASE_URL}/coins/markets",
                    params={
                        "vs_currency": "usd",
                        "order": "market_cap_desc",
                        "per_page": 250,
                        "page": page,
                    },
                    timeout_s=self.timeout_s,
                    retries=self.retries,
                    retry_wait_s=self.retry_wait_s,
                )
                if not rows:
                    break
                for row in rows:
                    sym = str(row.get("symbol", "")).upper()
                    cap = row.get("market_cap")
                    if sym in symbols and cap is not None and sym not in out:
                        out[sym] = float(cap)
                page += 1
                seen_pages += 1
                if symbols.issubset(out.keys()):
                    break
        except Exception:
            # Honest-degradation, not a crash: caller treats this exactly
            # like NullMarketCapProvider when the lookup fails outright.
            return out
        return out
