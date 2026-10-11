from __future__ import annotations

from .base import ExchangeClient, MarketSnapshot, Ticker24h
from .binance import BinanceClient
from .okx import OKXClient

_REGISTRY = {
    "binance": BinanceClient,
    "okx": OKXClient,
}


def get_exchange_client(name: str, **kwargs) -> ExchangeClient:
    """Factory: get_exchange_client("binance") / get_exchange_client("okx")."""
    try:
        cls = _REGISTRY[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown exchange '{name}'. Available: {sorted(_REGISTRY)}"
        ) from exc
    return cls(**kwargs)


__all__ = [
    "ExchangeClient",
    "MarketSnapshot",
    "Ticker24h",
    "BinanceClient",
    "OKXClient",
    "get_exchange_client",
]
