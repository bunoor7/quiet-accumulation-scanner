"""Step 1: is the market as a whole quiet, actively ranging, or trending?

This is built only from exchange data (no CoinGecko dependency): a
quote-volume-weighted average 24h change across every USDT pair on the
chosen exchange, and BTC's share of total USDT quote volume as a rough
dominance proxy. It is honestly weaker than a true total-market-cap
figure (CoinGecko aggregates far more venues and supply data), so the
report layer always labels it as an exchange-liquidity proxy, never as
"market cap".
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .config import ScreenConfig
from .exchanges.base import ExchangeClient, MarketSnapshot


class MarketRegime(str, Enum):
    SHARP_TREND = "sharp_trend"       # abs(weighted 24h change) > threshold
    ACTIVE_RANGE = "active_range"     # moderate activity, no sharp direction
    QUIET_LOW_LIQUIDITY = "quiet_low_liquidity"


@dataclass(frozen=True)
class MarketTimingResult:
    snapshot: MarketSnapshot
    regime: MarketRegime
    strict_mode: bool  # True -> apply the tighter RSI band from config


def assess_market_timing(client: ExchangeClient, cfg: ScreenConfig) -> MarketTimingResult:
    snapshot = client.get_market_wide_snapshot(quote_asset=cfg.quote_asset)
    change = snapshot.volume_weighted_change_pct_24h

    if abs(change) > cfg.sharp_market_move_pct:
        regime = MarketRegime.SHARP_TREND
    elif abs(change) < 0.5 and snapshot.total_quote_volume_24h <= 0:
        # Defensive branch; total_quote_volume_24h is validated > 0 upstream,
        # kept here so a future refactor that relaxes that check stays safe.
        regime = MarketRegime.QUIET_LOW_LIQUIDITY
    else:
        regime = MarketRegime.ACTIVE_RANGE

    strict_mode = regime is MarketRegime.SHARP_TREND
    return MarketTimingResult(snapshot=snapshot, regime=regime, strict_mode=strict_mode)
