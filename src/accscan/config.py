"""Tunable thresholds for the quiet-accumulation methodology.

Every number here mirrors a specific rule from the methodology
description in the README. Nothing is hard-coded elsewhere so the
whole screen can be re-tuned from one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ScreenConfig:
    # --- Universe filters (step 2) ---
    quote_asset: str = "USDT"
    min_24h_quote_volume_usd: float = 15_000_000.0
    min_market_cap_usd: float = 150_000_000.0  # only enforced if a market-cap
    #                                            provider is configured; see
    #                                            market_cap.py.
    exclude_bases: List[str] = field(
        default_factory=lambda: [
            "BTC", "ETH", "WBTC", "WETH", "CBBTC",
        ]
    )
    # Heuristic stablecoin/pegged-asset filter: any base symbol in this list,
    # or whose 90-day price range is tighter than stable_price_range_pct,
    # is dropped before scoring.
    known_stable_bases: List[str] = field(
        default_factory=lambda: [
            "USDT", "USDC", "FDUSD", "DAI", "TUSD", "USDP", "USD1", "USDE",
            "PYUSD", "USDD", "GUSD", "EURT", "EURS", "XAUT", "PAXG",
        ]
    )
    stable_price_range_pct: float = 1.5  # 90d (high-low)/low below this % -> treat as pegged, drop

    # --- Accumulation shortlist (step 2) ---
    rsi_band: tuple = (45.0, 60.0)
    monthly_perf_band_pct: tuple = (-15.0, 15.0)
    shortlist_min: int = 8
    shortlist_max: int = 15
    max_universe_size: int = 300  # safety cap on how many symbols get a
    #                               klines fetch during step 2, ranked by
    #                               24h quote volume so the most liquid
    #                               names are never the ones dropped.

    # --- Market-wide regime gate (step 1) ---
    sharp_market_move_pct: float = 4.0  # abs 24h market-wide move beyond this
    #                                     triggers stricter acceptance below.
    strict_rsi_band: tuple = (48.0, 56.0)  # used instead of rsi_band when the
    #                                        market-wide move is sharp.

    # --- Multi-timeframe confirmation (step 3) ---
    timeframes: List[str] = field(default_factory=lambda: ["1h", "4h", "1D"])
    overextended_price_vs_ema200_pct: float = 20.0  # exclude if price is this
    #                                                  far above EMA200 on the
    #                                                  1D frame -- already ran.

    # --- Quiet-compression / trigger thresholds (step 5) ---
    low_adx_threshold: float = 20.0
    adx_rising_min_delta: float = 2.0  # ADX must have risen at least this
    #                                    much over the lookback window to
    #                                    count as an "early trigger".
    bb_squeeze_percentile: float = 0.25  # current BB width must sit in the
    #                                      lowest 25% of its own trailing
    #                                      history to count as "compressed".
    bb_expansion_min_pct: float = 8.0  # width must have widened at least this
    #                                    many percent off its squeeze low to
    #                                    count as "starting to release".
    obv_lookback_bars: int = 20
    obv_rising_min_pct: float = 0.0  # OBV slope over the lookback must be >=
    #                                  this (0 = "flat or rising", matches
    #                                  "مرتفع أو مستقر" from the methodology).

    # --- Trade plan (step 6) ---
    atr_stop_multiple: float = 1.5
    min_reward_risk: float = 1.5

    # --- Data freshness / retry policy ---
    http_timeout_s: float = 10.0
    http_retries: int = 3
    http_retry_wait_s: float = 2.0
    klines_limit: int = 240  # bars fetched per timeframe
