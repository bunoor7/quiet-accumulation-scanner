"""Step 3: multi-timeframe confirmation.

For every shortlisted symbol, pull 1h/4h/1D bars, compute the full
indicator set on each, and apply the two exclusion rules from the
methodology:

1. Already extended on the higher timeframe (price too far above its
   1D EMA200) -> it has already run, this is no longer the ideal
   entry phase.
2. The move is indistinguishable from broad market beta -- only
   checked when step 1 found the market itself in a sharp trend,
   since that is the only situation where "this coin moved with
   everything else" is a real risk of being mistaken for
   coin-specific accumulation or weakness.

What survives both checks is genuinely "quiet and compressed", not
merely sitting in the RSI/performance band by coincidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from . import indicators as ind
from .config import ScreenConfig
from .exchanges.base import ExchangeClient
from .market_timing import MarketTimingResult, MarketRegime
from .screener import Candidate


@dataclass
class TimeframeRead:
    timeframe: str
    df: pd.DataFrame  # full indicator-enriched OHLCV, most recent bar last

    @property
    def last(self) -> pd.Series:
        return self.df.iloc[-1]


@dataclass
class ConfirmedCandidate:
    candidate: Candidate
    reads: Dict[str, TimeframeRead] = field(default_factory=dict)
    excluded_reason: Optional[str] = None

    @property
    def survived(self) -> bool:
        return self.excluded_reason is None


def _price_vs_ema200_pct(last_row: pd.Series) -> float:
    ema200 = last_row.get("ema200")
    if ema200 is None or ema200 != ema200 or ema200 == 0:
        return float("nan")
    return (last_row["close"] - ema200) / ema200 * 100.0


def confirm_candidates(
    client: ExchangeClient,
    shortlist: List[Candidate],
    market_timing: MarketTimingResult,
    cfg: ScreenConfig,
) -> List[ConfirmedCandidate]:
    results: List[ConfirmedCandidate] = []

    for cand in shortlist:
        result = ConfirmedCandidate(candidate=cand)
        ok = True
        for tf in cfg.timeframes:
            try:
                raw = client.get_klines(cand.symbol, tf, limit=cfg.klines_limit)
            except Exception as exc:  # noqa: BLE001
                result.excluded_reason = f"data_unavailable:{tf}:{exc}"
                ok = False
                break
            if len(raw) < 50:
                result.excluded_reason = f"insufficient_history:{tf}"
                ok = False
                break
            enriched = ind.compute_all(raw)
            result.reads[tf] = TimeframeRead(timeframe=tf, df=enriched)

        if not ok:
            results.append(result)
            continue

        daily = result.reads.get("1D")
        if daily is not None:
            ext_pct = _price_vs_ema200_pct(daily.last)
            if ext_pct == ext_pct and ext_pct > cfg.overextended_price_vs_ema200_pct:
                result.excluded_reason = (
                    f"already_extended: price is {ext_pct:.1f}% above the 1D EMA200, "
                    "this missed its ideal quiet-accumulation phase"
                )
                results.append(result)
                continue

        if market_timing.regime is MarketRegime.SHARP_TREND:
            market_change = market_timing.snapshot.volume_weighted_change_pct_24h
            coin_change_24h: Optional[float] = None
            # Use the 1D frame's last two closes for an apples-to-apples 24h change.
            if daily is not None and len(daily.df) >= 2:
                c_now = daily.df["close"].iloc[-1]
                c_prev = daily.df["close"].iloc[-2]
                coin_change_24h = (c_now - c_prev) / c_prev * 100.0 if c_prev else None
            if coin_change_24h is not None and abs(coin_change_24h - market_change) < 1.0:
                result.excluded_reason = (
                    "market_beta: the coin's recent move tracks the broad market's own "
                    "sharp swing too closely to call it independent accumulation or weakness"
                )
                results.append(result)
                continue

        results.append(result)

    return results


def surviving(confirmed: List[ConfirmedCandidate]) -> List[ConfirmedCandidate]:
    return [c for c in confirmed if c.survived]
