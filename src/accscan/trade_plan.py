"""Step 6: trade plan construction.

Only ever called on a candidate that already passed the step 5
honesty gate as a genuine signal. Finds the nearest real swing
support/resistance around the current price, picks a stop from
whichever of (ATR multiple, nearest support) is the more relevant
protective level, and builds one or two resistance-based targets with
their reward:risk ratios.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .config import ScreenConfig


@dataclass
class SwingLevels:
    supports: List[float]   # sorted descending, all below current price
    resistances: List[float]  # sorted ascending, all above current price


@dataclass
class TradePlan:
    symbol: str
    entry: float
    stop: float
    stop_basis: str  # "atr" | "structure"
    atr: float
    targets: List[float]
    reward_risk: List[float]
    conviction: str


def find_swing_levels(df: pd.DataFrame, current_price: float, pivot_window: int = 5) -> SwingLevels:
    """Fractal-style swing highs/lows: a bar is a pivot if it is the
    highest/lowest close within ``pivot_window`` bars on each side.
    """
    closes = df["close"]
    n = len(closes)
    supports, resistances = [], []
    for i in range(pivot_window, n - pivot_window):
        window = closes.iloc[i - pivot_window: i + pivot_window + 1]
        price = closes.iloc[i]
        if price == window.max():
            if price > current_price:
                resistances.append(float(price))
        if price == window.min():
            if price < current_price:
                supports.append(float(price))

    supports = sorted(set(supports), reverse=True)  # nearest first (highest below price)
    resistances = sorted(set(resistances))           # nearest first (lowest above price)
    return SwingLevels(supports=supports, resistances=resistances)


def build_trade_plan(
    symbol: str,
    df_decision_tf: pd.DataFrame,
    cfg: ScreenConfig,
    conviction: str,
) -> Optional[TradePlan]:
    last = df_decision_tf.iloc[-1]
    entry = float(last["close"])
    atr_val = float(last["atr14"]) if last["atr14"] == last["atr14"] else None
    if atr_val is None or atr_val <= 0:
        return None

    levels = find_swing_levels(df_decision_tf, entry)

    atr_stop = entry - cfg.atr_stop_multiple * atr_val
    structure_stop = levels.supports[0] * 0.999 if levels.supports else None

    if structure_stop is not None and structure_stop > atr_stop:
        # The nearest real support is tighter than the pure-ATR stop and
        # still sits below entry -- prefer it, it's a level the market
        # has actually respected before.
        stop = structure_stop
        stop_basis = "structure"
    else:
        stop = atr_stop
        stop_basis = "atr"

    risk = entry - stop
    if risk <= 0:
        return None

    targets: List[float] = []
    if levels.resistances:
        targets.append(levels.resistances[0])
        if len(levels.resistances) > 1:
            targets.append(levels.resistances[1])
    if not targets:
        # No resistance found in the fetched window -- fall back to a
        # measured move (2x the risk) rather than inventing a level.
        targets.append(entry + 2 * risk)

    reward_risk = [(t - entry) / risk for t in targets]

    return TradePlan(
        symbol=symbol, entry=entry, stop=stop, stop_basis=stop_basis,
        atr=atr_val, targets=targets, reward_risk=reward_risk, conviction=conviction,
    )
