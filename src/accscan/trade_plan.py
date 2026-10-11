"""Step 6: trade plan construction.

Only ever called on a candidate that already passed the step 5
honesty gate as a genuine signal. Finds the nearest real swing
support/resistance around the current price, picks a stop from
whichever of (ATR multiple, nearest support at least one ATR away) is
tighter, and builds one or two resistance-based targets with their
reward:risk ratios. A setup whose nearest resistance cannot pay
``min_reward_risk`` is rejected, not reported.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

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


def build_trade_plan_with_reason(
    symbol: str,
    df_decision_tf: pd.DataFrame,
    cfg: ScreenConfig,
    conviction: str,
) -> Tuple[Optional[TradePlan], Optional[str]]:
    """Return (plan, None) for a tradeable setup, or (None, arabic_reason).

    Stop rule: the nearest swing support that sits at least
    ``min_stop_atr_multiple`` ATRs below entry, when it is tighter than
    the pure ATR stop; otherwise the ATR stop. A support hugging the
    entry is noise, not protection, and would also inflate the R:R.

    Target rule: the nearest resistance must itself offer at least
    ``min_reward_risk``. Price has to clear it before any farther level
    matters, so a too-close first resistance rejects the trade rather
    than being skipped over.
    """
    last = df_decision_tf.iloc[-1]
    entry = float(last["close"])
    atr_val = float(last["atr14"]) if last["atr14"] == last["atr14"] else None
    if atr_val is None or atr_val <= 0:
        return None, "تعذر حساب المدى الحقيقي المتوسط على إطار القرار"

    levels = find_swing_levels(df_decision_tf, entry)

    atr_stop = entry - cfg.atr_stop_multiple * atr_val
    min_distance = cfg.min_stop_atr_multiple * atr_val
    structure_stop = next(
        (s * 0.999 for s in levels.supports if entry - s * 0.999 >= min_distance),
        None,
    )

    if structure_stop is not None and structure_stop > atr_stop:
        stop = structure_stop
        stop_basis = "structure"
    else:
        stop = atr_stop
        stop_basis = "atr"

    risk = entry - stop
    if risk <= 0:
        return None, "وقف الخسارة المحسوب لا يقع تحت سعر الدخول"

    targets: List[float] = []
    if levels.resistances:
        first_rr = (levels.resistances[0] - entry) / risk
        if first_rr < cfg.min_reward_risk:
            return None, (
                f"أقرب مقاومة عند {levels.resistances[0]:.6g} تعطي عائدا إلى مخاطرة "
                f"{first_rr:.2f} فقط، دون الحد الأدنى {cfg.min_reward_risk:.1f}"
            )
        targets.append(levels.resistances[0])
        if len(levels.resistances) > 1:
            targets.append(levels.resistances[1])
    else:
        # No resistance found in the fetched window -- fall back to a
        # measured move (2x the risk) rather than inventing a level.
        targets.append(entry + 2 * risk)

    reward_risk = [(t - entry) / risk for t in targets]

    return TradePlan(
        symbol=symbol, entry=entry, stop=stop, stop_basis=stop_basis,
        atr=atr_val, targets=targets, reward_risk=reward_risk, conviction=conviction,
    ), None


def build_trade_plan(
    symbol: str,
    df_decision_tf: pd.DataFrame,
    cfg: ScreenConfig,
    conviction: str,
) -> Optional[TradePlan]:
    plan, _ = build_trade_plan_with_reason(symbol, df_decision_tf, cfg, conviction)
    return plan
