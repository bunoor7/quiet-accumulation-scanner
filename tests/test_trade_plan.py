"""Step 6 (trade plan construction) tests.

Like test_signal_engine.py, these build the decision-frame DataFrame
directly (close + atr14 columns) rather than deriving it from raw
OHLCV, so each case is an exact statement of "the swing levels look
like this" instead of an artifact of a random walk.
"""
from __future__ import annotations

import pandas as pd
import pytest

from accscan.config import ScreenConfig
from accscan.trade_plan import build_trade_plan, find_swing_levels


def _df_with_pivots(closes, atr: float = 1.0):
    n = len(closes)
    idx = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"close": closes, "atr14": [atr] * n}, index=idx)


def test_find_swing_levels_identifies_nearest_support_and_resistance():
    # A clean V-then-up shape: a support pivot at 95 well below the
    # current price, a resistance pivot at 110 above it.
    closes = [100, 99, 97, 95, 97, 99, 102, 106, 110, 107, 103, 101, 103, 105]
    df = _df_with_pivots(closes)
    levels = find_swing_levels(df, current_price=103.5, pivot_window=2)
    assert 95.0 in levels.supports
    assert 110.0 in levels.resistances
    # Nearest-first ordering: supports descending, resistances ascending.
    assert levels.supports == sorted(levels.supports, reverse=True)
    assert levels.resistances == sorted(levels.resistances)


def test_build_trade_plan_prefers_structure_stop_when_tighter():
    # ATR is large enough that the pure-ATR stop would sit below the
    # nearest real support -- the tighter, structurally-respected
    # support should win instead. Default pivot_window (5) needs an
    # 11-bar neighborhood around each pivot, hence the longer V shape.
    closes = [105, 104, 103, 102, 101, 100, 99, 98, 95, 98, 99, 100, 101,
              102, 103, 104, 105, 106, 107, 108, 109, 110, 109, 108, 107,
              106, 105, 104, 103, 102]
    df = _df_with_pivots(closes, atr=5.0)
    cfg = ScreenConfig()
    plan = build_trade_plan("ETHFIUSDT", df, cfg, conviction="medium")

    assert plan is not None
    assert plan.entry == pytest.approx(102.0)
    atr_stop = 102.0 - cfg.atr_stop_multiple * 5.0  # = 94.5
    assert plan.stop > atr_stop
    assert plan.stop_basis == "structure"
    assert plan.stop == pytest.approx(95.0 * 0.999)


def test_build_trade_plan_falls_back_to_atr_stop_when_no_support():
    # Strictly rising closes -> find_swing_levels can never find a
    # pivot low below the current price (every prior low is itself
    # immediately undercut), so the stop must fall back to pure ATR.
    closes = [100 + i for i in range(14)]
    df = _df_with_pivots(closes, atr=2.0)
    cfg = ScreenConfig()
    plan = build_trade_plan("ETHFIUSDT", df, cfg, conviction="low")

    assert plan is not None
    assert plan.stop_basis == "atr"
    assert plan.stop == pytest.approx(plan.entry - cfg.atr_stop_multiple * 2.0)


def test_build_trade_plan_uses_measured_move_when_no_resistance_found():
    # Strictly rising closes also means no resistance pivot above the
    # current price -- target must fall back to a 2x-risk measured move
    # rather than inventing a level.
    closes = [100 + i for i in range(14)]
    df = _df_with_pivots(closes, atr=2.0)
    cfg = ScreenConfig()
    plan = build_trade_plan("ETHFIUSDT", df, cfg, conviction="low")

    assert plan is not None
    risk = plan.entry - plan.stop
    assert plan.targets == [pytest.approx(plan.entry + 2 * risk)]
    assert plan.reward_risk == [pytest.approx(2.0)]


def test_build_trade_plan_returns_none_when_atr_is_missing():
    closes = [100 + i for i in range(14)]
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="4h", tz="UTC")
    df = pd.DataFrame({"close": closes, "atr14": [float("nan")] * len(closes)}, index=idx)
    cfg = ScreenConfig()
    assert build_trade_plan("ETHFIUSDT", df, cfg, conviction="low") is None


def test_build_trade_plan_reward_risk_matches_two_targets():
    # Two peaks (111, 112) well above a final low around 92, each with
    # a full 11-bar neighborhood so both register as resistance pivots.
    closes = (
        [100, 98, 96, 94, 92, 90]
        + [93, 96, 99, 102, 105, 108, 111]
        + [108, 105, 102, 99, 96, 93, 90]
        + [93, 96, 99, 102, 105, 108, 112]
        + [108, 104, 100, 96, 92]
    )
    df = _df_with_pivots(closes, atr=1.0)
    cfg = ScreenConfig()
    plan = build_trade_plan("ETHFIUSDT", df, cfg, conviction="high")

    assert plan is not None
    assert plan.targets == [pytest.approx(111.0), pytest.approx(112.0)]
    risk = plan.entry - plan.stop
    assert risk > 0
    for target, rr in zip(plan.targets, plan.reward_risk):
        assert rr == pytest.approx((target - plan.entry) / risk)
