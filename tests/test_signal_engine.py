"""Signal-engine tests build the indicator columns directly rather than
round-tripping through compute_all on synthetic price noise -- that
keeps each case an exact, intentional statement of "ADX is this,
RSI is that", instead of fighting a random-walk generator to land on
the exact numbers a scenario needs.
"""
from __future__ import annotations

import pandas as pd
import pytest

from accscan.config import ScreenConfig
from accscan.multi_timeframe import ConfirmedCandidate, TimeframeRead
from accscan.screener import Candidate
from accscan.signal_engine import evaluate_candidate, pick_best_signal


def _frame(
    *,
    adx_baseline: float,
    adx_current: float,
    bb_baseline: float,
    bb_squeeze_low: float,
    bb_current: float,
    rsi14: float,
    obv_lookback_start: float,
    obv_current: float,
    price_rising: bool = True,
    n: int = 120,
) -> pd.DataFrame:
    """Build an n-bar 4h frame with each indicator's history shape under
    direct, independent control, instead of one shared ramp formula
    standing in for every scenario:

    - ADX sits flat at ``adx_baseline`` for the whole history except the
      very last bar, which is ``adx_current``. The trigger check's
      lookback (5 bars back) therefore reads the baseline, so adx_delta
      is exactly ``adx_current - adx_baseline`` -- no incidental ramp
      shape to account for.
    - Bollinger width sits at a wider ``bb_baseline`` for most of the
      history, dips to a genuine squeeze at ``bb_squeeze_low`` for the
      15 bars right before the end, then the final bar is ``bb_current``.
      That makes the current width both a true low percentile of its
      trailing 100-bar history (a real squeeze, not just "lower than a
      ramp's own peak") and comparable against the trailing 50-bar
      squeeze low for the expansion trigger.
    - OBV sits flat at 0 for most of the history, then over the last 20
      bars holds ``obv_lookback_start`` for all but the final bar, which
      is ``obv_current`` -- an exact, intentional slope and "fresh local
      high" check.
    - Close is flat except the last 3 bars, which tick slightly up when
      ``price_rising`` (for OBV-breakout's "price follows" check) or
      slightly down when not.
    """
    idx = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")

    adx_series = pd.Series([adx_baseline] * (n - 1) + [adx_current], index=idx)

    squeeze_len = 15
    baseline_len = n - squeeze_len - 1
    width_series = pd.Series(
        [bb_baseline] * baseline_len + [bb_squeeze_low] * squeeze_len + [bb_current],
        index=idx,
    )

    obv_lookback_bars = 20
    obv_series = pd.Series(
        [0.0] * (n - obv_lookback_bars)
        + [obv_lookback_start] * (obv_lookback_bars - 1)
        + [obv_current],
        index=idx,
    )

    close_series = pd.Series([1.0] * n, index=idx)
    close_series.iloc[-3:] = [1.0, 1.001, 1.002] if price_rising else [1.002, 1.001, 1.0]

    rsi_series = pd.Series([rsi14] * n, index=idx)

    return pd.DataFrame(
        {
            "close": close_series,
            "adx": adx_series,
            "bb_width_pct": width_series,
            "rsi14": rsi_series,
            "obv": obv_series,
        },
        index=idx,
    )


def _candidate() -> Candidate:
    return Candidate(
        symbol="ETHFIUSDT", base_asset="ETHFI", quote_asset="USDT",
        last_price=0.69, quote_volume_24h=36_000_000.0, market_cap_usd=665_000_000.0,
        rsi14_1d=50.0, perf_30d_pct=3.8, price_range_90bar_pct=40.0,
    )


def _confirmed_with(df_4h: pd.DataFrame) -> ConfirmedCandidate:
    return ConfirmedCandidate(
        candidate=_candidate(),
        reads={"4h": TimeframeRead(timeframe="4h", df=df_4h)},
    )


def test_clean_setup_fires_a_signal():
    cfg = ScreenConfig()
    # ADX low and rising (15 -> 19), Bollinger genuinely squeezed then
    # releasing (0.8 -> 1.0 off a 3.0 baseline), RSI neutral, OBV making
    # a fresh high with price following -- every mandatory compression
    # criterion holds AND there is a concrete trigger.
    df = _frame(
        adx_baseline=15.0, adx_current=19.0,
        bb_baseline=3.0, bb_squeeze_low=0.8, bb_current=1.0,
        rsi14=55.0,
        obv_lookback_start=0.0, obv_current=5_000_000.0, price_rising=True,
    )
    confirmed = _confirmed_with(df)
    verdict = evaluate_candidate(confirmed, cfg, decision_timeframe="4h")
    assert verdict.is_signal, verdict.reject_reason
    assert verdict.conviction in {"low", "medium", "high"}


def test_quiet_but_no_trigger_is_rejected():
    cfg = ScreenConfig()
    # Genuinely quiet (low flat ADX, a real squeeze, neutral RSI) but
    # nothing has actually released yet: ADX hasn't moved, Bollinger
    # width hasn't expanded off its squeeze low, and OBV isn't making a
    # fresh high with price following -- correctly "no signal yet", not
    # a weak signal.
    df = _frame(
        adx_baseline=15.0, adx_current=15.0,
        bb_baseline=3.0, bb_squeeze_low=0.9, bb_current=0.9,
        rsi14=52.0,
        obv_lookback_start=0.0, obv_current=0.0, price_rising=False,
    )
    confirmed = _confirmed_with(df)
    verdict = evaluate_candidate(confirmed, cfg, decision_timeframe="4h")
    assert not verdict.is_signal
    assert verdict.reject_reason is not None


def test_overbought_rsi_blocks_a_signal_even_with_a_trigger():
    cfg = ScreenConfig()
    # A real trigger (ADX rising, Bollinger releasing, OBV breaking out)
    # is present, but RSI is overbought -- RSI is one of the three
    # mandatory compression criteria, so this must still be rejected.
    df = _frame(
        adx_baseline=15.0, adx_current=19.0,
        bb_baseline=3.0, bb_squeeze_low=0.8, bb_current=1.2,
        rsi14=78.0,
        obv_lookback_start=0.0, obv_current=5_000_000.0, price_rising=True,
    )
    confirmed = _confirmed_with(df)
    verdict = evaluate_candidate(confirmed, cfg, decision_timeframe="4h")
    assert not verdict.is_signal
    assert not verdict.compression.rsi_neutral


def test_missing_timeframe_data_is_a_clean_rejection_not_a_crash():
    cfg = ScreenConfig()
    confirmed = ConfirmedCandidate(candidate=_candidate(), reads={})
    verdict = evaluate_candidate(confirmed, cfg, decision_timeframe="4h")
    assert not verdict.is_signal
    assert "no 4h data" in (verdict.reject_reason or "")


def test_pick_best_signal_prefers_higher_conviction():
    cfg = ScreenConfig()
    # Weak: just barely qualifies -- one trigger criterion (ADX turning
    # up) and a flat (not rising) OBV.
    weak_df = _frame(
        adx_baseline=15.0, adx_current=17.0,
        bb_baseline=3.0, bb_squeeze_low=1.0, bb_current=1.0,
        rsi14=58.0,
        obv_lookback_start=0.0, obv_current=0.0, price_rising=False,
    )
    # Strong: all three trigger criteria firing at once, plus an OBV
    # that is actually rising.
    strong_df = _frame(
        adx_baseline=10.0, adx_current=19.0,
        bb_baseline=6.0, bb_squeeze_low=0.8, bb_current=5.0,
        rsi14=55.0,
        obv_lookback_start=0.0, obv_current=8_000_000.0, price_rising=True,
    )

    weak = evaluate_candidate(_confirmed_with(weak_df), cfg, "4h")
    strong = evaluate_candidate(_confirmed_with(strong_df), cfg, "4h")
    assert weak.is_signal and strong.is_signal

    best = pick_best_signal([weak, strong])
    assert best is not None
    rank = {"high": 2, "medium": 1, "low": 0}
    assert rank[best.conviction] >= rank[weak.conviction]
    assert best is strong


def test_pick_best_signal_returns_none_when_nothing_qualifies():
    cfg = ScreenConfig()
    df = _frame(
        adx_baseline=15.0, adx_current=15.0,
        bb_baseline=3.0, bb_squeeze_low=0.9, bb_current=0.9,
        rsi14=52.0,
        obv_lookback_start=0.0, obv_current=0.0, price_rising=False,
    )
    verdict = evaluate_candidate(_confirmed_with(df), cfg, "4h")
    assert pick_best_signal([verdict]) is None
