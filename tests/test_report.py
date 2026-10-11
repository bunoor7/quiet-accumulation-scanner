"""Report formatting tests: mostly a smoke test that both output
shapes render without crashing and contain the right structural
markers (the methodology's seven numbered sections, or the single
terse no-signal line), not a check on exact Arabic wording.
"""
from __future__ import annotations

import pandas as pd
import pytest

from accscan.config import ScreenConfig
from accscan.exchanges.base import MarketSnapshot
from accscan.market_timing import MarketRegime, MarketTimingResult
from accscan.multi_timeframe import ConfirmedCandidate, TimeframeRead
from accscan.report import format_no_signal, format_signal_report
from accscan.screener import Candidate
from accscan.signal_engine import evaluate_candidate
from accscan.trade_plan import build_trade_plan


def test_format_no_signal_is_a_single_line_carrying_the_reason():
    msg = format_no_signal("لا يوجد تراكم حقيقي في هذا التشغيل")
    assert "\n" not in msg
    assert "لا يوجد تراكم حقيقي في هذا التشغيل" in msg


def _signal_frame() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=120, freq="4h", tz="UTC")
    n = len(idx)
    adx_series = pd.Series([15.0] * (n - 1) + [19.0], index=idx)
    squeeze_len, baseline_len = 15, n - 15 - 1
    width_series = pd.Series(
        [3.0] * baseline_len + [0.8] * squeeze_len + [1.0], index=idx,
    )
    obv_series = pd.Series([0.0] * (n - 20) + [0.0] * 19 + [5_000_000.0], index=idx)
    close_series = pd.Series([1.0] * n, index=idx)
    close_series.iloc[-3:] = [1.0, 1.001, 1.002]
    rsi_series = pd.Series([55.0] * n, index=idx)
    # build_trade_plan needs atr14 too.
    atr_series = pd.Series([0.05] * n, index=idx)
    df = pd.DataFrame(
        {
            "close": close_series, "adx": adx_series, "bb_width_pct": width_series,
            "rsi14": rsi_series, "obv": obv_series, "atr14": atr_series,
            "macd_hist": pd.Series([0.01] * n, index=idx),
        },
        index=idx,
    )
    # Give it a few swing lows/highs so build_trade_plan has real levels.
    closes = list(df["close"])
    closes[50] = 0.9
    closes[70] = 1.3
    df["close"] = closes
    return df


def _confirmed() -> ConfirmedCandidate:
    cand = Candidate(
        symbol="ETHFIUSDT", base_asset="ETHFI", quote_asset="USDT",
        last_price=1.002, quote_volume_24h=36_000_000.0, market_cap_usd=665_000_000.0,
        rsi14_1d=50.0, perf_30d_pct=3.8, price_range_90bar_pct=40.0,
    )
    return ConfirmedCandidate(candidate=cand, reads={"4h": TimeframeRead(timeframe="4h", df=_signal_frame())})


def test_format_signal_report_contains_all_seven_sections_and_no_signal_disclaimer():
    cfg = ScreenConfig()
    confirmed = _confirmed()
    verdict = evaluate_candidate(confirmed, cfg, decision_timeframe="4h")
    assert verdict.is_signal, verdict.reject_reason

    decision_df = confirmed.reads["4h"].df
    trade_plan = build_trade_plan(confirmed.candidate.symbol, decision_df, cfg, conviction=verdict.conviction or "low")

    snapshot = MarketSnapshot(
        exchange="binance", quote_asset="USDT", pair_count=120,
        total_quote_volume_24h=5_000_000_000.0,
        volume_weighted_change_pct_24h=1.2, btc_quote_volume_share_pct=42.0,
    )
    market_timing = MarketTimingResult(
        snapshot=snapshot, regime=MarketRegime.ACTIVE_RANGE, strict_mode=False,
    )

    report = format_signal_report("binance", market_timing, verdict, trade_plan)

    for marker in ["أولا،", "ثانيا،", "ثالثا،", "رابعا،", "خامسا،", "سادسا،", "سابعا،"]:
        assert marker in report
    assert "ETHFI" in report
    assert "نصيحة مالية" in report  # the honesty-gate disclaimer must always be present


def test_format_signal_report_handles_missing_trade_plan_gracefully():
    cfg = ScreenConfig()
    confirmed = _confirmed()
    verdict = evaluate_candidate(confirmed, cfg, decision_timeframe="4h")
    assert verdict.is_signal

    snapshot = MarketSnapshot(
        exchange="okx", quote_asset="USDT", pair_count=80,
        total_quote_volume_24h=2_000_000_000.0,
        volume_weighted_change_pct_24h=-0.5, btc_quote_volume_share_pct=38.0,
    )
    market_timing = MarketTimingResult(
        snapshot=snapshot, regime=MarketRegime.ACTIVE_RANGE, strict_mode=False,
    )

    report = format_signal_report("okx", market_timing, verdict, trade_plan=None)
    assert "تعذر بناء خطة صفقة موثوقة" in report
