"""Step 5 (decision) and step 7 (honesty / rejection gate).

The decision frame is 4h by default (cross-checked against 1h and
1D), matching the methodology. A real entry signal requires BOTH:

- genuine quiet compression right now: low/falling ADX, Bollinger
  Bands sitting in the tightest part of their own recent range,
  neutral RSI, and OBV flat-or-rising despite the flat price -- and
- an early, concrete trigger that compression is starting to break:
  ADX turning up, BB width starting to expand off its squeeze low, or
  OBV printing a fresh local high with price beginning to follow.

A coin that is merely quiet, with no trigger yet, is correctly a
"no signal" case, not a weak signal -- the whole point of this
project is to not invent urgency that isn't there yet (step 7).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from .config import ScreenConfig
from .multi_timeframe import ConfirmedCandidate, TimeframeRead


@dataclass
class CompressionRead:
    adx_low: bool
    adx_value: float
    bb_squeezed: bool
    bb_width_percentile: float
    rsi_neutral: bool
    rsi_value: float
    obv_flat_or_rising: bool
    obv_slope_pct: float

    @property
    def criteria_met(self) -> int:
        return sum([self.adx_low, self.bb_squeezed, self.rsi_neutral, self.obv_flat_or_rising])


@dataclass
class TriggerRead:
    adx_rising: bool
    adx_delta: float
    bb_expanding: bool
    bb_expansion_pct: float
    obv_breakout: bool

    @property
    def any_trigger(self) -> bool:
        return self.adx_rising or self.bb_expanding or self.obv_breakout

    @property
    def reasons(self) -> List[str]:
        out = []
        if self.adx_rising:
            out.append(f"ADX بدأ الارتفاع (+{self.adx_delta:.1f})")
        if self.bb_expanding:
            out.append(f"نطاقات بولينجر بدأت تتوسع بعد انضغاطها (+{self.bb_expansion_pct:.0f}%)")
        if self.obv_breakout:
            out.append("OBV يصنع قمة محلية جديدة مع بدء متابعة السعر له")
        return out


@dataclass
class SignalVerdict:
    confirmed: ConfirmedCandidate
    decision_timeframe: str
    compression: CompressionRead
    trigger: TriggerRead
    is_signal: bool
    conviction: Optional[str] = None  # "low" | "medium" | "high"
    reject_reason: Optional[str] = None


def _bb_width_percentile(series: pd.Series, window: int = 100) -> float:
    tail = series.tail(window).dropna()
    if len(tail) < 20:
        return float("nan")
    current = tail.iloc[-1]
    return float((tail <= current).mean())  # fraction of trailing bars at-or-below current width


def _read_compression(df: pd.DataFrame, cfg: ScreenConfig) -> CompressionRead:
    last = df.iloc[-1]
    adx_val = float(last["adx"]) if last["adx"] == last["adx"] else float("nan")
    adx_low = adx_val == adx_val and adx_val <= cfg.low_adx_threshold

    bb_pct = _bb_width_percentile(df["bb_width_pct"])
    bb_squeezed = bb_pct == bb_pct and bb_pct <= cfg.bb_squeeze_percentile

    rsi_val = float(last["rsi14"]) if last["rsi14"] == last["rsi14"] else float("nan")
    rsi_lo, rsi_hi = cfg.rsi_band
    rsi_neutral = rsi_val == rsi_val and rsi_lo <= rsi_val <= rsi_hi

    obv_tail = df["obv"].tail(cfg.obv_lookback_bars).dropna()
    if len(obv_tail) >= 2 and obv_tail.iloc[0] != 0:
        obv_slope_pct = (obv_tail.iloc[-1] - obv_tail.iloc[0]) / abs(obv_tail.iloc[0]) * 100.0
    elif len(obv_tail) >= 2:
        obv_slope_pct = 0.0 if obv_tail.iloc[-1] == obv_tail.iloc[0] else float("inf")
    else:
        obv_slope_pct = float("nan")
    obv_ok = obv_slope_pct == obv_slope_pct and obv_slope_pct >= cfg.obv_rising_min_pct

    return CompressionRead(
        adx_low=adx_low, adx_value=adx_val,
        bb_squeezed=bb_squeezed, bb_width_percentile=bb_pct,
        rsi_neutral=rsi_neutral, rsi_value=rsi_val,
        obv_flat_or_rising=obv_ok, obv_slope_pct=obv_slope_pct,
    )


def _read_trigger(df: pd.DataFrame, cfg: ScreenConfig, lookback: int = 5) -> TriggerRead:
    adx = df["adx"].dropna()
    adx_delta = float(adx.iloc[-1] - adx.iloc[-lookback]) if len(adx) > lookback else float("nan")
    adx_rising = adx_delta == adx_delta and adx_delta >= cfg.adx_rising_min_delta

    width = df["bb_width_pct"].dropna()
    squeeze_window = width.tail(50)
    if len(squeeze_window) >= 10:
        squeeze_low = squeeze_window.min()
        current = width.iloc[-1]
        bb_expansion_pct = ((current - squeeze_low) / squeeze_low * 100.0) if squeeze_low > 0 else 0.0
        bb_expanding = bb_expansion_pct >= cfg.bb_expansion_min_pct
    else:
        bb_expansion_pct = float("nan")
        bb_expanding = False

    obv = df["obv"].dropna()
    obv_window = obv.tail(cfg.obv_lookback_bars)
    close_window = df["close"].tail(3)
    obv_breakout = False
    if len(obv_window) >= 5 and len(close_window) >= 2:
        is_new_high = obv_window.iloc[-1] >= obv_window.max()
        price_following = close_window.iloc[-1] >= close_window.iloc[0]
        obv_breakout = bool(is_new_high and price_following)

    return TriggerRead(
        adx_rising=adx_rising, adx_delta=adx_delta if adx_delta == adx_delta else 0.0,
        bb_expanding=bb_expanding, bb_expansion_pct=bb_expansion_pct if bb_expansion_pct == bb_expansion_pct else 0.0,
        obv_breakout=obv_breakout,
    )


def evaluate_candidate(
    confirmed: ConfirmedCandidate,
    cfg: ScreenConfig,
    decision_timeframe: str = "4h",
) -> SignalVerdict:
    tf_read: Optional[TimeframeRead] = confirmed.reads.get(decision_timeframe)
    if tf_read is None:
        return SignalVerdict(
            confirmed=confirmed, decision_timeframe=decision_timeframe,
            compression=CompressionRead(False, float("nan"), False, float("nan"), False, float("nan"), False, float("nan")),
            trigger=TriggerRead(False, 0.0, False, 0.0, False),
            is_signal=False,
            reject_reason=f"no {decision_timeframe} data available for this candidate",
        )

    compression = _read_compression(tf_read.df, cfg)
    trigger = _read_trigger(tf_read.df, cfg)

    # ADX, Bollinger squeeze, and RSI are the three load-bearing markers of
    # genuine quiet compression -- all three are mandatory. OBV is the
    # noisiest of the four on altcoins, so it alone is allowed to be
    # "unavailable/unreliable" without killing an otherwise clean setup.
    # A concrete trigger is required on top of that to call it a signal.
    compressed_enough = compression.adx_low and compression.bb_squeezed and compression.rsi_neutral
    is_signal = compressed_enough and trigger.any_trigger

    conviction = None
    reject_reason = None
    if is_signal:
        aligned_timeframes = sum(
            1 for tf, read in confirmed.reads.items()
            if read.df["rsi14"].iloc[-1] == read.df["rsi14"].iloc[-1]
            and cfg.rsi_band[0] - 5 <= read.df["rsi14"].iloc[-1] <= cfg.rsi_band[1] + 10
        )
        trigger_count = sum([trigger.adx_rising, trigger.bb_expanding, trigger.obv_breakout])
        score = compression.criteria_met + trigger_count + aligned_timeframes
        if score >= 8:
            conviction = "high"
        elif score >= 6:
            conviction = "medium"
        else:
            conviction = "low"
    else:
        missing = []
        if not compression.adx_low:
            missing.append(f"ADX غير منخفض بما يكفي ({compression.adx_value:.1f})")
        if not compression.bb_squeezed:
            missing.append("نطاقات بولينجر ليست في منطقة الانضغاط")
        if not compression.rsi_neutral:
            missing.append(f"RSI خارج النطاق المحايد ({compression.rsi_value:.1f})")
        if not compression.obv_flat_or_rising:
            missing.append("OBV يتراجع رغم تسطح السعر")
        if compressed_enough and not trigger.any_trigger:
            reject_reason = "تراكم هادئ محتمل لكن بلا زناد مبكر فعلي حتى الآن"
        else:
            reject_reason = "؛ ".join(missing) if missing else "لم تتحقق معايير التراكم الهادئ"

    return SignalVerdict(
        confirmed=confirmed, decision_timeframe=decision_timeframe,
        compression=compression, trigger=trigger,
        is_signal=is_signal, conviction=conviction, reject_reason=reject_reason,
    )


def pick_best_signal(verdicts: List[SignalVerdict]) -> Optional[SignalVerdict]:
    """Step 7's honesty gate lives here as much as in evaluate_candidate:

    if nothing qualifies, this returns None and the caller must emit the
    terse "no signal" line -- never the full report format.
    """
    signals = [v for v in verdicts if v.is_signal]
    if not signals:
        return None
    rank = {"high": 2, "medium": 1, "low": 0}
    signals.sort(key=lambda v: rank.get(v.conviction or "low", 0), reverse=True)
    return signals[0]
