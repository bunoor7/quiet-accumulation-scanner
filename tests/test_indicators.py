from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from accscan import indicators as ind


def _flat_df(n=60, price=100.0, vol=1000.0):
    idx = pd.date_range("2026-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame(
        {
            "open": [price] * n,
            "high": [price] * n,
            "low": [price] * n,
            "close": [price] * n,
            "volume": [vol] * n,
        },
        index=idx,
    )


def _monotonic_up_df(n=40, start=100.0, step=1.0, vol=1000.0):
    closes = [start + i * step for i in range(n)]
    idx = pd.date_range("2026-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.1 for c in closes],
            "low": [c - 0.1 for c in closes],
            "close": closes,
            "volume": [vol] * n,
        },
        index=idx,
    )


def test_rsi_is_100_for_strictly_rising_prices():
    df = _monotonic_up_df(n=40)
    rsi = ind.rsi(df["close"], period=14)
    assert rsi.dropna().round(2).eq(100.0).all()


def test_rsi_is_0_for_strictly_falling_prices():
    df = _monotonic_up_df(n=40, start=200.0, step=-1.0)
    rsi = ind.rsi(df["close"], period=14)
    assert rsi.dropna().round(2).eq(0.0).all()


def test_rsi_bounded_0_100_on_noisy_data():
    rng = np.random.default_rng(42)
    n = 300
    idx = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame(
        {
            "open": close, "high": close + 1, "low": close - 1,
            "close": close, "volume": np.abs(rng.normal(1000, 100, n)),
        },
        index=idx,
    )
    rsi = ind.rsi(df["close"], 14)
    assert rsi.dropna().between(0, 100).all()


def test_obv_on_rising_prices_equals_cumulative_volume():
    df = _monotonic_up_df(n=10, step=1.0, vol=50.0)
    obv = ind.obv(df)
    # first bar has no prior close -> diff is NaN -> treated as 0 by fillna(0)
    expected = pd.Series([0.0] + [50.0] * 9, index=df.index).cumsum()
    pd.testing.assert_series_equal(obv, expected, check_names=False)


def test_obv_flat_price_never_moves():
    df = _flat_df(n=20, vol=123.0)
    obv = ind.obv(df)
    assert (obv == 0.0).all()


def test_adx_near_zero_on_flat_price():
    df = _flat_df(n=60)
    out = ind.adx(df, period=14)
    # No directional movement at all -> ADX should be ~0 (or NaN from 0/0 DX),
    # never a large number.
    tail = out["adx"].dropna()
    if len(tail):
        assert (tail.fillna(0) < 1.0).all()


def test_bollinger_bands_width_zero_on_flat_price():
    df = _flat_df(n=40)
    bb = ind.bollinger_bands(df["close"], period=20)
    assert bb["bb_width"].dropna().eq(0.0).all()


def test_macd_hist_sign_matches_trend_direction():
    df = _monotonic_up_df(n=80, step=0.5)
    out = ind.macd(df["close"])
    # Once warmed up, a steadily rising series should show a non-negative
    # MACD histogram (fast EMA pulling away from slow EMA).
    tail = out["hist"].dropna().tail(10)
    assert (tail >= -1e-9).all()


def test_stochastic_bounded_0_100():
    rng = np.random.default_rng(1)
    n = 100
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    close = 50 + np.cumsum(rng.normal(0, 0.5, n))
    df = pd.DataFrame(
        {
            "open": close, "high": close + np.abs(rng.normal(0.2, 0.1, n)),
            "low": close - np.abs(rng.normal(0.2, 0.1, n)), "close": close,
            "volume": np.abs(rng.normal(500, 50, n)),
        },
        index=idx,
    )
    stoch = ind.stochastic(df, 14, 3)
    assert stoch["stoch_k"].dropna().between(0, 100).all()
    assert stoch["stoch_d"].dropna().between(0, 100).all()


def test_compute_all_adds_every_expected_column():
    rng = np.random.default_rng(5)
    n = 260
    idx = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    close = 10 + np.cumsum(rng.normal(0, 0.05, n))
    df = pd.DataFrame(
        {
            "open": close, "high": close + 0.05, "low": close - 0.05,
            "close": close, "volume": np.abs(rng.normal(1000, 50, n)),
        },
        index=idx,
    )
    out = ind.compute_all(df)
    for col in [
        "rsi14", "macd", "macd_signal", "macd_hist", "atr14",
        "plus_di", "minus_di", "adx", "bb_mid", "bb_upper", "bb_lower",
        "bb_width", "bb_width_pct", "stoch_k", "stoch_d",
        "ema9", "ema20", "ema50", "ema200", "sma200", "obv",
    ]:
        assert col in out.columns, f"missing column {col}"
    assert not out[["rsi14", "adx", "bb_width_pct"]].iloc[-1].isna().any()
