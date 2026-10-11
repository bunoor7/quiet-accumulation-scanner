"""Technical indicators computed locally from OHLCV bars.

No TA library dependency on purpose: every formula here is the
standard, textbook definition (Wilder smoothing where Wilder defined
the indicator -- RSI, ATR, ADX, +DI/-DI), so the numbers can be
cross-checked against any charting platform by hand. Every function
takes a DataFrame with at least ``high``, ``low``, ``close`` (and
``volume`` where relevant) and returns a ``pandas.Series`` aligned to
the same index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # Wilder's edge case: if avg_loss is 0 and avg_gain > 0, RSI is 100.
    out = out.where(avg_loss != 0.0, 100.0)
    return out


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    a = df["high"] - df["low"]
    b = (df["high"] - prev_close).abs()
    c = (df["low"] - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder's ADX, plus the +DI/-DI lines it is built from."""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    tr = true_range(df)
    atr_sm = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_dm_sm = plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    minus_dm_sm = minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * (plus_dm_sm / atr_sm.replace(0.0, np.nan))
    minus_di = 100.0 * (minus_dm_sm / atr_sm.replace(0.0, np.nan))

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_line = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx_line})


def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(close, period)
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = upper - lower
    width_pct_of_price = (width / mid.replace(0.0, np.nan)) * 100.0
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
         "bb_width": width, "bb_width_pct": width_pct_of_price}
    )


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    low_n = df["low"].rolling(window=k_period, min_periods=k_period).min()
    high_n = df["high"].rolling(window=k_period, min_periods=k_period).max()
    rng = (high_n - low_n).replace(0.0, np.nan)
    percent_k = 100.0 * (df["close"] - low_n) / rng
    percent_d = percent_k.rolling(window=d_period, min_periods=d_period).mean()
    return pd.DataFrame({"stoch_k": percent_k, "stoch_d": percent_d})


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff().fillna(0.0))
    signed_volume = direction * df["volume"]
    return signed_volume.cumsum()


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Append every indicator this project needs as columns on a copy of ``df``."""
    out = df.copy()
    out["rsi14"] = rsi(out["close"], 14)
    macd_df = macd(out["close"]).rename(columns={"macd": "macd", "signal": "macd_signal", "hist": "macd_hist"})
    out = out.join(macd_df)
    out["atr14"] = atr(out, 14)
    adx_df = adx(out, 14)
    out = out.join(adx_df)
    bb_df = bollinger_bands(out["close"], 20, 2.0)
    out = out.join(bb_df)
    stoch_df = stochastic(out, 14, 3)
    out = out.join(stoch_df)
    out["ema9"] = ema(out["close"], 9)
    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["ema200"] = ema(out["close"], 200)
    out["sma200"] = sma(out["close"], 200)
    out["obv"] = obv(out)
    return out
