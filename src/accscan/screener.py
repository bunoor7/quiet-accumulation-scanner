"""Step 2: accumulation shortlist.

Universe -> liquidity/major/stable filter -> per-symbol RSI(14) and
~30-day performance on the daily frame -> keep only names sitting in
the neutral RSI band with a roughly flat month, and narrow that down
to ``shortlist_min``..``shortlist_max`` names for the step-3 deep dive.

Network calls (one klines fetch per surviving symbol) run through a
small thread pool since they are independent, unauthenticated GET
requests -- see ``README.md`` for why that's fine against Binance/OKX
public endpoints at a modest concurrency.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional

from . import indicators as ind
from .config import ScreenConfig
from .exchanges.base import ExchangeClient, Ticker24h
from .market_cap import MarketCapProvider, NullMarketCapProvider

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    symbol: str
    base_asset: str
    quote_asset: str
    last_price: float
    quote_volume_24h: float
    market_cap_usd: Optional[float]  # None = unknown, filter not enforced
    rsi14_1d: float
    perf_30d_pct: float
    price_range_90bar_pct: float  # (max-min)/min over the fetched window -- pegged-asset check


def _is_probably_pegged(df_1d, cfg: ScreenConfig) -> bool:
    window = df_1d["close"].tail(90)
    if len(window) < 10:
        return False
    lo, hi = window.min(), window.max()
    if lo <= 0:
        return True
    range_pct = (hi - lo) / lo * 100.0
    return range_pct < cfg.stable_price_range_pct


def _score_one_symbol(client: ExchangeClient, ticker: Ticker24h, cfg: ScreenConfig) -> Optional[Candidate]:
    try:
        df = client.get_klines(ticker.symbol, "1D", limit=100)
    except Exception as exc:  # noqa: BLE001
        log.warning("skip %s: klines fetch failed (%s)", ticker.symbol, exc)
        return None

    if len(df) < 31:
        return None
    if _is_probably_pegged(df, cfg):
        return None

    rsi_series = ind.rsi(df["close"], 14)
    rsi_now = float(rsi_series.iloc[-1])
    if rsi_now != rsi_now:  # NaN check without importing math
        return None

    close_now = float(df["close"].iloc[-1])
    close_30_ago = float(df["close"].iloc[-31])
    perf_30d = ((close_now - close_30_ago) / close_30_ago) * 100.0 if close_30_ago > 0 else 0.0

    window = df["close"].tail(90)
    range_pct = ((window.max() - window.min()) / window.min() * 100.0) if window.min() > 0 else 0.0

    return Candidate(
        symbol=ticker.symbol,
        base_asset=ticker.base_asset,
        quote_asset=ticker.quote_asset,
        last_price=ticker.last_price,
        quote_volume_24h=ticker.quote_volume_24h,
        market_cap_usd=None,
        rsi14_1d=rsi_now,
        perf_30d_pct=perf_30d,
        price_range_90bar_pct=range_pct,
    )


def build_universe(
    client: ExchangeClient,
    cfg: ScreenConfig,
) -> List[Ticker24h]:
    tickers = client.get_tickers_24h(quote_asset=cfg.quote_asset)
    excluded = {b.upper() for b in cfg.exclude_bases} | {b.upper() for b in cfg.known_stable_bases}
    out = [
        t for t in tickers
        if t.base_asset.upper() not in excluded
        and t.quote_volume_24h >= cfg.min_24h_quote_volume_usd
    ]
    out.sort(key=lambda t: t.quote_volume_24h, reverse=True)
    if len(out) > cfg.max_universe_size:
        log.info(
            "universe has %d symbols after filtering, capping to the top %d by volume",
            len(out), cfg.max_universe_size,
        )
        out = out[: cfg.max_universe_size]
    return out


def score_candidates(
    client: ExchangeClient,
    universe: List[Ticker24h],
    cfg: ScreenConfig,
    market_cap_provider: Optional[MarketCapProvider] = None,
    max_workers: int = 8,
) -> List[Candidate]:
    provider = market_cap_provider or NullMarketCapProvider()
    caps = provider.get_market_caps_usd(t.base_asset for t in universe)

    scored: List[Candidate] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_score_one_symbol, client, t, cfg): t for t in universe}
        for fut in as_completed(futures):
            cand = fut.result()
            if cand is None:
                continue
            cap = caps.get(cand.base_asset.upper())
            if cap is not None:
                cand.market_cap_usd = cap
                if cap < cfg.min_market_cap_usd:
                    continue  # enforced only when we actually know the cap
            scored.append(cand)
    return scored


def shortlist_accumulation_candidates(
    scored: List[Candidate],
    cfg: ScreenConfig,
    strict_mode: bool = False,
) -> List[Candidate]:
    rsi_lo, rsi_hi = cfg.strict_rsi_band if strict_mode else cfg.rsi_band
    perf_lo, perf_hi = cfg.monthly_perf_band_pct

    in_band = [
        c for c in scored
        if rsi_lo <= c.rsi14_1d <= rsi_hi and perf_lo <= c.perf_30d_pct <= perf_hi
    ]
    # Rank the flattest, most neutral names first: smallest absolute monthly
    # move, then closest RSI to the midpoint of the band -- both proxies for
    # "truly quiet" rather than "happens to currently sit in the band".
    mid_rsi = (rsi_lo + rsi_hi) / 2.0
    in_band.sort(key=lambda c: (abs(c.perf_30d_pct), abs(c.rsi14_1d - mid_rsi)))

    return in_band[: cfg.shortlist_max]
