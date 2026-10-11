"""Entry point: python -m accscan.cli --exchange binance

Runs the full seven-step pipeline once and prints either the one-line
"no signal" message or the full report. Nothing here talks to a
broker or places an order -- this is read-only market analysis.
"""
from __future__ import annotations

import argparse
import logging
import sys

from .config import ScreenConfig
from .exchanges import get_exchange_client
from .market_cap import CoinGeckoMarketCapProvider, NullMarketCapProvider
from .market_timing import assess_market_timing
from .multi_timeframe import confirm_candidates, surviving
from .report import format_no_signal, format_signal_report
from .screener import build_universe, score_candidates, shortlist_accumulation_candidates
from .signal_engine import evaluate_candidate, pick_best_signal
from .trade_plan import build_trade_plan


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="accscan",
        description="Quiet-accumulation screener for liquid altcoins, read directly from Binance or OKX.",
    )
    p.add_argument("--exchange", choices=["binance", "okx"], default="binance")
    p.add_argument("--quote", default="USDT", help="Quote asset to scan against (default USDT)")
    p.add_argument(
        "--market-cap",
        choices=["none", "coingecko"],
        default="none",
        help="How to enforce the market-cap floor from the methodology. "
             "'none' (default) relies on 24h quote volume alone, honestly skipping "
             "the market-cap rule rather than approximating it. 'coingecko' calls "
             "CoinGecko's free /coins/markets endpoint to enforce it for real.",
    )
    p.add_argument("--decision-timeframe", default="4h", choices=["1h", "4h", "1D"])
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def run(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("accscan.cli")

    cfg = ScreenConfig(quote_asset=args.quote)
    client = get_exchange_client(
        args.exchange,
        timeout_s=cfg.http_timeout_s,
        retries=cfg.http_retries,
        retry_wait_s=cfg.http_retry_wait_s,
    )

    # Step 1
    try:
        market_timing = assess_market_timing(client, cfg)
    except Exception as exc:  # noqa: BLE001
        print(format_no_signal(f"تعذر الوصول إلى بيانات السوق الإجمالية من {args.exchange} ({exc})"))
        return 1
    log.info(
        "market timing: regime=%s weighted_change=%.2f%% btc_share=%.1f%%",
        market_timing.regime, market_timing.snapshot.volume_weighted_change_pct_24h,
        market_timing.snapshot.btc_quote_volume_share_pct,
    )

    # Step 2
    universe = build_universe(client, cfg)
    if len(universe) < cfg.shortlist_min:
        print(format_no_signal("عدد العملات المستوفية لشرط السيولة الأساسي غير كاف لهذا التشغيل"))
        return 0

    market_cap_provider = (
        CoinGeckoMarketCapProvider() if args.market_cap == "coingecko" else NullMarketCapProvider()
    )
    scored = score_candidates(client, universe, cfg, market_cap_provider=market_cap_provider)
    shortlist = shortlist_accumulation_candidates(scored, cfg, strict_mode=market_timing.strict_mode)
    log.info("shortlist: %d candidates after RSI/performance filter", len(shortlist))

    if len(shortlist) < 1:
        print(format_no_signal("غياب انضغاط حقيقي، لا توجد عملة تجمع بين RSI محايد وأداء شهري شبه مسطح"))
        return 0

    # Step 3
    confirmed = confirm_candidates(client, shortlist, market_timing, cfg)
    survivors = surviving(confirmed)
    log.info("survivors after multi-timeframe confirmation: %d", len(survivors))
    for c in confirmed:
        if not c.survived:
            log.info("excluded %s: %s", c.candidate.symbol, c.excluded_reason)

    if not survivors:
        print(format_no_signal("كل المرشحين إما ممتد فعلا على الأطر الأعلى أو حركته مجرد انعكاس لحركة السوق العامة"))
        return 0

    # Step 4 + 5 (evaluate_candidate reads the already-fetched 4h/1h/1D frames)
    verdicts = [evaluate_candidate(c, cfg, decision_timeframe=args.decision_timeframe) for c in survivors]
    for v in verdicts:
        if not v.is_signal:
            log.info("no signal for %s: %s", v.confirmed.candidate.symbol, v.reject_reason)

    # Step 7 (honesty gate)
    best = pick_best_signal(verdicts)
    if best is None:
        print(format_no_signal("لا توجد عملة تجمع بين تراكم هادئ حقيقي وزناد مبكر موثوق في هذا التشغيل"))
        return 0

    # Step 6
    decision_df = best.confirmed.reads[args.decision_timeframe].df
    trade_plan = build_trade_plan(
        best.confirmed.candidate.symbol, decision_df, cfg, conviction=best.conviction or "low"
    )

    print(format_signal_report(args.exchange, market_timing, best, trade_plan))
    return 0


if __name__ == "__main__":
    sys.exit(run())
