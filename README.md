# quiet-accumulation-scanner

A read-only screener that looks for liquid altcoins sitting in a genuine
**quiet accumulation** phase with an early breakout trigger, reading live
market data directly from **Binance's** or **OKX's public spot REST
endpoints** — no API key, no authenticated trading, no order placement.
Every number is computed locally in Python; nothing is bought, sold, or
staked by this project.

It is a from-scratch reimplementation of a 7-step accumulation-screening
methodology, originally run against third-party market-data tools, rebuilt
here to depend on nothing but each exchange's own free, keyless public API
plus local technical-indicator math.

## What "quiet accumulation" means here

The methodology looks for a coin that is quiet *and* just starting to move,
not one that has already run. It rejects "maybe" outright: if the evidence
is incomplete or ambiguous, the tool reports **no signal**, not a weak one.

1. **Market timing.** Is the market as a whole quiet, actively ranging, or
   in a sharp directional move? This uses a volume-weighted average 24h
   change across every pair on the chosen exchange, plus BTC's share of
   quote volume as a liquidity-weighted dominance proxy — honestly labeled
   as an *exchange-liquidity proxy*, never as "market cap", since it isn't
   built from real circulating-supply data the way CoinGecko's figure is.
2. **Universe and accumulation shortlist.** Filter out majors (BTC, ETH,
   wrapped variants), stablecoins/pegged assets, and anything below a
   minimum 24h quote-volume floor. Score what remains by daily RSI(14) and
   ~30-day performance, and keep only names sitting in a neutral RSI band
   with a roughly flat month — the opposite of "already pumped".
3. **Multi-timeframe confirmation.** Pull 1h/4h/1D bars for every
   shortlisted name and exclude anything already extended far above its 1D
   EMA200 (it already ran), or — only when the market itself is in a sharp
   trend — anything whose recent move is indistinguishable from broad
   market beta rather than coin-specific behavior.
4. **Technical read on the decision frame** (4h by default). Full
   indicator set: RSI, MACD, ATR, ADX/+DI/-DI, Bollinger Bands, Stochastic,
   OBV, EMAs/SMA.
5. **Signal decision.** A real entry signal requires *both* genuine
   compression right now (low ADX, Bollinger Bands in the tightest part of
   their own recent range, neutral RSI — all three mandatory) *and* a
   concrete, early trigger that the compression is starting to break (ADX
   turning up, Bollinger width expanding off its own squeeze low, or OBV
   printing a fresh local high with price beginning to follow). OBV alone
   is allowed to be the "unreliable" fourth compression criterion, since
   it is the noisiest of the four on altcoins — but it is never enough on
   its own.
6. **Trade plan.** Only built for a confirmed signal: a stop from whichever
   of (ATR multiple, nearest real swing support) is tighter and still
   protective, one or two resistance-based targets (or a measured-move
   fallback when no resistance is visible in the fetched window), and each
   target's reward:risk ratio.
7. **Honesty gate.** If nothing across the whole shortlist clears both the
   compression and trigger bars, the tool prints a single short line and
   stops — it never pads out a report around a borderline case. Silence is
   the correct behavior here, not a failure of the screen.

## Installation

```bash
git clone <this-repo-url>
cd quiet-accumulation-scanner
pip install -r requirements.txt
```

Requires Python 3.10+. No API keys, `.env` file, or account of any kind —
every endpoint used is Binance's and OKX's free public market-data API.

## Usage

```bash
# Scan Binance USDT pairs (default)
PYTHONPATH=src python3 -m accscan.cli --exchange binance

# Scan OKX instead
PYTHONPATH=src python3 -m accscan.cli --exchange okx

# Verbose logging (shows why candidates were excluded at each step)
PYTHONPATH=src python3 -m accscan.cli --exchange binance -v

# Use a different decision timeframe for step 4/5 (default: 4h)
PYTHONPATH=src python3 -m accscan.cli --exchange binance --decision-timeframe 1h

# Enforce the methodology's ~$150M market-cap floor for real via
# CoinGecko's free /coins/markets endpoint, instead of relying on 24h
# quote volume alone (the honest default — see "Market cap" below)
PYTHONPATH=src python3 -m accscan.cli --exchange binance --market-cap coingecko
```

(If you installed the package with `pip install -e .`, the `accscan`
console command is also available directly, without `PYTHONPATH=src`.)

Output is Arabic, in the same two shapes the original methodology uses:
a single short "no signal" line most of the time, or a full seven-section
report on the rare run where a real signal clears every gate.

## Market cap

Binance and OKX's public endpoints expose price and volume, not
circulating supply or market cap. By default this project does **not**
pretend 24h quote volume is market cap — it honestly skips that specific
rule from the methodology and relies on the volume floor alone. Pass
`--market-cap coingecko` to enforce the real $150M floor via CoinGecko's
free, keyless `/coins/markets` endpoint; if that call fails for any reason,
it degrades the same way the default does (skipped, not a crash), and this
is logged rather than silently assumed.

## Project layout

```
src/accscan/
  config.py          every tunable threshold in one place
  exchanges/         Binance + OKX adapters behind one shared interface
  indicators.py      RSI/MACD/ATR/ADX/Bollinger/Stochastic/OBV, from scratch
  market_timing.py   step 1
  screener.py        step 2
  multi_timeframe.py step 3
  signal_engine.py   steps 4-5 and the step-7 honesty gate
  trade_plan.py      step 6
  report.py          Arabic output formatting
  cli.py             orchestrates all seven steps
tests/               fully offline: every indicator and decision-rule is
                      tested against hand-built or directly-constructed
                      data, and every exchange adapter against a mocked
                      HTTP response shaped like the real API -- no test
                      in this repository ever makes a network call.
```

## Testing

```bash
pip install -r requirements.txt pytest
PYTHONPATH=src pytest tests/ -v
```

The full suite (indicators, the signal-decision logic, both exchange
adapters' response parsing, market timing, the screener's filtering/
ranking, and trade-plan construction) runs offline against synthetic or
mocked data — no network access is required or used by any test, and the
CI workflow in `.github/workflows/ci.yml` runs it on Python 3.10, 3.11 and
3.12 on every push and pull request.

**A note on where this was built:** this project was written and
offline-tested inside a sandboxed cloud environment whose network egress
policy blocks direct connections to `api.binance.com`, `www.okx.com`, and
`api.coingecko.com`. That means the full offline test suite above is real,
passing evidence that the parsing and decision logic are correct against
realistic sample data -- but this project has **not** been run end-to-end
against the live Binance/OKX APIs from that environment. Please run
`python3 -m accscan.cli --exchange binance -v` yourself once you have this
checked out somewhere with normal internet access, to confirm live
connectivity and sanity-check a real run's output, before relying on it.

## Disclaimer

This is a technical-analysis tool built entirely on public market data. It
is not financial advice, not a recommendation to buy or sell anything, and
not a guarantee of any outcome. Markets can and do behave in ways that
invalidate a "quiet accumulation" read after the fact. Nothing here places
an order, holds a position, or connects to any account — do your own
research and risk management before acting on anything it reports.

## License

MIT — see [LICENSE](LICENSE).
