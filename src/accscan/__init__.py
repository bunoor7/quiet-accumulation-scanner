"""accscan: a quiet-accumulation screener for liquid altcoins.

Reads public market data directly from Binance and OKX (no API keys
required) and runs the same seven-step methodology described in
``README.md``: market timing, accumulation screening, multi-timeframe
confirmation, full technical read, entry-signal decision, trade-plan
construction, and an honesty/rejection gate that refuses to invent a
signal when the evidence does not clearly support one.
"""

__version__ = "0.1.0"
