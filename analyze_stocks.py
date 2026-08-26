"""
Stock universe collector and analyzer.

Pulls daily history for individual stocks via yfinance, runs the null engine
on each, and writes results the dashboard reads.

    .venv/bin/python3 analyze_stocks.py AAPL MSFT NVDA TSLA GOOG AMZN META
    .venv/bin/python3 analyze_stocks.py --watchlist default
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from engine.core.storage import Store
from engine.nulls.compare import compare
from engine.nulls.models import fit_garch11
from engine.nulls.statistics import (
    ann_vol, excess_kurtosis, sharpe, acf_abs, acf_signed,
    same_band_rate, realised_vol, vol_bands,
)

# A sensible default universe: mega-caps, sector leaders, high-retail-interest.
DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA",
    "JPM", "V", "UNH", "XOM", "LLY", "AVGO", "COST",
    "SPY", "QQQ", "IWM", "GLD", "TLT",
]


def fetch_stock(symbol: str, period: str = "5y") -> pd.DataFrame | None:
    """Pull daily OHLCV from Yahoo Finance."""
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        df = t.history(period=period, auto_adjust=True)
        if df is None or len(df) < 100:
            print(f"  {symbol}: insufficient data ({len(df) if df is not None else 0} rows)")
            return None
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df["symbol"] = symbol
        df["event_time"] = pd.to_datetime(df["date"])
        df["knowable_at"] = df["event_time"]
        df["retrieved_at"] = pd.Timestamp.now(tz="UTC")
        return df
    except Exception as e:
        print(f"  {symbol}: fetch failed — {type(e).__name__}: {e}")
        return None


def analyze_one(symbol: str, r: np.ndarray, n_sim: int = 300) -> dict:
    """Run the full null-comparison engine on one stock's returns."""
    p = fit_garch11(r)
    rep = compare(r, n_sim=n_sim, seed=abs(hash(symbol)) % 100_000)
    survivors = [n for n, s in rep.stats.items() if s.survives()]

    # Trailing stats
    rv = realised_vol(r, 20)
    current_vol = float(rv[-1]) if not np.isnan(rv[-1]) else None
    vol_20d = float(np.std(r[-20:], ddof=1) * np.sqrt(252)) if len(r) >= 20 else None
    vol_60d = float(np.std(r[-60:], ddof=1) * np.sqrt(252)) if len(r) >= 60 else None

    return {
        "symbol": symbol,
        "n_obs": int(r.size),
        "ann_return": float(np.mean(r) * 252),
        "ann_vol": float(ann_vol(r)),
        "sharpe": float(sharpe(r)),
        "excess_kurtosis": float(excess_kurtosis(r)),
        "acf_abs_lag1": float(acf_abs(r, 1)),
        "acf_signed_lag1": float(acf_signed(r, 1)),
        "same_band_rate": float(same_band_rate(r, 20, 5, 5)),
        "current_vol_20d": vol_20d,
        "current_vol_60d": vol_60d,
        "vol_ratio": round(vol_20d / vol_60d, 2) if vol_20d and vol_60d else None,
        "garch": {
            "alpha": round(p.alpha, 4),
            "beta": round(p.beta, 4),
            "persistence": round(p.persistence, 5),
            "half_life": round(p.half_life(), 1),
        },
        "survives_all_nulls": survivors,
        "n_survivors": len(survivors),
        "statistics": {
            name: {
                "real": round(s.real, 6),
                "p_garch": round(s.nulls["garch"].p_two_sided, 4),
                "p_shuffle": round(s.nulls["shuffled"].p_two_sided, 4),
            }
            for name, s in rep.stats.items()
        },
        "verdict": (
            f"{len(survivors)}/{len(rep.stats)} statistics survive every null"
            if survivors else
            "NO SIGNAL — GARCH explains everything observed"
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Analyze individual stocks")
    ap.add_argument("symbols", nargs="*", help="tickers to analyze")
    ap.add_argument("--watchlist", default=None,
                    help="'default' for built-in mega-cap list")
    ap.add_argument("--period", default="5y", help="history length (default: 5y)")
    ap.add_argument("--n-sim", type=int, default=300,
                    help="surrogates per null (default: 300)")
    args = ap.parse_args(argv)

    symbols = args.symbols or []
    if args.watchlist == "default" or (not symbols and not args.watchlist):
        symbols = DEFAULT_WATCHLIST
    elif args.watchlist:
        p = Path(args.watchlist)
        if p.exists():
            symbols = [s.strip() for s in p.read_text().splitlines() if s.strip()]

    symbols = [s.upper() for s in symbols]
    print(f"analyzing {len(symbols)} stocks · {args.period} history · "
          f"{args.n_sim} surrogates per null\n")

    store = Store("data")
    results = []

    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {sym}")

        # Fetch
        df = fetch_stock(sym, args.period)
        if df is None:
            continue

        # Store
        store.append("bronze", f"stock_{sym.lower()}", df,
                     source=f"yfinance {sym}", partition=date.today().isoformat())

        # Returns
        px = df["close"].to_numpy(float)
        r = np.diff(np.log(px[px > 0]))
        r = r[np.isfinite(r)]
        if r.size < 252:
            print(f"  {sym}: only {r.size} returns, skipping analysis (need 252+)")
            continue

        # Analyze
        result = analyze_one(sym, r, args.n_sim)
        result["last_close"] = float(df["close"].iloc[-1])
        result["last_date"] = str(df["date"].iloc[-1].date())
        results.append(result)

        v = result["verdict"]
        sr = result["sharpe"]
        vol = result["ann_vol"]
        print(f"  {r.size} obs · Sharpe {sr:.2f} · vol {vol*100:.1f}% · {v}")

    # Write findings
    out_dir = Path("research/findings")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"stocks-{date.today().isoformat()}.json"
    out.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(),
        "n_sim": args.n_sim,
        "period": args.period,
        "stocks": results,
    }, indent=2))

    # Print summary table
    print(f"\n{'='*90}")
    print(f"{'SYMBOL':>7} {'DAYS':>6} {'RETURN':>8} {'VOL':>7} {'SHARPE':>7} "
          f"{'KURT':>6} {'GARCH_P':>8} {'VOL_RATIO':>10} {'VERDICT':>20}")
    print(f"{'-'*90}")
    for s in sorted(results, key=lambda x: -x["sharpe"]):
        print(f"{s['symbol']:>7} {s['n_obs']:>6} {s['ann_return']*100:>7.1f}% "
              f"{s['ann_vol']*100:>6.1f}% {s['sharpe']:>7.2f} "
              f"{s['excess_kurtosis']:>6.1f} {s['garch']['persistence']:>8.4f} "
              f"{s['vol_ratio'] or 0:>10.2f} "
              f"{'SIGNAL' if s['n_survivors']>0 else 'no signal':>20}")

    print(f"\nwrote {out}")
    print(f"rebuild dashboard:  .venv/bin/python3 build_dashboard.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
