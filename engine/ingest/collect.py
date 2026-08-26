"""
Forward-collection entrypoint. This is what the daily cron runs.

Free *real-time* data is easy; free *history* is not. Every day this job does
not run is a day of point-in-time data you can never buy back, at any price.
So it is written to be boring: fetch what it can, skip what it can't, never
abort the whole run because one source is down, and always leave the store in a
verifiable state.

    python3 -m engine.ingest.collect
    python3 -m engine.ingest.collect --verify-only
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import date

from engine.core.storage import Store
from engine.ingest import sources

# The SEC requires a declaring User-Agent with real contact details.
if os.environ.get("SEC_CONTACT"):
    sources.SEC_UA = f"trading-agent research {os.environ['SEC_CONTACT']}"


def _try(store: Store, layer: str, dataset: str, fn, source: str) -> tuple:
    """Run one collector. A failure is logged and skipped, never fatal.

    One dead endpoint must not cost you the other six sources for that day.
    """
    try:
        df = fn()
        if df is None or len(df) == 0:
            return dataset, "empty", 0
        store.append(layer, dataset, df, source=source)
        return dataset, "ok", len(df)
    except Exception as exc:                        # noqa: BLE001
        print(f"  ! {dataset}: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return dataset, f"failed ({type(exc).__name__})", 0


def verify(store: Store) -> int:
    """Re-hash every dataset against the manifest. Non-zero exit on mismatch."""
    bad = []
    for key in store.manifest:
        layer, dataset = key.split("/", 1)
        if not store.verify(layer, dataset):
            bad.append(key)
    if bad:
        print("MANIFEST MISMATCH: " + ", ".join(bad), file=sys.stderr)
        return 1
    print(f"manifest verified — {len(store.manifest)} datasets intact")
    return 0


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    store = Store("data")

    if "--verify-only" in argv:
        return verify(store)

    print(f"forward-collection · {date.today().isoformat()}")

    jobs = [
        # (layer, dataset, callable, source label)
        ("bronze", "vix_daily", sources.vix_cboe, "CBOE via GitHub mirror"),
        ("bronze", "sp500_daily", sources.sp500_arch, "arch package (bundled)"),
        # Below need open outbound HTTPS. They are expected to fail inside a
        # restricted sandbox and to succeed on a real runner.
        ("bronze", "cftc_cot", lambda: sources.cftc_cot(date.today().year),
         "CFTC Commitments of Traders"),
        ("bronze", "edgar_tickers", sources.edgar_company_tickers,
         "SEC EDGAR company_tickers.json"),
    ]

    results = [_try(store, layer, ds, fn, src) for layer, ds, fn, src in jobs]

    print("\nsummary")
    ok = 0
    for dataset, status, n in results:
        print(f"  {dataset:20s} {status:24s} {n or ''}")
        ok += status == "ok"

    print(f"\n{ok}/{len(results)} sources collected")

    # Exit non-zero only if *everything* failed — that means the runner has no
    # network or the code is broken, which is worth an alert. Partial failure is
    # normal and should not page anyone.
    if ok == 0:
        print("all sources failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
