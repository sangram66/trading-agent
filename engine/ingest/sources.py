"""
Ingest clients for the free-data stack.

Two of these run anywhere (`sp500_arch`, `vix_cboe`) because their data is
bundled or served from GitHub. The rest hit sec.gov / stooq.com / cftc.gov and
will work on your machine but not inside a restricted sandbox.

Every client returns a tidy DataFrame carrying its own point-in-time stamps and
declares its source string for the manifest.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# SEC requires a declaring User-Agent and throttles at 10 req/s. Being a good
# citizen here is not optional; they will block you.
SEC_UA = "trading-agent research contact@example.com"
CACHE = Path("data/.cache")


def cached_get(url: str, headers: dict | None = None, ttl_hours: int = 24,
               rate_limit_s: float = 0.15) -> bytes:
    """GET with on-disk caching keyed by URL hash.

    This is what makes stingy free tiers viable: a rate limit bounds your
    initial backfill once, not your research forever. Never let the agent loop
    call this directly — only the ingest layer touches the network.
    """
    import hashlib
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()[:24]
    p = CACHE / key
    if p.exists() and (time.time() - p.stat().st_mtime) < ttl_hours * 3600:
        return p.read_bytes()
    time.sleep(rate_limit_s)
    resp = requests.get(url, headers=headers or {"User-Agent": SEC_UA}, timeout=60)
    resp.raise_for_status()
    p.write_bytes(resp.content)
    return resp.content


# --------------------------------------------------------------------------
# runs anywhere
# --------------------------------------------------------------------------

def sp500_arch() -> pd.DataFrame:
    """Real S&P 500 daily OHLCV, 1999-2018, bundled inside the `arch` package.

    5,031 trading days of genuine index history with no network call. Ideal for
    validating the statistics engine before pointing it at anything you paid for.
    """
    import arch.data.sp500 as ds
    df = ds.load().reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"adj_close": "adj_close"})
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = "^GSPC"
    df["event_time"] = df["date"]
    df["knowable_at"] = df["date"]          # a close is knowable at that close
    df["retrieved_at"] = pd.Timestamp.utcnow()
    return df[["symbol", "date", "open", "high", "low", "close", "adj_close",
               "volume", "event_time", "knowable_at", "retrieved_at"]]


VIX_URL = ("https://raw.githubusercontent.com/datasets/finance-vix/"
           "main/data/vix-daily.csv")


def vix_cboe(url: str = VIX_URL) -> pd.DataFrame:
    """Real CBOE VIX daily history from 1990, mirrored on GitHub.

    Upstream is cdn.cboe.com; the GitHub mirror is a straight copy and is
    reachable from more places.
    """
    raw = cached_get(url, headers={"User-Agent": "trading-agent"})
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = "^VIX"
    df["event_time"] = df["date"]
    df["knowable_at"] = df["date"]
    df["retrieved_at"] = pd.Timestamp.utcnow()
    return df[["symbol", "date", "open", "high", "low", "close",
               "event_time", "knowable_at", "retrieved_at"]].dropna()


# --------------------------------------------------------------------------
# needs open network — run these locally
# --------------------------------------------------------------------------

def stooq_daily(symbol: str = "^spx") -> pd.DataFrame:
    """Free EOD from Stooq. No API key, global coverage, CSV over HTTP."""
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    df = pd.read_csv(io.BytesIO(cached_get(url)))
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = symbol
    df["event_time"] = df["date"]
    df["knowable_at"] = df["date"]
    df["retrieved_at"] = pd.Timestamp.utcnow()
    return df


def cftc_cot(year: int) -> pd.DataFrame:
    """CFTC Commitments of Traders — weekly futures positioning, free, to 1986.

    This *is* the FL-CTA actor. Nothing paid improves on it.

    knowable_at is the Friday release, three days after the Tuesday snapshot;
    using the report date directly would give you three days of foresight into
    managed-money positioning.
    """
    import zipfile
    url = f"https://www.cftc.gov/files/dea/history/deacot{year}.zip"
    with zipfile.ZipFile(io.BytesIO(cached_get(url))) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    date_col = next(c for c in df.columns if "as_of_date_in_form" in c
                    or c.startswith("report_date"))
    # Assign all PIT columns in one shot. The COT file has ~200 columns, and
    # pandas warns (correctly) about inserting into a wide frame one column at
    # a time — each insert copies the whole block map.
    event = pd.to_datetime(df[date_col], errors="coerce")
    df = df.assign(
        event_time=event,
        knowable_at=event + pd.Timedelta(days=3),
        retrieved_at=pd.Timestamp.utcnow(),
    )
    return df.dropna(subset=["event_time"])


def edgar_company_tickers() -> pd.DataFrame:
    """CIK <-> ticker map. The entry point for every other EDGAR pull."""
    raw = cached_get("https://www.sec.gov/files/company_tickers.json")
    import json
    d = json.loads(raw)
    df = pd.DataFrame(d.values())
    df["cik"] = df["cik_str"].astype(int)
    return df[["cik", "ticker", "title"]]


def edgar_filings(cik: int, forms: tuple = ("13F-HR", "NPORT-P")) -> pd.DataFrame:
    """A filer's submission history, filtered to the forms you care about.

    Note `knowable_at` comes from `filingDate`, the actual date it hit EDGAR —
    always prefer this to a statutory lag estimate when you have it.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    import json
    d = json.loads(cached_get(url))
    recent = d["filings"]["recent"]
    df = pd.DataFrame(recent)
    df = df[df["form"].isin(forms)].copy()
    df["event_time"] = pd.to_datetime(df["reportDate"], errors="coerce")
    df["knowable_at"] = pd.to_datetime(df["filingDate"], errors="coerce")
    df["retrieved_at"] = pd.Timestamp.utcnow()
    df["cik"] = cik
    return df.dropna(subset=["event_time", "knowable_at"])


def ishares_holdings(fund_url: str) -> pd.DataFrame:
    """Daily full holdings CSV published by an iShares ETF.

    Diff consecutive days and you have inferred the creation/redemption basket —
    i.e. what the authorised participants were forced to trade yesterday. That
    is the FL-ETF-AP actor, free, straight from the issuer.
    """
    raw = cached_get(fund_url, headers={"User-Agent": "trading-agent"})
    txt = raw.decode("utf-8", errors="ignore")
    start = txt.find("Ticker,")
    df = pd.read_csv(io.StringIO(txt[start:]))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


# --------------------------------------------------------------------------
# derived
# --------------------------------------------------------------------------

def to_returns(df: pd.DataFrame, price_col: str = "adj_close",
               log: bool = True) -> pd.DataFrame:
    """Daily returns from a price frame, with PIT stamps preserved.

    The return for day t is knowable at the close of day t and not one second
    earlier, which is the shift that kills most naive signal code.
    """
    out = df.sort_values("date").copy()
    p = out[price_col].to_numpy(float)
    r = np.full(p.size, np.nan)
    r[1:] = np.log(p[1:] / p[:-1]) if log else p[1:] / p[:-1] - 1.0
    out["ret"] = r
    out["knowable_at"] = out["date"]
    return out.dropna(subset=["ret"]).reset_index(drop=True)
