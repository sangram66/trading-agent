# Setup

Tested clean-room on Python 3.12 / Linux. No API keys, no accounts, no paid data.

---

## 1. Install

```bash
tar xzf trading-agent.tar.gz
cd trading-agent

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Roughly 150 MB of wheels, about a minute. Python 3.10+ required.

## 2. Verify

```bash
python3 -m tests.test_engine
```

Expect, in about 11 seconds:

```
======================================================================
43 passed, 0 failed
```

**If anything fails, stop and fix it before running research.** These are
known-answer gates, not smoke tests — a failure means a statistic is wrong, and
every number downstream inherits it.

## 3. Run the vol lab on real data

```bash
python3 run_vol_lab.py
```

Takes about 13 seconds on one core (1,000 surrogates × 4 null models × 11 statistics over 5,030
real trading days). Writes `data/{bronze,silver,gold}/*.parquet` plus
`data/manifest.json`.

---

## What you should see

```
S&P 500 · 5,030 trading days · 1999-01-05 to 2018-12-31

GARCH(1,1) fit   omega=1.718e-06  alpha=0.1004  beta=0.8878
                 persistence=0.98815  vol half-life=58d

STATISTIC                   REAL   SHUFFLED     BLOCK  STATIONARY     GARCH
ACF |r| lag 1             0.2443      3e-05     0.229      0.2256    0.2545
Ljung-Box Q(22) on |r|      8417       22.2      2744        3427      5720
Same-band rate            0.7022     0.5164    0.5972      0.6253    0.6635

SAME-BAND RATE — WHAT THE BASELINE SHOULD BE
  real                          70.2%
  naive 1/k independence        20.0%   -> claims +50.2 pts of lift
  shuffled null (honest)        51.6%   -> real lift +18.6 pts
  garch null                    66.5%   -> lift beyond clustering +3.7 pts

VERDICT
  NO SIGNAL — every statistic is reproducible by at least one null.
```

Those numbers are exact, not approximate. If yours differ, something is wrong
with your install — see *Reproducibility* below.

---

## 4. Data sources

Two work out of the box, and they produced every number above:

| Source | Data | How |
|---|---|---|
| `arch` package | S&P 500 daily OHLCV, 1999–2018, 5,031 days | bundled in the wheel, no network |
| GitHub `datasets/finance-vix` | CBOE VIX daily from 1990 | one HTTPS GET, cached to disk |

```python
from engine.ingest.sources import sp500_arch, vix_cboe, to_returns
px = sp500_arch()          # real OHLCV, no network
r  = to_returns(px)["ret"].to_numpy()
```

The rest need open outbound HTTPS — they work on your machine but not in a
locked-down sandbox. All free, no keys:

```python
from engine.ingest.sources import (stooq_daily, cftc_cot,
                                   edgar_company_tickers, edgar_filings,
                                   ishares_holdings)

cftc_cot(2025)                      # weekly futures positioning -> FL-CTA
edgar_filings(cik=102909)           # 13F / N-PORT with real filing dates
stooq_daily("^spx")                 # free EOD, no API key
```

**Before using EDGAR, put your real contact details in `SEC_UA`** at the top of
`engine/ingest/sources.py`. The SEC requires a declaring User-Agent and will
block you otherwise. The client already throttles below their 10 req/s limit and
caches every response under `data/.cache`, so a rate limit bounds your first
backfill once — not your research forever.

---

## 5. Reproducibility

Two bugs surfaced only when the code was run in a fresh virtualenv, and both are
now gated by test V12:

- **`hash()` on strings is salted per process.** Using it to seed the per-null
  RNG gave different results on every run. Now a SHA-256 digest.
- **`retrieved_at` is wall-clock**, so it entered the content hash and the
  manifest tracked download times instead of data identity. Volatile provenance
  columns are now excluded from the digest but still stored.

Verified: `run_vol_lab.py` is byte-identical across processes, across
`PYTHONHASHSEED` values, and across library versions (numpy 2.4.4/pandas 3.0.2
vs numpy 2.5.2/pandas 3.0.5).

To check your own install:

```bash
python3 run_vol_lab.py > a.txt
PYTHONHASHSEED=999 python3 run_vol_lab.py > b.txt
diff a.txt b.txt && echo REPRODUCIBLE
```

---

## 6. Using it on your own series

```python
import numpy as np
from engine.nulls.compare import compare

r = np.asarray(my_daily_returns, float)
rep = compare(r, n_sim=1000, seed=42)

print(rep.table())         # the THREE WORLDS panel
print(rep.pvalue_table())
print(rep.verdict())       # what survives every null
```

Two knobs worth understanding:

- `n_sim` sets p-value resolution. The floor is `1/(n_sim+1)`, so 1,000 gives
  ~0.001. Runtime is linear; 200 is fine while iterating.
- `block_len` should be about **twice the horizon you are claiming**. Beating the
  block bootstrap then means your effect is genuinely longer-ranged than the
  autocorrelation you happen to be sitting on.

Read the verdict conservatively. `survives()` requires rejecting *every* null,
weakest link decides, and the GARCH null is usually the one that kills things —
which is the point.

---

## 7. Layout

```
engine/core/storage.py    Parquet + DuckDB + content-hashed manifest
engine/core/asof.py       vintage tables, as-of joins, statutory filing lags
engine/nulls/             statistics · null models · compare engine
engine/audit/lookahead.py knowable-timestamp audit + plausibility ceiling
engine/ingest/sources.py  free-data clients
tests/test_engine.py      43 known-answer gates
tasks/lessons.md          what broke and why
data/                     gitignored; manifest.json is the reproducibility record
```

## Troubleshooting

**`ModuleNotFoundError: engine`** — run from the repo root, and use
`python3 -m tests.test_engine`, not `python3 tests/test_engine.py`.

**`arch` fails to build** — needs a C compiler for some Python versions. On
Ubuntu: `sudo apt install build-essential python3-dev`. Python 3.12 has prebuilt
wheels and needs nothing.

**V5 `arch cross-check` fails** — the from-scratch GARCH fit disagrees with the
`arch` package's estimator. That is a real numerical problem, not a flaky test.
Don't proceed.

**Want it faster still** — drop `N_SIM` at the top of the file to 200 while
you are exploring; restore 1,000 before recording a verdict.

**HTTP 403 from sec.gov** — you did not set a declaring `User-Agent`. See step 4.
