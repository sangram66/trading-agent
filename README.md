# trading-agent — foundation

Phase 1 + 2 of the build plan: the data plane and the null engine, built on the
free-data stack and verified against **real market data**.

No backtester yet, on purpose. The null engine and the lookahead audit are the
immune system; they get built and proven before anything is allowed to produce a
performance number.

```
pip install numpy pandas scipy pyarrow duckdb arch
python3 -m tests.test_engine      # 43 verification gates
python3 run_vol_lab.py            # the VOLATILITY surface on real S&P 500 data
```

---

## The headline result

Running the `THREE WORLDS, ONE PIPELINE` comparison over **5,030 days of real
S&P 500 history (1999–2018)**:

```
STATISTIC                   REAL   SHUFFLED     BLOCK  STATIONARY     GARCH    p(garch)
Ann. volatility           0.1911     0.1911    0.1911      0.1905     0.186      0.607
Excess kurtosis            8.169      8.169     7.768       7.618     10.52      0.701
ACF |r| lag 1             0.2443      3e-05     0.229      0.2256    0.2545      0.999
ACF |r| lag 22            0.2249  -0.000561 -0.001794     0.06679    0.1765      0.432
Ljung-Box Q(22) on |r|      8417       22.2      2744        3427      5720      0.348
Same-band rate            0.7022     0.5164    0.5972      0.6253    0.6635      0.104
Regression R2             0.8905     0.5341    0.7486      0.7878    0.8721      0.486

VERDICT: NO SIGNAL — every statistic is reproducible by at least one null.
```

Two things fall out of this, and both matter more than the code.

**The independence baseline is not 1/k.** Vol-band persistence looks
spectacular against the analytic 20% baseline — 70.2% observed, a claimed
**+50.2 points of lift**. But a 20-day vol window sampled 5 days apart shares 15
of its 20 observations, and that mechanical overlap lifts *shuffled noise* to
51.6%. The honest lift is +18.6 points, not +50.2.

**And that residue is just GARCH.** Against a GARCH(1,1) null with bootstrapped
residuals, the same-band rate reaches 66.3% and the remaining lift is +3.9 points
at p = 0.10. So volatility regime persistence in the S&P is real, it is large,
and it is *entirely* explained by a three-parameter model you can fit in half a
second. There is nothing tradable left in it.

This is the correct answer, and it is the one a data-mining loop would never
find — it would have reported the +50.2 and moved on to sizing the position.

---

## Layout

```
engine/
  core/storage.py      Parquet + DuckDB + manifest with content hashes
  core/asof.py         vintage tables, as-of joins, statutory filing lags
  nulls/statistics.py  the reel's statistic set, from scratch
  nulls/models.py      shuffle · circular block · stationary · GARCH-FHS
  nulls/compare.py     THREE WORLDS, ONE PIPELINE
  audit/lookahead.py   knowable-timestamp audit + plausibility ceiling
  ingest/sources.py    free-data clients
tests/test_engine.py   43 known-answer gates
tasks/lessons.md       what broke and why
run_vol_lab.py
```

## Data sources wired up

Runs anywhere (used for the results above):

| Source | Data | Real? |
|---|---|---|
| `arch` package | S&P 500 daily OHLCV, 1999–2018, 5,031 days | yes, bundled |
| GitHub `datasets/finance-vix` | CBOE VIX daily from 1990 | yes, CBOE mirror |

Needs open network — run locally:
`stooq_daily` · `cftc_cot` (weekly positioning to 1986) · `edgar_company_tickers`
· `edgar_filings` (13F, N-PORT) · `ishares_holdings` (daily creation/redemption
baskets). All free. All carry their own `knowable_at`.

## The three-timestamp rule

Every row, everywhere:

```
event_time    what period the row describes
knowable_at   when the market could first have known it   <- every join uses this
retrieved_at  when we pulled it
```

A 13F describing Q1 has `event_time` 2026-03-31 and `knowable_at` 2026-05-15.
Joining on the former hands you six weeks of foresight and produces a backtest
that looks perfectly well-behaved. `validate_pit` raises rather than warns.

## Verification

43 gates, each asserting a known answer rather than that code executes. The pair
that carries the design:

- **V2** GARCH-simulated series **rejects** the shuffle null (p = 0.007) — there
  is real structure.
- **V3** the same series **does not reject** the GARCH null (p = 0.25) — the
  structure is only clustering.

An engine that can't tell those apart will certify noise as edge. V5 additionally
cross-checks the from-scratch GARCH fit against the `arch` package's estimator
(α 0.079 vs 0.079).

## Next

Backtester + cost model, then the gate (deflated Sharpe, PBO/CSCV,
cost-sensitivity, capacity), then the 500-random-strategies test that proves the
gate rejects ≥95% of data-mined junk. Atlas dossiers start with the actors free
data can actually see: `FL-CTA` (COT), `FL-INDEX`, `FL-ETF-AP` (issuer holdings).

And start the forward-collection cron now — it only gets more valuable.
