# todo.md

## Session 1 — Foundation (free-data path)

Scope: the parts that must be correct before any strategy is evaluated. Deliberately stops
short of the backtester — the null engine and the lookahead audit are the immune system, and
they get built and proven first.

### Plan

- [x] 1. Repo skeleton
- [x] 2. `core/storage.py` — Parquet + DuckDB, manifest with content hashes
- [x] 3. `core/fetch.py` — cached, rate-limited, checkpointing HTTP wrapper
- [x] 4. `core/asof.py` — vintage tables + as-of joins on `knowable_at`
- [x] 5. `nulls/statistics.py` — the reel's statistic set, implemented from scratch
- [x] 6. `nulls/models.py` — iid shuffle, circular block, stationary bootstrap, GARCH(1,1) FHS
- [x] 7. `nulls/compare.py` — the `THREE WORLDS, ONE PIPELINE` engine
- [x] 8. `audit/lookahead.py` — knowable-timestamp audit + shift test
- [x] 9. `ingest/` — Stooq, EDGAR, CFTC clients (offline-replayable)
- [x] 10. Tests with known answers; all must pass

### Verification gates for this session

Not "it runs" — each must produce a *known* answer:

| # | Gate | Expected |
|---|---|---|
| V1 | White noise vs every null | No rejection (p > 0.05) on all stats |
| V2 | GARCH-simulated series vs iid-shuffle null | **Rejects** — vol clustering is real structure |
| V3 | GARCH-simulated series vs GARCH null | **Does not reject** — the null captures it |
| V4 | AR(1) series vs iid shuffle / vs block bootstrap | Shuffle rejects; block bootstrap does not |
| V5 | GARCH parameter recovery | Fitted (ω,α,β) within tolerance of truth |
| V6 | Known synthetic Sharpe | Recovered inside its confidence interval |
| V7 | Lookahead audit, dirty feature | **Caught** |
| V8 | Lookahead audit, clean feature | **Passes** |
| V9 | As-of join with a filing lag | Never returns a row before `knowable_at` |
| V10 | Storage round-trip + manifest | Byte-identical, hash stable |

V2 + V3 together are the whole thesis: the engine must be able to tell the difference between
"there is structure here" and "the structure is only vol clustering." If it can't, every
downstream verdict is noise.

### Review

See `REVIEW.md` — written after tests pass, not before.

## Next session

- [ ] Backtester + cost model (spread, √(Q/ADV) impact, borrow)
- [ ] Gate: deflated Sharpe, PBO/CSCV, cost-sensitivity curve, capacity
- [ ] 500-random-strategies test (≤5% pass rate or the gate is broken)
- [ ] Atlas actor dossiers: FL-CTA, FL-INDEX, FL-ETF-AP first (free-data visible)
- [ ] Forward-collection cron — **start this on day 1, it only gets more valuable**

---

## Review (written after tests passed)

**Status: 39/39 gates green.** Two real bugs found and root-caused during
verification, both recorded in `lessons.md`:

- L-001 Arrow buffer hashing was unstable across a Parquet round-trip, so the
  manifest verifier false-positived on intact files. Fixed by hashing a
  canonical IPC serialisation.
- L-002 `shift_test` gated on one-bar degradation, which has no reliable
  direction under the null. Rewritten to gate only on the plausibility ceiling;
  unknowable-data features are caught by `audit_knowable` instead.

**Substantive finding (L-003):** on real S&P 500 data the vol-band persistence
statistic shows +50.2 pts of lift against the analytic 1/k baseline, +18.6 pts
against a shuffled null, and +3.9 pts (p=0.10) against GARCH. The 1/k baseline is
wrong because overlapping vol windows manufacture persistence in pure noise. The
null engine absorbed that artefact without being told about it.

**Not built, deliberately:** no backtester. Producing a performance number before
the immune system exists is the exact failure the whole design is arguing against.
