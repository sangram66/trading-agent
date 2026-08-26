# lessons.md

Patterns worth not repeating. Append-only.

---

### L-001 · Don't hash Arrow buffers for content identity
**Found:** V10 `manifest verifies` failed. `Store.verify()` reported corruption on
files that were perfectly intact.

**Root cause:** `_hash_table` hashed raw column buffers. Arrow buffer padding and
alignment are not preserved across a Parquet round-trip, so identical logical
data hashed differently after read-back.

**Rule:** content hashes must be taken over a *canonical serialisation*
(Arrow IPC with schema metadata stripped), never over in-memory buffers. Anything
that survives a write/read cycle is the only thing safe to call identity.

**Generalises to:** any reproducibility check. Verify the hash is stable across
the round-trip *before* trusting it to detect tampering — a verifier that
false-positives gets switched off, and then it protects nothing.

---

### L-002 · Don't gate on a metric that has no reliable direction
**Found:** V7c churned. First I asserted "a peeking signal degrades little when
lagged" and wrote a test that failed. Then I built a test case that degraded a
lot, so the detector passed it. Then a third case sat right on the threshold.

**Root cause:** one-bar degradation is a function of *signal persistence*, not of
honesty. A slow signal legitimately loses almost nothing to a one-day delay; a
fast one legitimately loses most of its edge. There is no threshold that
separates clean from dirty, so any threshold produces both error types.

**Rule:** if a diagnostic's expected value under the null hypothesis depends on a
free parameter you don't control, report it — don't gate on it. Split the job:

- `audit_knowable` catches unknowable-data features by **reconstruction** (exact,
  reliable, catches centred windows and full-sample normalisation).
- `shift_test` gates only on the **plausibility ceiling** (a daily Sharpe > 5
  does not exist; near-conclusive).

**Generalises to:** the gate design in Phase 4. Every check must have a known
direction under the null. "Looks suspicious" is not a threshold.

---

### L-003 · The independence baseline is almost never 1/k
**Found:** real S&P same-band rate 70.2% vs a naive 1/k baseline of 20.0% — an
apparent +50.2 pts of lift. The shuffled null puts the honest baseline at 51.6%,
so the real lift is +18.6 pts. Against GARCH it is +3.9 pts, p = 0.10.

**Root cause:** a 20-day volatility window sampled 5 days apart shares 15 of its
20 observations. That overlap creates persistence in *any* series, including pure
noise. The analytic 1/k baseline silently assumes independent draws.

**Rule:** never compare a statistic to an analytic baseline when a surrogate can
supply an empirical one. The surrogate absorbs every mechanical artefact —
overlap, burn-in, banding, finite-sample bias — automatically and for free. This
is the single strongest argument for building the null engine before the
backtester.

**Generalises to:** every "vs. random chance" claim in the whole project.

---

### L-004 · Fit with Gaussian QMLE, simulate with bootstrapped residuals
**Found:** standardised residuals from the fitted GARCH still carry excess
kurtosis 1.75 while raw returns carry 8.17. GARCH explains most but not all of
the fat tails.

**Rule:** simulating the GARCH null with normal innovations would leave that 1.75
on the table and make the null artificially easy to beat. Filtered historical
simulation — bootstrapping the fitted standardised residuals — keeps it, and
produces a null that is honest rather than flattering.

**Generalises to:** always ask what a null *fails* to reproduce, because that gap
is exactly where false discoveries will appear.

---

### L-005 · Wall-clock provenance must not enter a content hash
**Found:** clean-room run produced bronze hash `05604f09` where the dev run gave
`6042ad76`, on byte-identical 5,031-row data.

**Root cause:** `retrieved_at = utcnow()` is a column in the frame, so it entered
the digest. The manifest was tracking *when we downloaded* rather than *what we
downloaded* — exactly inverting its purpose.

**Rule:** partition columns into content and provenance. Hash content only; store
provenance alongside. `VOLATILE_COLUMNS` in `storage.py` is the registry.

---

### L-006 · `hash()` on strings is salted per process
**Found:** identical seed, identical data, GARCH null mean drifted 66.3% -> 66.5%
between runs.

**Root cause:** `compare()` derived each null's RNG offset from
`abs(hash(null_name))`. Python randomises string hashing per interpreter via
PYTHONHASHSEED, so "seed=42" was never actually fixed.

**Rule:** never use built-in `hash()` for anything that must be stable across
processes — seeds, cache keys, shard assignment, IDs. Use `hashlib`.

**Meta-lesson, and the important one:** both L-005 and L-006 were invisible in
the development environment and appeared the moment the code ran in a fresh
virtualenv. A single-environment test suite cannot catch reproducibility bugs by
construction. The clean-room run is now part of verification, not a nicety.

---

### L-007 · Measure before you size
**Found:** I wrote "~4 minutes" for `run_vol_lab.py` and "~2 minutes" for the
test suite in SETUP.md, from intuition. Measured: **13 seconds** and **11
seconds**, on one core, 227 MB peak RSS.

**Why it mattered:** the next question was cloud sizing. A 4-minute
multi-core-sounding job invites a $20–60/month instance. A 13-second 227 MB job
invites a free GitHub Actions cron and nothing else. An unmeasured guess was
about to become an architecture decision and a recurring bill.

**Rule:** never size infrastructure against an estimate. `time` and
`ru_maxrss` cost thirty seconds and change the answer by an order of magnitude.
Also: don't ship performance numbers in documentation that you have not run.

---

### L-010 · Don't rewrite a whole file to append, if the store lives in git
**Found:** `Store.append` read the existing Parquet, concatenated, and wrote it
back. Measured over 60 daily commits: **6.8 MB of `.git` versus 0.5 MB** for the
same data written as one file per day.

**Root cause:** compressed Parquet produces a completely different byte stream on
every rewrite, and git cannot delta binary blobs. Every daily commit therefore
stored a full copy of the entire dataset, and the cost grows quadratically with
time — file size grows linearly, and you pay it again every single day.

**Rule:** append-only stores must be *partitioned*, not rewritten.
`data/<layer>/<dataset>/YYYY-MM-DD.parquet`, old partitions never touched.

**Generalises to:** this only became visible when the deployment target was git.
Storage design cannot be decided independently of where the storage lives.

---

### L-011 · Concurrency groups are the lock
**Found:** two workflows writing `research/trial_budget.json` would last-write-
wins away a trial charge — silently lowering the significance bar, the exact
failure the ledger exists to prevent.

**Fix:** both workflows share `concurrency: group: repo-writer` with
`cancel-in-progress: false`, plus `git pull --rebase --autostash` with retries.
Tested end to end: a stale push is rejected, the retry lands it, and both
writers' changes survive.

**Rule:** identify the single shared mutable state early and give it exactly one
writer. For this system that is the trial ledger, and everything else is
reconstructible.

**Meta:** the first version of this race test used an empty-clone git harness
that reported "pushed on attempt 1" while silently losing the commit. A test
that passes for the wrong reason is worse than no test — the harness has to be
verified before its result means anything.

---

### L-012 · A workflow that references a module nobody wrote
**Found:** `agent-run.yml` called `python3 -m agent.run`. That module did not
exist. The workflow would have failed on its first scheduled Saturday, and
because GitHub does not notify on scheduled failures, silently.

**Rule:** every entry point a workflow invokes gets an existence check in CI
before deploy. `deploy/dry-run.sh` step 0 now does exactly this, and
`bootstrap.sh` refuses to deploy unless the dry run is green.

**Generalises to:** YAML is not type-checked and nothing validates that a
`run:` line points at real code. Treat workflow files as untested code until
something executes them.

---

### L-013 · A harness that falls through to the source tree
**Found:** `dry-run.sh` used `rsync` to build a sandbox copy. `rsync` was not
installed; the copy silently did nothing, the following `cd` failed, and the
script carried on executing `git init` and commits **against the real
repository**.

**Rule:** in any script that operates on a throwaway copy, every `cd` into the
sandbox is fatal (`cd "$X" || exit 1`), and prefer `tar` piping over `rsync`
for portability. A harness that quietly relocates its work into your source
tree is worse than no harness.

**Second occurrence of this shape** (see L-011's note on the empty-clone git
test). Pattern: *verify the harness before trusting its verdict.* Both times the
output looked plausible while measuring the wrong thing.
