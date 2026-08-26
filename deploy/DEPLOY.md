# Deploy — almost free

**Answer: GitHub Actions on a private repo. $0/month.** Everything below is
measured, and the deployment scripts are in the repo and tested.

---

## Why not a server

Measured on this workload, one core:

| | |
|---|---|
| Full research run (`run_vol_lab.py`) | **13 s** |
| All 65 verification gates | **24 s** |
| Peak memory | **227 MB** |
| Daily ingest | **~2 min** |

There is nothing here that justifies renting a machine. What you need is a
*reliable scheduler*, and GitHub gives that away.

---

## Deploy

```bash
./deploy/bootstrap.sh
```

It verifies all 65 gates first and **refuses to deploy a broken engine** — once
this runs on a schedule nobody is watching, a statistics bug quietly poisons the
permanent record. Then it creates a private repo, sets three secrets, and
triggers the first ingest.

Manually, if you prefer:

```bash
git init -b main && git add -A && git commit -m "initial"
gh repo create trading-agent --private --source=. --push
gh secret set SEC_CONTACT      --body "you@example.com"   # sec.gov requires it
gh secret set ALERT_WEBHOOK    --body "https://..."       # do not skip
gh secret set ANTHROPIC_API_KEY --body "sk-ant-..."       # agent runs only
gh workflow run ingest.yml
```

**Private is load-bearing.** It keeps your research private, and the widely
misquoted "scheduled workflows auto-disable after 60 days" rule applies only to
*public* repositories.

---

## What runs

| Workflow | Schedule | Duration | Cost |
|---|---|---|---|
| `ingest.yml` | weekdays 22:30 UTC | ~2 min | free |
| `agent-run.yml` | Saturdays 03:00 UTC | ~20 min | free compute, you pay tokens |

**Actions minutes:** 22 × 2 + 4 × 20 = **~124 min/month of the 2,000 free** on a
private repo. About 6%. You will not come close.

---

## The two things that would have broken this

Both were found by testing, not by reasoning.

### 1. The ledger race

`research/trial_budget.json` is the only shared mutable state in the whole
system. Two concurrent runs would last-write-wins away a trial charge — silently
lowering the significance bar, which is the exact failure the ledger exists to
prevent.

Fixed by giving **both workflows the same concurrency group**:

```yaml
concurrency:
  group: repo-writer          # shared by ingest.yml AND agent-run.yml
  cancel-in-progress: false   # cancelling mid-write truncates the ledger
```

That's the lock — GitHub queues them. Belt and braces, both jobs also
`git pull --rebase --autostash` with three retries before pushing.

**Tested:** a stale agent commit pushed after an ingest commit is correctly
rejected, then the retry loop lands it, and **both** writers' changes survive.

### 2. Committing data to git costs 13× more than it should

`Store.append` originally read the whole Parquet, concatenated, and wrote it
back. That produces a completely different blob every day, and git cannot delta
compressed binaries — so each day cost a full copy of the entire dataset.

Measured over 60 daily commits:

| | `.git` size |
|---|---|
| Monolithic (rewrite whole file) | **6.8 MB** |
| Date-partitioned (new file per day) | **0.5 MB** |

The gap widens without bound. `Store.append` now writes
`data/<layer>/<dataset>/YYYY-MM-DD.parquet` and never touches old partitions;
`read()` globs them back together and `verify()` hashes the set. Projected
growth is now roughly **0.5 GB/year**, which sits comfortably inside GitHub's
1 GB recommended repo size for years.

---

## Cost, honestly

| Item | Cost |
|---|---|
| Actions compute | **$0** (124 of 2,000 free minutes) |
| Storage in git | **$0** (~0.5 GB/year) |
| All market data | **$0** (EDGAR, CFTC, FINRA, CBOE, ETF issuers, Stooq) |
| LLM tokens for agent runs | **the only real cost** |

The token bill is the thing to govern, not the CPU. Three controls, all already
wired:

- `max_trials` input on `agent-run.yml` caps how much of the shared budget one
  run may consume.
- The orchestrator calls a model only at hypothesis selection, registration, and
  autopsy. Every other step is a deterministic tool call and costs nothing.
- The gate check runs *before* the agents, so a broken engine fails in 24 free
  seconds rather than after 20 minutes of paid tokens.

Start with `max_trials: 5` weekly. That is a handful of model calls a week —
single-digit dollars a month — and it is genuinely the right pace anyway, since
each trial permanently raises the bar for every future one.

---

## When to add a real server

Not yet. Add a **Hetzner CX23 (~€5.49/mo)** only when one of these is true:

- You want to serve the four UI surfaces from somewhere.
- Agent runs exceed the 6-hour job limit.
- You start collecting intraday data and the archive outgrows a git repo.

Until then a server is a thing that can break at 22:30 UTC while you're asleep,
in exchange for nothing.

---

## Verify it's alive

```bash
gh run list --limit 5          # recent runs
gh run watch                   # follow the current one
git log --oneline data/        # what the collector has actually captured
python3 -c "from agent.ledger import TrialLedger; print(TrialLedger().summary())"
```

**Configure `ALERT_WEBHOOK`.** GitHub does not tell you when a scheduled run
fails, and scheduled runs can be delayed 10–30 minutes under load. A silently
dead collector is worse than no collector: you find out months later that the
point-in-time history you were counting on has a hole in it, and that is the one
kind of damage you cannot repair by spending money.
