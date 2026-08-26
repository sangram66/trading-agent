# Multi-agent desk — architecture

Referencing the Nimbus reel (@cloud9.markets), mapped onto what we've built.

---

## 1. What's in the video

A 3D topology of ten named agents around a master node, filmed off a monitor.
Built with Claude Fable 5 (badge visible in frame 1).

```
                      CAPITOL (smart money)
        ATLAS (macro)                      SCOUT (recon)
                       — INTELLIGENCE —

  ATHENA (analyst)      ┌───────────┐      SENTINEL (risk officer)
  CHARTIST (technician) │  NIMBUS   │              │  ◇ gate
  ORACLE (quant)        │  master · │      PILOT (execution)
        — ANALYSIS —    │ synthesis │              │
                        │ risk route│      LEDGER (the book)
                        └───────────┘       — EXECUTION —
```

Captions: *"Everyone's building Jarvis, I built Nimbus, multi agent trading
desk … Let's [see what] your [agents] know … run. No cowboy … this desk."*

**What it is not:** there are no agent traces, no tool calls, no outputs, no
P&L — just the topology, rotating, with a hand pointing at nodes. Take the org
chart as a good idea and the demo as a UI. The interesting question the video
doesn't answer is what each node is *made of*, and that turns out to be the
whole design.

---

## 2. The decision that matters: which of these are LLMs?

Most of them must not be.

An LLM is the right tool for *proposing* — reading filings, generating
hypotheses, spotting that two situations rhyme. It is the wrong tool for
*adjudicating*, because it can be argued into a different answer, and an
autonomous loop will eventually do the arguing. A risk officer you can talk out
of a limit is not a risk officer; it's a formality with a nice name.

| Nimbus node | Job | Implementation | Why |
|---|---|---|---|
| SCOUT, ATLAS, CAPITOL | gather | **deterministic collectors** + LLM summarisation | `engine/ingest/` — the fetching is code; the LLM only reads the result |
| ATHENA, CHARTIST | propose hypotheses | **LLM** | this is the one job LLMs are genuinely good at |
| ORACLE (quant) | test against nulls | **deterministic** | `engine/nulls/` — already built, 43 gates |
| SENTINEL (risk officer) | approve / veto | **deterministic** | `agent/ledger.py` — thresholds in config, not in a prompt |
| LEDGER (the book) | positions, P&L, provenance | **deterministic** | `engine/core/storage.py`, content-hashed manifest |
| PILOT (execution) | place orders | **deterministic** + broker API | an LLM must never emit an order directly |
| NIMBUS (master) | route, synthesise | **LLM, narrow** | picks what to work on; cannot compute or approve |

So: **three LLM roles, seven deterministic modules.** The reel's picture is a
good org chart. It is not an instruction to make ten chatbots talk to each other.

---

## 3. The thing that gets multi-agent wrong

Here is the arithmetic that should govern the whole design.

One researcher testing 20 hypotheses will find a Sharpe of 2 by luck eventually.
**Ten agents get there ten times faster.** Parallelism doesn't discover more
edge; it discovers more *coincidence*, and it does so overnight, unsupervised,
while producing confident prose about mechanisms.

A multi-agent desk without a shared multiple-testing budget is not ten
researchers. It is one researcher with ten times the opportunities to fool
themselves.

So the ledger is **global, append-only, and has no reset method** — deliberately,
because an agent that could clear its own record of failed attempts would be
able to launder an overfit strategy into a clean one, and an LLM asked to "start
fresh" will reach for that button if it exists.

### Measured, not asserted

`tests/test_agent.py`, 22 gates. Two results worth reading:

**200 data-mined noise strategies across 10 agents → 0 passed.** The luckiest
reached an annualised Sharpe of 1.30; the bar had risen to 1.17 and the deflated
Sharpe ratio finished the job.

**After 200 trials, an annualised Sharpe of 1.8 on six years of daily data is
not established** (DSR 0.947, need > 0.95). The same edge with ten years does
pass. That is the correct answer, and it is the number to sit with before
building ten agents: the search cost is real, it is charged to you, and it is
larger than most people's intuition.

---

## 4. The contract agents run under

Three steps, enforced in code rather than in a prompt, because a prompt is a
suggestion:

```python
tools = AgentTools("athena", ledger, registry, returns_provider)

# 1. Write down what you expect, before seeing anything.
hid = tools.register_hypothesis(
    mechanism="Quarter-end rebalancing forces target-weight funds to sell "
              "the outperforming asset regardless of price.",
    who_is_forced="Defined-benefit pension funds and target-date funds",
    counterparty="Market makers, compensated for inventory risk",
    predicted_sign="negative in the outperforming leg",
    predicted_magnitude_bp=[4, 15],
    predicted_timing="T-3 to T-0 of quarter end, in the closing auction",
    falsifier="No relationship between prior-quarter dispersion and drift",
)

# 2. Only now may you look. get_returns() raises without step 1.
r = tools.get_returns(hid)

# 3. The gate charges a trial whether you pass or fail.
verdict = tools.submit_for_verdict(hid, strategy_returns)
```

Enforced, and tested:

- `get_returns()` on an unregistered hypothesis raises `PreRegistrationError`.
- Placeholder fields are rejected — `falsifier: "TBD"` is not a falsifier, and
  an agent under pressure will write exactly that.
- **An agent cannot test another agent's hypothesis.** Otherwise ten agents
  share one registration and the attribution in the ledger becomes fiction.
- **A parameter tweak is a new hypothesis** with a new trial charge. This is what
  stops "iterate until it works" from being free.
- `tools.journal()` exposes every hypothesis every agent has already killed.
  This is the only part of the design that makes the desk *self-improving*
  rather than merely parallel: the prior gets pruned, not resampled.

---

## 5. Cloud shape for a multi-agent desk

Multi-agent changes the cloud answer, but less than you'd think. The compute is
still trivial — what changes is that you now have **long-running LLM calls with
a real token bill**, and that's what needs governing.

```
GitHub Actions (free)          daily ingest cron — unchanged
        │
        ▼
  data/ in git                 point-in-time store, ~0.5 GB/year
        │
        ▼
Agent run: on-demand job       3 LLM roles, N iterations, 10-30 min
        │                      → Modal / Cloud Run Job / a Hetzner box
        ▼
research/trial_budget.json     committed back. THE shared state.
```

**The ledger is the only shared mutable state**, and it must be serialised. Two
agents running concurrently against the same budget file will last-write-wins
away a trial charge, which quietly makes the bar too low — the exact failure the
ledger exists to prevent. Two safe options: run agents sequentially within one
process (simplest, and 13-second research means this is fine), or put the ledger
behind a lock. Do not skip this; it is silent when it breaks.

**Cost governor:** the token bill, not the CPU, is the budget. Cap iterations per
run, cap trials per agent per day, and make the orchestrator invoke a model only
at steps 2, 3 and 6 of the loop — every other step is a deterministic tool call
and costs nothing.

**Recommendation, unchanged from before:** free GitHub Actions cron for ingest.
Add a **Hetzner CX23 (~€5.49/mo)** once agents run nightly — a persistent box is
simpler than serverless here because the agent loop wants a warm filesystem, the
ledger wants a single writer, and you'll want somewhere to serve the UI surfaces
later. Skip Kubernetes; this is one process with a JSON file.

---

## 6. Build order

1. **Backtester + cost model** — the ledger currently rules on a returns series
   somebody else produced. Without a real cost model (spread, √(Q/ADV) impact,
   borrow) the verdicts are optimistic regardless of how good the statistics are.
2. **Wire ORACLE to the existing null engine** — it's built and tested; the
   agent surface just needs to call it.
3. **One LLM role only: ATHENA.** Get a single proposer working end-to-end
   against the real ledger before adding a second. Two agents with a broken
   contract is not progress over one.
4. **Then CHARTIST, then NIMBUS routing.**
5. **PILOT last, paper-only, for at least 3 months or 100 trades.**

The reel's caption is right about the destination — *"No cowboy."* The way you
get there is that SENTINEL and LEDGER are Python, not prompts.
