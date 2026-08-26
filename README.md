# Trading Agent

A self-improving AI trading research engine. Analyses stocks against four null models to separate real structure from volatility clustering, tracks every hypothesis in a shared trial ledger that prevents data-mining, and runs on free GitHub Actions.

Built on the thesis: **the backtester is a courtroom, not a search engine.** The system cannot look at returns before writing down what it expects, and every test permanently raises the bar for all future tests.

---

## What it does

- **Collects free market data** — S&P 500, VIX, CFTC futures positioning, EDGAR filings, individual stocks via Yahoo Finance
- **Runs four null models** on every series (IID shuffle, circular block bootstrap, stationary bootstrap, GARCH filtered historical simulation) across 11 statistics
- **Separates real signal from noise** — most apparent "edge" in daily returns is just volatility clustering wearing a hat. The engine catches that
- **Tracks a shared trial ledger** — every test any agent runs raises the significance bar. No reset, no delete. Ten agents share one budget
- **Produces a dashboard** — stocks screener, volatility regime tape, GARCH fits, pipeline health, trial budget status

---

## Quick start (5 minutes)

### Prerequisites

- Python 3.10+ (check with `python3 --version`)
- Git

### 1. Clone and install

```bash
git clone https://github.com/sangram66/trading-agent.git
cd trading-agent
PYTHON=/usr/local/bin/python3.12 ./setup.sh
```

> **Note:** Replace `/usr/local/bin/python3.12` with wherever your Python 3.10+ lives.
> Find it with: `which python3.12` or `which python3`

Expected output:

```
==> using /usr/local/bin/python3.12 (Python 3.12.x)
==> engine gates
43 passed, 0 failed
==> multi-agent gates
22 passed, 0 failed

SETUP OK
```

**If it says SETUP FAILED, stop.** Fix whatever gate broke before proceeding — a statistics bug poisons everything downstream.

### 2. Set your SEC contact

SEC requires a declaring User-Agent. Edit one line:

```bash
nano engine/ingest/sources.py
```

Find near the top:

```python
SEC_UA = "trading-agent research contact@example.com"
```

Change `contact@example.com` to your real email. Save.

### 3. Collect data

```bash
.venv/bin/python3 -m engine.ingest.collect
```

Expected:

```
forward-collection · 2026-08-26

summary
  vix_daily            ok                       9258
  sp500_daily          ok                       5031
  cftc_cot             ok                       ...
  edgar_tickers        ok                       ...

4/4 sources collected
```

### 4. Analyze stocks

Run against any tickers you want:

```bash
.venv/bin/python3 analyze_stocks.py AAPL MSFT NVDA TSLA GOOG AMZN META
```

Or use the built-in watchlist (19 stocks + ETFs):

```bash
.venv/bin/python3 analyze_stocks.py --watchlist default
```

Takes about 2-3 minutes. Pulls 5 years of daily data per stock and runs 4 null models × 11 statistics × 300 surrogates on each.

### 5. Build and open the dashboard

```bash
.venv/bin/python3 build_dashboard.py
open dashboard.html
```

The dashboard has four tabs:

| Tab | What it shows |
|---|---|
| **STOCKS** | Screener table: Sharpe, vol, kurtosis, GARCH persistence, vol ratio, signal count. Detail cards with per-statistic p-values |
| **VOLATILITY** | Same-band rate vs independence baseline, GARCH fit, regime tape, equity curve |
| **RESEARCH** | ORACLE findings — which statistics survive every null model |
| **BUDGET** | Trial ledger status — how many tests have been run, what the current bar is |

### 6. Run the vol lab (optional, 13 seconds)

Full THREE WORLDS, ONE PIPELINE comparison on the S&P 500:

```bash
.venv/bin/python3 run_vol_lab.py
```

---

## Customize the stock watchlist

The default list is in `analyze_stocks.py`:

```python
DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA",
    "JPM", "V", "UNH", "XOM", "LLY", "AVGO", "COST",
    "SPY", "QQQ", "IWM", "GLD", "TLT",
]
```

Edit this list, or pass tickers directly:

```bash
.venv/bin/python3 analyze_stocks.py PLTR AMD COIN SOFI RIVN
```

Or use a file:

```bash
echo "AAPL
NVDA
PLTR
AMD" > watchlist.txt

.venv/bin/python3 analyze_stocks.py --watchlist watchlist.txt
```

---

## Schedule it (runs automatically)

### Option A — Local cron (no accounts needed)

```bash
crontab -e
```

Add this line (replace the path with yours):

```
30 22 * * 1-5 cd /Users/YOUR_NAME/trading-agent && ./.venv/bin/python3 -m engine.ingest.collect >> collect.log 2>&1 && ./.venv/bin/python3 analyze_stocks.py --watchlist default --n-sim 200 >> collect.log 2>&1 && ./.venv/bin/python3 build_dashboard.py >> collect.log 2>&1
```

> **Cron gotchas:**
> - Use the full path `./.venv/bin/python3`, not `python3`
> - Escape `%` as `\%` in date commands
> - macOS: grant Full Disk Access to `/usr/sbin/cron` if the job never runs

### Option B — GitHub Actions (runs when laptop is off)

Already configured. Two workflows run automatically:

| Workflow | Schedule | What it does |
|---|---|---|
| `daily-ingest` | Weekdays 22:30 UTC | Collects data, commits to repo |
| `agent-run` | Saturdays 03:00 UTC | Analyzes stocks, runs ORACLE, rebuilds dashboard, commits |

**First-time setup:**

1. Push the code:

```bash
git add -A && git commit -m "update" && git push
```

2. Set secrets at https://github.com/sangram66/trading-agent/settings/secrets/actions :

| Secret | Value | Required? |
|---|---|---|
| `SEC_CONTACT` | Your email | Yes |
| `ALERT_WEBHOOK` | Slack/webhook URL for failure alerts | Recommended |
| `ANTHROPIC_API_KEY` | `sk-ant-...` | Not yet (nothing spends tokens today) |

3. Enable write permissions: Settings → Actions → General → Workflow permissions → **Read and write permissions** → Save

4. Trigger manually to test: Actions → agent-run → Run workflow

5. Pull results:

```bash
git pull
open dashboard.html
```

**Cost: $0/month.** Uses ~124 of 2,000 free Actions minutes.

---

## How it works

### The null engine

For every return series, the engine asks: *"Could this pattern have appeared by accident?"*

It generates thousands of alternative histories that preserve some features and destroy others, then checks whether the real series stands out:

| Null model | What it keeps | What it destroys |
|---|---|---|
| **IID shuffle** | Every return value | All time ordering |
| **Circular block** | Short-range dependence | Long-range structure |
| **Stationary bootstrap** | Same, with random block lengths | Same |
| **GARCH(1,1) FHS** | Vol clustering + fat tails | Everything else |

The GARCH null is the one that matters. Most apparent "edge" in daily data is volatility clustering, and GARCH reproduces it almost exactly. A statistic that survives the GARCH null is telling you something GARCH cannot explain.

### The trial ledger

Every backtest any agent runs increments a global counter. The Sharpe required to pass rises accordingly (Bailey-López de Prado deflated Sharpe ratio). The ledger has no reset method — deliberately.

After 200 trials across 10 agents, the bar is an annualised Sharpe of ~1.17. A Sharpe of 1.8 on six years of daily data is *not* established at that point. Ten years of the same edge passes.

### Pre-registration wall

An agent cannot see returns until it has written down: the mechanism, who is forced to trade, who takes the other side, the predicted sign and magnitude, and what observation would kill the hypothesis. `get_returns()` raises without this.

---

## Project structure

```
trading-agent/
├── setup.sh                    # One-command install + verification
├── analyze_stocks.py           # Stock analyzer (yfinance + null engine)
├── build_dashboard.py          # Dashboard generator (static HTML)
├── run_vol_lab.py              # Full vol comparison on S&P 500
├── requirements.txt
├── dashboard.html              # Generated — open in browser
│
├── engine/
│   ├── core/
│   │   ├── storage.py          # Parquet + DuckDB + content-hashed manifest
│   │   └── asof.py             # Point-in-time joins (knowable_at, not event_time)
│   ├── nulls/
│   │   ├── statistics.py       # 11 statistics from scratch
│   │   ├── models.py           # 4 null models including GARCH-FHS
│   │   └── compare.py          # THREE WORLDS, ONE PIPELINE engine
│   ├── audit/
│   │   └── lookahead.py        # Knowable-timestamp audit + shift test
│   └── ingest/
│       ├── sources.py          # Free data clients (EDGAR, CFTC, Stooq, yfinance)
│       └── collect.py          # Forward-collection entrypoint
│
├── agent/
│   ├── ledger.py               # Shared trial budget (append-only, no reset)
│   ├── tools.py                # Pre-registration wall + guarded tool surface
│   └── run.py                  # Agent orchestrator
│
├── research/
│   ├── hypotheses/             # Pre-registered hypothesis files
│   ├── findings/               # ORACLE + stock analysis results (JSON)
│   └── trial_budget.json       # The shared ledger (append-only)
│
├── data/
│   ├── bronze/                 # Raw collected data (partitioned Parquet)
│   ├── manifest.json           # Content hashes for reproducibility
│   └── .cache/                 # HTTP response cache (gitignored)
│
├── deploy/
│   ├── bootstrap.sh            # One-command GitHub deploy
│   ├── dry-run.sh              # Replays workflows locally before deploying
│   ├── DEPLOY.md               # Cloud deployment guide
│   ├── CLOUD.md                # Cloud options compared
│   └── MULTI-AGENT.md          # Multi-agent architecture
│
├── tasks/
│   ├── todo.md                 # Build plan + verification gates
│   └── lessons.md              # What broke and why (append-only)
│
├── tests/
│   ├── test_engine.py          # 43 known-answer gates
│   └── test_agent.py           # 22 multi-agent guardrail gates
│
└── .github/workflows/
    ├── ingest.yml              # Daily data collection (weekdays 22:30 UTC)
    └── agent-run.yml           # Stock analysis + ORACLE (Saturdays 03:00 UTC)
```

---

## Verification

65 gates, each asserting a known answer:

```bash
.venv/bin/python3 -m tests.test_engine    # 43 engine gates
.venv/bin/python3 -m tests.test_agent     # 22 multi-agent gates
```

Key gates:

| Gate | What it proves |
|---|---|
| V2/V3 | GARCH series rejects shuffle (there is structure) but not GARCH null (structure is only clustering) |
| V5 | From-scratch GARCH fit agrees with the `arch` package's estimator |
| V7/V8 | Lookahead audit catches dirty features, passes clean ones |
| V9 | As-of join never returns a row before `knowable_at` |
| V12 | Reproducible across processes and library versions |
| M4 | Bar rises as agents consume the shared budget |
| M5 | 200 data-mined noise strategies: 0 passed |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: engine` | Run from the repo root, use `python3 -m tests.test_engine` not `python3 tests/test_engine.py` |
| `403` from sec.gov | Set your real email in `SEC_UA` in `engine/ingest/sources.py` |
| Cron never runs (macOS) | Grant Full Disk Access to `/usr/sbin/cron` |
| `SETUP FAILED` | Read the `[FAIL]` lines it prints. Do not proceed until green |
| `0/4 sources collected` | No network — check your connection |
| Actions workflow fails | Check Settings → Actions → Workflow permissions → must be Read and write |

---

## What's next

The **backtester and cost model** — spread, √(Q/ADV) impact, borrow. This is what unlocks the LLM agents (ATHENA, CHARTIST) to propose and test strategies. Without it, every verdict is optimistic fiction, which is why those agents currently refuse to run.

---

## Data sources (all free)

| Source | Data | Cost |
|---|---|---|
| Yahoo Finance | Individual stock daily OHLCV | $0 |
| `arch` package | S&P 500 1999-2018 (bundled) | $0 |
| CBOE via GitHub | VIX daily from 1990 | $0 |
| CFTC | Weekly futures positioning (COT) | $0 |
| SEC EDGAR | Company tickers, 13F, N-PORT filings | $0 |
| FRED | Rates, curves, macro | $0 |

---

## License

Research tool. Not financial advice. Not a recommendation to trade.
