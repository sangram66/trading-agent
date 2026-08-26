# Running this in the cloud

## First, the measurement that decides everything

Before pricing anything, I benchmarked the actual workload on one core:

| | Measured |
|---|---|
| Full `run_vol_lab.py` (1,000 surrogates × 4 nulls × 11 statistics × 5,030 days) | **13 seconds** |
| Full test suite, 43 gates | **11 seconds** |
| Peak RSS | **227 MB** |
| Cores used | **1** |
| Data written | **604 KB** |

I had estimated "about 4 minutes" in SETUP.md without measuring. That was wrong,
and the correction matters: **this is not a cloud-compute workload.** It is a
13-second, quarter-gigabyte, single-core job. Your laptop is a supercomputer for
it. Renting compute to run research would be paying to make it slower and less
convenient.

What you actually need in the cloud is something quite different, and much
cheaper.

---

## The workload splits in two, with opposite requirements

Conflating these is how people end up paying $60/month for this.

**1. The ingest cron — needs reliability, not power.**
Free real-time data is easy; free *history* is not. Every weekday this job
doesn't run is a day of point-in-time data you can never buy back at any price.
It uses ~2 minutes of CPU and must simply never silently stop. This is the only
part that genuinely belongs on someone else's always-on machine.

**2. Research — needs nothing.**
13 seconds, one core, 227 MB. Run it locally. If a future backtest sweep ever
gets heavy, rent a big box by the hour for that afternoon and destroy it.

---

## Option A — GitHub Actions (recommended, $0)

Put the repo on GitHub **private** and let Actions run the collector on a cron.
This is the whole deployment. There is no server.

The workflow is already written: `.github/workflows/ingest.yml`, validated and
scheduled for 22:30 UTC on weekdays. The collector is `engine/ingest/collect.py`
and it is tested — blocked sources skip rather than aborting the run, and the
manifest is verified before anything is committed.

**Setup**

```bash
gh repo create trading-agent --private --source=. --push
gh secret set SEC_CONTACT --body "you@example.com"
gh secret set ALERT_WEBHOOK --body "https://..."     # optional but do it
```

That's it. Data lands as Parquet committed back to the repo, so you get
versioned history and can diff yesterday's ETF holdings against today's for free.

**Cost:** private repos get <cite index="20-1">2,000 free Linux minutes per month on the Free plan</cite>. A daily
2-minute job uses about 60. You will not come close to the limit.

**The gotcha everyone repeats wrong.** You will read that scheduled workflows
auto-disable after 60 days of inactivity. The qualifier is load-bearing:
<cite index="19-1">the documented rule is that *in a public repository* scheduled workflows are automatically disabled when no repository activity has occurred in 60 days — it does not cover private repositories</cite>.
Since your repo is private (it contains your research), this doesn't apply. If
you ever make it public, add a keepalive job.

**The real gotcha.** <cite index="21-1">Scheduled workflows do not run at the exact time specified; delays of 10–30 minutes are common during high demand, and GitHub does not notify you when scheduled runs fail</cite>. A silently dead collector is
worse than no collector — you discover months later that your history has a
hole. The workflow already has a failure webhook step; configure it.

---

## Option B — A small VPS (~€5.50/mo)

If you'd rather have a box you can SSH into, run long jobs on, and eventually
serve the four UI surfaces from.

**Hetzner** is the price/performance leader. Note it got more expensive this
year: <cite index="3-1">prices rose on 15 June 2026, with some cloud instance families increasing more than 2.5x; CX (shared Intel/AMD) and CAX (Arm) saw smaller rises of roughly 1.3–1.4x</cite>. Current entry is <cite index="7-1">€5.49/month for a CX23, or €5.99 for a CAX11 Arm instance</cite>. Either is
overkill for a 227 MB job.

```bash
# on the box
git clone <your repo> && cd trading-agent && ./setup.sh
crontab -e
# 30 22 * * 1-5  cd ~/trading-agent && .venv/bin/python3 -m engine.ingest.collect >> ~/ingest.log 2>&1
```

Use `systemd` timers over `cron` if you want failure notification without
writing it yourself. Add restic or a nightly `rclone` push to object storage —
a VPS is a single point of failure and your archive is the irreplaceable asset.

---

## Option C — Oracle Always Free ($0, with asterisks)

Historically the most generous free tier. **Check the news before relying on it:**
<cite index="10-1">Oracle halved the Always Free Ampere allocation from 4 OCPU / 24 GB to 2 OCPU / 12 GB, with no announcement — the docs were simply edited</cite>, and <cite index="12-1">emails went out warning that Always-Free compute instances above the new limits would be terminated on or after August 18, 2026</cite>.

2 OCPU / 12 GB is still far more than this workload needs, and 200 GB of block
storage is genuinely useful. But two things to weigh: capacity for Arm instances
is <cite index="10-1">frequently unavailable in busy regions, throwing "Out of host capacity" errors</cite>, and a
provider that halves a headline free tier without notice may do it again. Fine
as a free box. Don't make it the only copy of your data.

---

## Comparison

| | GitHub Actions | Hetzner CX23 | Oracle Always Free |
|---|---|---|---|
| Cost | **$0** | ~€5.49/mo | **$0** |
| Setup effort | ~10 min | ~1 hour | ~1 hour + capacity luck |
| Ops burden | none | OS patching, backups | patching, backups |
| Cron reliability | best-effort, 10–30 min jitter | exact | exact |
| Data versioning | free, it's git | you build it | you build it |
| Long jobs | 6 h limit | unlimited | unlimited |
| Can serve the UI later | no | yes | yes |
| Risk | GitHub policy change | provider outage | tier cut without notice |

---

## Storage: smaller than you think

Measured on our own real output: Parquet + zstd lands at **53–62 bytes/row**.
Projecting the forward-collection sources:

| Source | Rows/day | Per year |
|---|---|---|
| 20 ETF issuer holdings files | ~12,000 | **263 MB** |
| FINRA daily short volume | ~10,000 | **219 MB** |
| CFTC COT (weekly) | ~200 | 4 MB |

So the entire archive grows at roughly **half a gigabyte per year**. That fits in
a private Git repo for years, with full history and diffs, at zero cost. Only
once you add intraday bars does object storage become worth the trouble —
Cloudflare R2's free tier (10 GB, no egress fees) is the natural next step, and
DuckDB reads Parquet straight off S3-compatible storage, so nothing in the code
has to change.

---

## What not to do

Skip Kubernetes, Airflow, managed Postgres, and anything with "platform" in the
name. This is one Python process writing files. Every layer you add is a thing
that can break at 22:30 UTC while you're asleep, and none of them make a
13-second job faster. The `PIPELINE` surface in the reel is a *visualisation* of
a DAG — it does not imply you need a DAG orchestrator to run four HTTP requests.

Also skip AWS/GCP/Azure for this. The free tiers are time-limited, the pricing
has many small edges, and a forgotten resource that bills $8/month is exactly the
thing you were trying to avoid.

---

## Recommendation

**Start with Option A today.** Push the repo private, set the two secrets, and
the collector begins accumulating point-in-time data tonight. It costs nothing,
takes ten minutes, and the only irreversible resource here is *time* — data you
didn't record.

Add a €5.50 Hetzner box later, when you want to serve the UI surfaces or run
overnight backtest sweeps. There is no reason to pay for one before then.
