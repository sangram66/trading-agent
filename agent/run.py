"""
Agent run orchestrator. Entry point for `agent-run.yml`.

Honest scope note, because this is the file where it would be easiest to fake
progress: **the hypothesis-proposing agents are not wired yet, and cannot be
until the backtester and cost model exist.** ATHENA can happily generate a
beautiful hypothesis today, but there is nothing that can turn it into a returns
series net of spread, impact and borrow — so any verdict would be fiction, and
it would be a fiction permanently recorded in the shared ledger.

So this run does the part that is genuinely real today: ORACLE. It takes every
return series in the store and runs the null-comparison engine over it, which
needs no LLM and no backtester, and writes findings to the research journal.

It deliberately charges **zero trials**. A null comparison is not a strategy
test — nothing is being selected on performance — and charging the shared
multiple-testing budget for it would inflate the bar for no reason.

    python3 -m agent.run --max-trials 5 --agents athena
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from agent.ledger import TrialLedger
from agent.tools import HypothesisRegistry

# Roles that exist as deterministic modules today.
IMPLEMENTED = {"oracle"}
# Roles that need the backtester before they can produce an honest verdict.
BLOCKED_ON_BACKTESTER = {"athena", "chartist", "nimbus", "sentinel", "pilot"}


def run_oracle(store, n_sim: int, out_dir: Path) -> list:
    """Null comparison over every return series in the store.

    This is the ORACLE node from the desk topology, and it is deterministic on
    purpose: an LLM must never be the thing that decides whether a statistic
    survived its null.
    """
    from engine.nulls.compare import compare

    findings = []
    for key in sorted(store.manifest):
        layer, dataset = key.split("/", 1)
        try:
            df = store.read(layer, dataset)
        except Exception:                              # noqa: BLE001
            continue

        col = next((c for c in ("ret", "adj_close", "close") if c in df.columns), None)
        if col is None or len(df) < 500:
            continue

        x = df[col].to_numpy(float)
        r = x if col == "ret" else np.diff(np.log(x[x > 0]))
        r = r[np.isfinite(r)]
        if r.size < 500:
            continue

        print(f"  oracle · {key} ({r.size} obs)")
        rep = compare(r, n_sim=n_sim, seed=42)
        survivors = [n for n, s in rep.stats.items() if s.survives()]
        print(f"      {rep.verdict()[:100]}")

        findings.append({
            "dataset": key,
            "n_obs": int(r.size),
            "n_sim": n_sim,
            "garch": {"alpha": rep.garch_params.alpha,
                      "beta": rep.garch_params.beta,
                      "persistence": rep.garch_params.persistence},
            "survives_all_nulls": survivors,
            "statistics": {n: {"real": s.real,
                               "p_garch": s.nulls["garch"].p_two_sided}
                           for n, s in rep.stats.items()},
        })

    if findings:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"oracle-{date.today().isoformat()}.json"
        p.write_text(json.dumps(
            {"run_at": datetime.now(timezone.utc).isoformat(),
             "findings": findings}, indent=2))
        print(f"  wrote {p}")
    return findings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-trials", type=int, default=5,
                    help="cap on shared-budget trials this run may charge")
    ap.add_argument("--agents", default="athena",
                    help="comma-separated roles to run")
    ap.add_argument("--n-sim", type=int, default=500)
    args = ap.parse_args(argv)

    requested = {a.strip().lower() for a in args.agents.split(",") if a.strip()}
    ledger = TrialLedger()
    registry = HypothesisRegistry()

    print(f"agent run · {date.today().isoformat()}")
    print(f"requested: {', '.join(sorted(requested)) or '(none)'}")
    print(f"trial cap this run: {args.max_trials}\n")
    print("budget before")
    print("  " + ledger.summary().replace("\n", "\n  ") + "\n")

    from engine.core.storage import Store
    store = Store("data")

    # Always run ORACLE — it is the one role that is real today, and a run that
    # did nothing would still cost a workflow slot.
    findings = run_oracle(store, args.n_sim, Path("research/findings"))

    blocked = requested & BLOCKED_ON_BACKTESTER
    if blocked:
        print(f"\n  NOT RUN: {', '.join(sorted(blocked))}")
        print("  These propose or approve strategies, which requires a "
              "backtester with a\n  real cost model (spread, sqrt(Q/ADV) "
              "impact, borrow). Without it any\n  verdict would be optimistic "
              "fiction — and permanently recorded.")
        print("  Build engine/backtest/ first. See tasks/todo.md.")

    unknown = requested - IMPLEMENTED - BLOCKED_ON_BACKTESTER
    if unknown:
        print(f"\n  UNKNOWN ROLES: {', '.join(sorted(unknown))}")

    print("\nbudget after")
    print("  " + ledger.summary().replace("\n", "\n  "))
    print(f"\n{len(findings)} dataset(s) analysed, "
          f"{ledger.n_trials} trials charged in total "
          f"(this run charged 0 — null comparison is not a strategy test)")
    print(f"open hypotheses on file: {len(registry.all())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
