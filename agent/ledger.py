"""
The shared trial ledger.

This is the single most important module in a multi-agent setup, and the reason
is arithmetic rather than philosophical.

One researcher testing 20 hypotheses will stumble onto a Sharpe of 2 by luck.
Ten agents testing in parallel reach that point ten times faster. A multi-agent
desk without a *shared* multiple-testing budget is not ten researchers — it is
one researcher with ten times the opportunity to fool themselves, running
unsupervised, overnight.

So the ledger is global, append-only, and no agent can reset it. Every backtest
any agent runs increments it, and the Sharpe required to pass the gate rises
accordingly. Agents compete for a scarce shared resource; they do not each get
a fresh budget.

Threshold maths follows Bailey & Lopez de Prado: the expected maximum Sharpe
under the null grows with the number of trials, so the honest bar is not "is
this Sharpe good" but "is this Sharpe better than the best of N coin flips".
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


# --------------------------------------------------------------------------
# multiple-testing maths
# --------------------------------------------------------------------------

def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum Sharpe across `n_trials` independent worthless strategies.

    The bar a strategy must clear simply to be more impressive than the luckiest
    of N coin flips. Grows like sqrt(log N), so it rises slowly — but it never
    stops rising, and after a few hundred trials it is brutal.

    Units are per-period (not annualised), matching `deflated_sharpe`.
    """
    if n_trials < 1:
        return 0.0
    if n_trials == 1:
        return 0.0
    sd = np.sqrt(max(sr_variance, 1e-12))
    n = float(n_trials)
    a = stats.norm.ppf(1.0 - 1.0 / n)
    b = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(sd * ((1.0 - EULER_MASCHERONI) * a + EULER_MASCHERONI * b))


def deflated_sharpe(sr: float, n_obs: int, n_trials: int,
                    sr_variance: float, skew: float = 0.0,
                    kurtosis: float = 3.0) -> float:
    """Probability that the true Sharpe exceeds zero, given the search that found it.

    Corrects for three things a raw Sharpe ignores: how many strategies were
    tried, how non-normal the returns are, and how short the sample is. All
    Sharpes here are per-period.

    Returns a probability. Treat < 0.95 as "not established".
    """
    sr_star = expected_max_sharpe(n_trials, sr_variance)
    denom = 1.0 - skew * sr + 0.25 * (kurtosis - 1.0) * sr ** 2
    if denom <= 0 or n_obs < 2:
        return float("nan")
    z = (sr - sr_star) * np.sqrt(n_obs - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


# --------------------------------------------------------------------------
# the ledger
# --------------------------------------------------------------------------

@dataclass
class Trial:
    trial_id: int
    hypothesis_id: str
    agent: str
    sharpe: float                 # per-period
    n_obs: int
    timestamp: str
    passed: bool = False
    note: str = ""


@dataclass
class Verdict:
    hypothesis_id: str
    agent: str
    sharpe_annual: float
    n_trials_at_test: int
    threshold_annual: float
    dsr: float
    passed: bool
    reason: str


class TrialLedger:
    """Append-only, shared across every agent, persisted to disk.

    Deliberately has no `reset()` and no `delete()`. An agent that could clear
    its own record of failed attempts would be able to launder an overfit
    strategy into a clean one, and an LLM asked to "start fresh" will reach for
    exactly that button if you give it one.
    """

    def __init__(self, path: str | Path = "research/trial_budget.json",
                 periods_per_year: int = 252):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.periods = periods_per_year
        self.trials: list = []
        self._load()

    # -- persistence ----------------------------------------------------
    def _load(self):
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self.trials = [Trial(**t) for t in raw.get("trials", [])]

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"trials": [asdict(t) for t in self.trials]}, indent=2))
        os.replace(tmp, self.path)          # atomic: a crash mid-write cannot
                                            # truncate the ledger

    # -- state ----------------------------------------------------------
    @property
    def n_trials(self) -> int:
        return len(self.trials)

    def sharpe_variance(self) -> float:
        """Variance of Sharpes observed so far — the scale of luck in this search.

        Before enough trials exist to estimate it, fall back to the theoretical
        variance of a Sharpe estimate under the null, 1/n_obs. Guessing low here
        would make the threshold too lenient, so the fallback is deliberately
        the more conservative of the two.
        """
        if len(self.trials) < 10:
            n = self.trials[-1].n_obs if self.trials else 1000
            return 1.0 / max(n, 1)
        return float(np.var([t.sharpe for t in self.trials], ddof=1))

    def threshold(self, n_trials: int | None = None) -> float:
        """Per-period Sharpe a new strategy must beat, given the search so far."""
        n = self.n_trials + 1 if n_trials is None else n_trials
        return expected_max_sharpe(n, self.sharpe_variance())

    def threshold_annual(self) -> float:
        return self.threshold() * np.sqrt(self.periods)

    # -- the only mutator -----------------------------------------------
    def record(self, hypothesis_id: str, agent: str, returns: np.ndarray,
               note: str = "") -> Verdict:
        """Charge one trial to the shared budget and rule on it.

        Called by the gate, not by agents directly. The charge happens whether
        the strategy passes or fails — that is the entire point. A failed test
        still consumed a lottery ticket.
        """
        r = np.asarray(returns, float)
        r = r[np.isfinite(r)]
        sd = float(np.std(r, ddof=1))
        sr = float(r.mean() / sd) if sd > 0 else 0.0

        n_at_test = self.n_trials + 1
        thresh = expected_max_sharpe(n_at_test, self.sharpe_variance())
        dsr = deflated_sharpe(sr, r.size, n_at_test, self.sharpe_variance(),
                              skew=float(stats.skew(r)),
                              kurtosis=float(stats.kurtosis(r, fisher=False)))
        passed = bool(sr > thresh and dsr > 0.95)

        self.trials.append(Trial(
            trial_id=n_at_test, hypothesis_id=hypothesis_id, agent=agent,
            sharpe=sr, n_obs=int(r.size),
            timestamp=datetime.now(timezone.utc).isoformat(),
            passed=passed, note=note))
        self._save()

        ann = np.sqrt(self.periods)
        if passed:
            reason = (f"Sharpe {sr*ann:.2f} clears the {n_at_test}-trial bar of "
                      f"{thresh*ann:.2f} with DSR {dsr:.3f}")
        elif sr <= thresh:
            reason = (f"Sharpe {sr*ann:.2f} does not clear the {n_at_test}-trial "
                      f"bar of {thresh*ann:.2f} — this is what {n_at_test} "
                      f"searches buys you by luck alone")
        else:
            reason = (f"Sharpe {sr*ann:.2f} clears the bar but DSR is only "
                      f"{dsr:.3f} (need > 0.95); sample too short or too "
                      f"fat-tailed to be sure")

        return Verdict(hypothesis_id, agent, sr * ann, n_at_test,
                       thresh * ann, dsr, passed, reason)

    # -- reporting ------------------------------------------------------
    def by_agent(self) -> dict:
        out: dict = {}
        for t in self.trials:
            d = out.setdefault(t.agent, {"trials": 0, "passed": 0})
            d["trials"] += 1
            d["passed"] += int(t.passed)
        return out

    def summary(self) -> str:
        ann = np.sqrt(self.periods)
        lines = [f"trials used: {self.n_trials}",
                 f"current bar: annualised Sharpe {self.threshold_annual():.2f} "
                 f"(trial #{self.n_trials + 1})",
                 "per agent:"]
        for agent, d in sorted(self.by_agent().items()):
            lines.append(f"  {agent:12s} {d['trials']:4d} trials  "
                         f"{d['passed']:3d} passed")
        return "\n".join(lines)
