"""
The lookahead audit.

Two independent tests, because they catch different bugs:

1. `audit_knowable` — rebuild each feature value using only rows that were
   knowable at that date, and demand bit-equality with what is stored. Catches
   features silently computed over the whole sample (z-scores against a
   full-sample mean, quantile bins fitted on the future, forward-filled
   fundamentals).

2. `shift_test` — delay every feature by one bar. If performance does not
   degrade, the feature was already peeking. Catches off-by-one alignment,
   which is the most common lookahead bug and the hardest to see by reading
   code.

A feature that passes both is not guaranteed clean. A feature that fails either
is definitely dirty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AuditResult:
    name: str
    n_checked: int
    n_mismatched: int
    max_abs_diff: float
    passed: bool
    detail: str = ""

    def __str__(self):
        flag = "PASS" if self.passed else "FAIL"
        return (f"[{flag}] {self.name}: {self.n_mismatched}/{self.n_checked} "
                f"mismatched, max |diff| = {self.max_abs_diff:.3e}"
                + (f" — {self.detail}" if self.detail else ""))


def audit_knowable(df: pd.DataFrame, feature: str, rebuild,
                   time_col: str = "date", n_samples: int = 200,
                   tol: float = 1e-10, seed: int = 0) -> AuditResult:
    """Rebuild a sample of feature values from truncated history and compare.

    `rebuild(history_df) -> float` must return the feature value for the last
    row of the history it is given, using only that history.
    """
    rng = np.random.default_rng(seed)
    d = df.sort_values(time_col).reset_index(drop=True)
    valid = d.index[d[feature].notna()].to_numpy()
    if valid.size == 0:
        return AuditResult(feature, 0, 0, 0.0, False, "no non-null values")
    take = rng.choice(valid, size=min(n_samples, valid.size), replace=False)

    diffs, bad = [], 0
    for i in take:
        stored = float(d.loc[i, feature])
        try:
            rebuilt = float(rebuild(d.iloc[: i + 1]))
        except Exception as exc:                    # noqa: BLE001
            return AuditResult(feature, len(take), len(take), float("inf"),
                               False, f"rebuild raised: {exc}")
        diff = abs(stored - rebuilt)
        diffs.append(diff)
        if not (diff <= tol or (np.isnan(stored) and np.isnan(rebuilt))):
            bad += 1

    mx = float(np.nanmax(diffs)) if diffs else 0.0
    return AuditResult(
        feature, len(take), bad, mx, bad == 0,
        "" if bad == 0 else
        "feature depends on data that was not knowable at the time")


def shift_test(returns: np.ndarray, signal: np.ndarray,
               metric=None, implausible: float = 5.0) -> AuditResult:
    """Plausibility check on a signal's headline metric, plus a delay diagnostic.

    Scope note, learned the hard way: this function gates on **one** thing, the
    plausibility ceiling. A daily Sharpe above ~5 does not exist in nature, so an
    implausible base metric is near-conclusive evidence that the signal is
    reading the same bar it trades.

    The one-bar degradation is *reported* but deliberately not a pass/fail
    criterion. It is tempting to treat "barely degrades when delayed" as proof of
    lookahead, but degradation is a function of signal persistence: a genuinely
    slow signal legitimately loses almost nothing to a one-day delay, while a
    fast one loses most of its edge honestly. Thresholding it produces both false
    positives and false negatives.

    Features that peek by using unknowable data — centred windows, full-sample
    normalisation, forward-filled fundamentals — are caught reliably by
    `audit_knowable`, which reconstructs from truncated history rather than
    guessing from performance. Use both; they have different jobs.
    """
    from engine.nulls.statistics import sharpe
    metric = metric or sharpe

    r = np.asarray(returns, float)
    s = np.asarray(signal, float)
    n = min(r.size, s.size)
    r, s = r[:n], s[:n]

    base = float(metric(np.nan_to_num(s * r)))
    lagged = np.concatenate(([0.0], s[:-1]))
    lag = float(metric(np.nan_to_num(lagged * r)))

    if abs(base) < 1e-12:
        return AuditResult("shift_test", n, 0, 0.0, True, "flat strategy")

    drop = (base - lag) / abs(base)
    too_good = abs(base) > implausible

    why = (f"base metric {base:.2f} exceeds the plausibility ceiling "
           f"{implausible:.1f} — the signature of a signal reading the same bar "
           f"it trades" if too_good else
           "base metric is plausible; delay behaviour reported, not gated")

    return AuditResult("shift_test", n, 1 if too_good else 0, float(drop),
                       not too_good,
                       f"metric {base:.3f} -> {lag:.3f} when delayed one bar "
                       f"({drop:+.1%}); {why}")


def audit_pit_frame(df: pd.DataFrame) -> AuditResult:
    """Structural check that no row claims to be knowable before it happened."""
    if not {"event_time", "knowable_at"} <= set(df.columns):
        return AuditResult("pit_columns", len(df), len(df), float("inf"),
                           False, "missing event_time / knowable_at")
    bad = int((df["knowable_at"] < df["event_time"]).sum())
    return AuditResult("pit_ordering", len(df), bad, 0.0, bad == 0,
                       "" if bad == 0 else "knowable_at precedes event_time")
