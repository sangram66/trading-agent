"""
Point-in-time correctness.

Three timestamps, on every row, always:

    event_time    what period the row describes
    knowable_at   when the market could first have known it
    retrieved_at  when we pulled it

Every research join is AS OF `knowable_at`. Never `event_time`. A 13F describing
Q1 has event_time 2026-03-31 and knowable_at 2026-05-15; joining on the former
hands you six weeks of foresight and manufactures alpha out of nothing. This is
the failure mode that survives every other test, because the resulting backtest
looks perfectly well-behaved.
"""

from __future__ import annotations

import pandas as pd

REQUIRED = ("event_time", "knowable_at")


def validate_pit(df: pd.DataFrame, name: str = "frame") -> None:
    """Structural check. Raises rather than warns — a silent PIT violation is
    worse than a crash."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing point-in-time columns {missing}")
    bad = df["knowable_at"] < df["event_time"]
    if bad.any():
        n = int(bad.sum())
        raise ValueError(
            f"{name}: {n} rows are knowable before they happened — "
            f"knowable_at < event_time. This is a lookahead bug, not a warning.")


def asof_snapshot(vintages: pd.DataFrame, asof, key: str = "entity",
                  value_cols: list | None = None) -> pd.DataFrame:
    """What did we know about each entity as of `asof`?

    Takes the latest row per entity whose knowable_at <= asof. Restatements are
    handled correctly for free: an amended filing is simply a later row, so
    before its filing date you still see the original — which is what the market
    saw, and therefore what the backtest must see.
    """
    asof = pd.Timestamp(asof)
    v = vintages[vintages["knowable_at"] <= asof]
    if v.empty:
        return v.iloc[0:0]
    v = v.sort_values("knowable_at")
    out = v.groupby(key, as_index=False).tail(1)
    if value_cols:
        out = out[[key, "event_time", "knowable_at", *value_cols]]
    return out.reset_index(drop=True)


def asof_join(left: pd.DataFrame, right: pd.DataFrame,
              left_time: str = "date", by: str | None = None,
              suffix: str = "_r") -> pd.DataFrame:
    """Backward as-of join: attach the most recent `right` row that was knowable
    at or before each `left` timestamp.

    Thin wrapper over pandas.merge_asof, but it enforces the join key rather
    than trusting the caller to pass the right column — which is precisely where
    this goes wrong in practice.
    """
    validate_pit(right, "right")
    lt = left.sort_values(left_time)
    rt = right.sort_values("knowable_at")
    kw = dict(left_on=left_time, right_on="knowable_at",
              direction="backward", suffixes=("", suffix))
    if by:
        kw["by"] = by
    return pd.merge_asof(lt, rt, **kw)


def add_pit(df: pd.DataFrame, event_time, knowable_at, retrieved_at=None):
    """Stamp a frame with its three timestamps.

    `knowable_at` may be a column name, a fixed timestamp, or a callable applied
    to event_time (e.g. the statutory filing lag for a given form type).
    """
    out = df.copy()
    out["event_time"] = (out[event_time] if isinstance(event_time, str)
                         and event_time in out else pd.Timestamp(event_time))
    if callable(knowable_at):
        out["knowable_at"] = out["event_time"].map(knowable_at)
    elif isinstance(knowable_at, str) and knowable_at in out:
        out["knowable_at"] = out[knowable_at]
    else:
        out["knowable_at"] = pd.Timestamp(knowable_at)
    out["retrieved_at"] = pd.Timestamp(retrieved_at or pd.Timestamp.utcnow())
    validate_pit(out, "add_pit")
    return out


# Statutory filing lags — worst case, i.e. the last day the market could still
# have been ignorant. Using the actual filing date when you have it is better;
# these are the fallback when you only know the period.
FILING_LAG_DAYS = {
    "13F-HR": 45,
    "NPORT-P": 60,
    "10-Q": 45,
    "10-K": 90,
    "4": 2,
    "SC 13D": 10,
    "SC 13G": 45,
    "COT": 3,
}


def statutory_knowable(form: str):
    """Callable mapping event_time -> knowable_at for a given filing type."""
    days = FILING_LAG_DAYS.get(form)
    if days is None:
        raise KeyError(f"no statutory lag registered for form {form!r}")
    return lambda t: pd.Timestamp(t) + pd.Timedelta(days=days)
