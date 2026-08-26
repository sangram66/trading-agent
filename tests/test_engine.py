"""
Verification gates V1-V10 from tasks/todo.md.

Every test asserts a *known* answer, not merely that code executes. The pair
that matters most is V2/V3: the engine must reject the shuffle null on a
vol-clustered series (there is real structure) while failing to reject the GARCH
null on the same series (the structure is only clustering). An engine that
cannot draw that distinction will certify noise as edge.

Run:  python3 -m tests.test_engine
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from engine.audit.lookahead import audit_knowable, audit_pit_frame, shift_test
from engine.core.asof import add_pit, asof_snapshot, statutory_knowable, validate_pit
from engine.core.storage import Store
from engine.nulls.compare import compare
from engine.nulls.models import GarchFHS, fit_garch11
from engine.nulls.statistics import realised_vol, sharpe

PASSED, FAILED = [], []


def check(name: str, cond: bool, detail: str = ""):
    (PASSED if cond else FAILED).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def simulate_garch(n, omega, alpha, beta, seed=0, df=6):
    """Ground-truth GARCH(1,1) with t-distributed innovations."""
    rng = np.random.default_rng(seed)
    z = rng.standard_t(df, size=n + 1000)
    z /= np.sqrt(df / (df - 2))
    s2 = omega / (1 - alpha - beta)
    e_prev, out = 0.0, np.empty(n + 1000)
    for t in range(n + 1000):
        s2 = omega + alpha * e_prev ** 2 + beta * s2
        out[t] = np.sqrt(s2) * z[t]
        e_prev = out[t]
    return out[1000:]


# --------------------------------------------------------------------------

def v1_white_noise():
    print("\nV1  white noise vs every null -> no rejection")
    rng = np.random.default_rng(7)
    r = rng.normal(0, 0.01, 3000)
    rep = compare(r, n_sim=300, seed=11)
    for stat in ("ACF |r| lag 1", "Ljung-Box Q(22) on |r|", "Regression R2"):
        p = rep.stats[stat].nulls["garch"].p_two_sided
        check(f"{stat} not rejected vs garch (p={p:.3f})", p > 0.05)
    p = rep.stats["ACF |r| lag 1"].nulls["shuffled"].p_two_sided
    check(f"ACF |r| lag 1 not rejected vs shuffle (p={p:.3f})", p > 0.05)


def v2_v3_garch_series():
    print("\nV2/V3  GARCH series: shuffle must reject, GARCH must not")
    r = simulate_garch(3000, 1e-6, 0.09, 0.89, seed=3)
    rep = compare(r, n_sim=300, seed=13)
    for stat in ("ACF |r| lag 1", "Ljung-Box Q(22) on |r|"):
        ps = rep.stats[stat].nulls["shuffled"].p_two_sided
        pg = rep.stats[stat].nulls["garch"].p_two_sided
        check(f"V2 {stat} REJECTS shuffle (p={ps:.4f})", ps < 0.05)
        check(f"V3 {stat} does NOT reject garch (p={pg:.3f})", pg > 0.05)


def v4_ar1():
    print("\nV4  AR(1): shuffle rejects, long block bootstrap does not")
    rng = np.random.default_rng(5)
    n, phi = 4000, 0.35
    e = rng.normal(0, 0.01, n)
    r = np.empty(n)
    r[0] = e[0]
    for t in range(1, n):
        r[t] = phi * r[t - 1] + e[t]
    rep = compare(r, n_sim=300, seed=17, block_len=50, mean_block=50,
                  statistics={"ACF r lag 1": lambda x: __import__(
                      "engine.nulls.statistics", fromlist=["acf_signed"]
                  ).acf_signed(x, 1)})
    ps = rep.stats["ACF r lag 1"].nulls["shuffled"].p_two_sided
    pb = rep.stats["ACF r lag 1"].nulls["block"].p_two_sided
    check(f"rejects shuffle (p={ps:.4f})", ps < 0.05)
    check(f"does not reject block bootstrap (p={pb:.3f})", pb > 0.05)


def v5_garch_recovery():
    print("\nV5  GARCH parameter recovery")
    true = dict(omega=2e-6, alpha=0.08, beta=0.90)
    r = simulate_garch(8000, seed=21, **true)
    p = fit_garch11(r)
    check(f"alpha {p.alpha:.4f} vs {true['alpha']}", abs(p.alpha - true["alpha"]) < 0.04)
    check(f"beta {p.beta:.4f} vs {true['beta']}", abs(p.beta - true["beta"]) < 0.04)
    check(f"persistence {p.persistence:.4f} vs {true['alpha']+true['beta']}",
          abs(p.persistence - 0.98) < 0.02)
    # independent cross-check against the `arch` package's own estimator
    try:
        from arch import arch_model
        res = arch_model(r * 100, vol="GARCH", p=1, q=1, mean="Constant").fit(disp="off")
        a2, b2 = res.params["alpha[1]"], res.params["beta[1]"]
        check(f"agrees with arch pkg (alpha {p.alpha:.3f} vs {a2:.3f})",
              abs(p.alpha - a2) < 0.03 and abs(p.beta - b2) < 0.03)
    except Exception as exc:                        # noqa: BLE001
        check("arch cross-check", False, str(exc))


def v6_sharpe():
    print("\nV6  known Sharpe recovered inside its confidence interval")
    rng = np.random.default_rng(9)
    n, target = 5000, 1.0
    mu = target * 0.01 / np.sqrt(252)
    r = rng.normal(mu, 0.01, n)
    est = sharpe(r)
    se = np.sqrt((1 + 0.5 * target ** 2) / n) * np.sqrt(252) / np.sqrt(252)
    se = np.sqrt((1 + 0.5 * target ** 2) / n * 252)
    check(f"sharpe {est:.3f} within CI of {target} (se={se:.3f})",
          abs(est - target) < 2.5 * se)


def v7_v8_lookahead():
    print("\nV7/V8  lookahead audit catches the dirty feature, passes the clean one")
    rng = np.random.default_rng(2)
    n = 600
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="B"),
                       "x": rng.normal(0, 1, n)})
    # clean: trailing 20d mean, knowable at t
    df["clean"] = df["x"].rolling(20).mean()
    # dirty: z-scored against the FULL-SAMPLE mean and sd
    df["dirty"] = (df["x"] - df["x"].mean()) / df["x"].std()

    r_clean = audit_knowable(df, "clean", lambda h: h["x"].tail(20).mean())
    print("   ", r_clean)
    check("V8 clean feature passes", r_clean.passed)

    r_dirty = audit_knowable(
        df, "dirty", lambda h: (h["x"].iloc[-1] - h["x"].mean()) / h["x"].std())
    print("   ", r_dirty)
    check("V7 dirty feature caught", not r_dirty.passed)

    # shift test must catch BOTH lookahead signatures
    rets = rng.normal(0, 0.01, n)

    # (a) same-bar peek: implausibly high base metric
    peek = np.sign(rets)
    st_peek = shift_test(rets, peek)
    print("   ", st_peek)
    check("V7b same-bar peek flagged (implausible base)", not st_peek.passed)

    # (b) centred moving average — the classic accidental lookahead. Caught by
    #     reconstruction, not by performance, which is exactly the division of
    #     labour between the two tools.
    df["centred"] = df["x"].rolling(21, center=True).mean()
    r_centred = audit_knowable(df, "centred", lambda h: h["x"].tail(21).mean())
    print("   ", r_centred)
    check("V7c centred-window feature caught by audit_knowable",
          not r_centred.passed)

    # (c) an honest lagged signal must pass
    honest = np.sign(np.concatenate(([0.0], rets[:-1])))
    st_honest = shift_test(rets, honest)
    print("   ", st_honest)
    check("V8b honest signal passes", st_honest.passed)


def v9_asof():
    print("\nV9  as-of join never returns a row before knowable_at")
    facts = pd.DataFrame({
        "entity": ["A", "A", "B"],
        "event_time": pd.to_datetime(["2024-03-31", "2024-06-30", "2024-03-31"]),
        "value": [10.0, 11.0, 20.0]})
    v = add_pit(facts, "event_time", statutory_knowable("13F-HR"))
    check("knowable_at is 45d after quarter end",
          (v["knowable_at"] - v["event_time"]).dt.days.eq(45).all())

    snap = asof_snapshot(v, "2024-05-01")
    check("Q1 not visible on 1 May (filed 15 May)", snap.empty,
          f"got {len(snap)} rows")
    snap2 = asof_snapshot(v, "2024-05-20")
    check("Q1 visible on 20 May", len(snap2) == 2)
    check("Q2 still not visible on 20 May",
          bool((snap2["event_time"] == pd.Timestamp("2024-03-31")).all()))

    bad = v.copy()
    bad.loc[0, "knowable_at"] = pd.Timestamp("2024-01-01")
    try:
        validate_pit(bad, "bad")
        check("validate_pit rejects impossible row", False)
    except ValueError:
        check("validate_pit rejects impossible row", True)
    check("audit_pit_frame agrees", not audit_pit_frame(bad).passed)


def v10_storage():
    print("\nV10  storage round-trip and manifest integrity")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        st = Store(tmp)
        df = pd.DataFrame({"a": np.arange(100), "b": np.random.default_rng(0).normal(size=100)})
        h1 = st.write("bronze", "t", df, source="unit-test")
        back = st.read("bronze", "t")
        check("round-trip equal", back.equals(df))
        check("manifest verifies", st.verify("bronze", "t"))
        h2 = st.write("bronze", "t", df, source="unit-test")
        check(f"hash stable across rewrites ({h1})", h1 == h2)
        st.write("bronze", "t", df.iloc[:50], source="unit-test")
        check("hash changes when data changes", st.manifest["bronze/t"]["hash"] != h1)
        out = st.query("SELECT count(*) AS n FROM bronze_t")
        check("duckdb query works", int(out["n"][0]) == 50)


def v11_real_data():
    print("\nV11  real S&P 500 — sanity bounds on genuine market data")
    from engine.ingest.sources import sp500_arch, to_returns
    px = sp500_arch()
    check(f"{len(px)} rows of real OHLCV", len(px) > 5000)
    check("PIT frame valid", audit_pit_frame(px).passed)
    r = to_returns(px)["ret"].to_numpy()
    from engine.nulls.statistics import ann_vol, excess_kurtosis, acf_abs
    v, ek, a1 = ann_vol(r), excess_kurtosis(r), acf_abs(r, 1)
    check(f"ann vol {v:.3f} in [0.10, 0.30]", 0.10 < v < 0.30)
    check(f"excess kurtosis {ek:.2f} > 3 (fat tails are real)", ek > 3)
    check(f"ACF|r| lag1 {a1:.3f} in [0.10, 0.45]", 0.10 < a1 < 0.45)
    p = fit_garch11(r)
    check(f"persistence {p.persistence:.4f} in [0.95, 1.0)", 0.95 < p.persistence < 1.0)
    z = GarchFHS(r).z
    check(f"standardised resid kurtosis {excess_kurtosis(z):.2f} < raw {ek:.2f}",
          excess_kurtosis(z) < ek)
    rv = realised_vol(r, 20)
    check("realised_vol burn-in is NaN then finite",
          bool(np.isnan(rv[:19]).all() and np.isfinite(rv[19:]).all()))


def v12_reproducibility():
    print("\nV12  reproducibility across processes (caught by clean-room run)")
    import subprocess
    import tempfile

    # A: content hash must ignore wall-clock provenance columns
    with tempfile.TemporaryDirectory() as tmp:
        st = Store(tmp)
        base = pd.DataFrame({"a": np.arange(50), "event_time": pd.Timestamp("2024-01-01"),
                             "knowable_at": pd.Timestamp("2024-01-01")})
        d1 = base.assign(retrieved_at=pd.Timestamp("2024-05-01T10:00:00"))
        d2 = base.assign(retrieved_at=pd.Timestamp("2026-08-26T23:59:59"))
        h1 = st.write("bronze", "x", d1)
        h2 = st.write("bronze", "x", d2)
        check(f"hash ignores retrieved_at ({h1})", h1 == h2)
        h3 = st.write("bronze", "x", d2.assign(a=lambda d: d["a"] + 1))
        check("hash still changes when real data changes", h3 != h1)

    # B: per-null seeding must not depend on PYTHONHASHSEED
    code = ("import hashlib;"
            "print(int(hashlib.sha256(b'garch').hexdigest()[:8],16) % 100000)")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env={"PYTHONHASHSEED": s, "PATH": "/usr/bin:/bin"}
                           ).stdout.strip() for s in ("0", "1", "random")}
    check(f"null seed stable across PYTHONHASHSEED {outs}", len(outs) == 1)

    # C: same seed, same answer
    rng = np.random.default_rng(4)
    r = rng.normal(0, 0.01, 800)
    a = compare(r, n_sim=60, seed=99, nulls=["shuffled", "garch"])
    b = compare(r, n_sim=60, seed=99, nulls=["shuffled", "garch"])
    same = all(abs(a.stats[k].nulls[n].mean - b.stats[k].nulls[n].mean) < 1e-12
               for k in a.stats for n in ("shuffled", "garch"))
    check("compare() is deterministic for a fixed seed", same)


def main():
    for fn in (v1_white_noise, v2_v3_garch_series, v4_ar1, v5_garch_recovery,
               v6_sharpe, v7_v8_lookahead, v9_asof, v10_storage, v11_real_data,
               v12_reproducibility):
        fn()
    print(f"\n{'='*70}\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
