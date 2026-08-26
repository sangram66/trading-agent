"""
Verification gates for the multi-agent guardrails.

The claim under test is narrow and important: adding agents must not make it
easier to produce a false discovery. Gate M4 is the one that matters — ten
agents sharing a budget must face a strictly higher bar than one agent, because
otherwise parallelism is just faster self-deception.

Run:  python3 -m tests.test_agent
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from agent.ledger import TrialLedger, deflated_sharpe, expected_max_sharpe
from agent.tools import AgentTools, HypothesisRegistry, PreRegistrationError

PASSED, FAILED = [], []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


GOOD = dict(
    mechanism=("Quarter-end rebalancing forces target-weight funds to sell the "
               "outperforming asset regardless of price."),
    who_is_forced="Defined-benefit pension funds and target-date funds",
    counterparty="Market makers and stat-arb desks, compensated for inventory risk",
    predicted_sign="negative in the outperforming leg",
    predicted_magnitude_bp=[4, 15],
    predicted_timing="T-3 to T-0 of quarter end, in the closing auction",
    falsifier="No relationship between prior-quarter dispersion and quarter-end drift",
)


def series_with_sharpe(ann_sr, n, rng, vol=0.01, periods=252):
    """Returns with an *exactly* known realised Sharpe.

    Drawing from a normal and hoping gives a realised Sharpe that wanders by
    +/-0.3 around the target, which is fine for research and useless in a test
    asserting behaviour at a threshold. Standardising first removes the sampling
    noise so the assertion is about the gate, not about the draw.
    """
    r = rng.normal(0, vol, n)
    r = (r - r.mean()) / r.std(ddof=1)
    return (r + ann_sr / np.sqrt(periods)) * vol


def fresh(tmp):
    reg = HypothesisRegistry(Path(tmp) / "hyp")
    led = TrialLedger(Path(tmp) / "budget.json")
    return reg, led


def m1_preregistration():
    print("\nM1  no returns before a registered hypothesis")
    with tempfile.TemporaryDirectory() as tmp:
        reg, led = fresh(tmp)
        t = AgentTools("athena", led, reg, returns_provider=lambda h, **k: np.zeros(10))
        try:
            t.get_returns("H-0001")
            check("unregistered get_returns blocked", False)
        except PreRegistrationError as e:
            check("unregistered get_returns blocked", True, str(e)[:60])

        hid = t.register_hypothesis(**GOOD)
        check(f"registration returns an id ({hid})", hid == "H-0001")
        arr = t.get_returns(hid)
        check("registered get_returns allowed", arr.size == 10)


def m2_placeholders():
    print("\nM2  placeholder and empty fields rejected")
    with tempfile.TemporaryDirectory() as tmp:
        reg, led = fresh(tmp)
        t = AgentTools("chartist", led, reg)
        for bad, label in [({**GOOD, "falsifier": "TBD"}, "falsifier=TBD"),
                           ({**GOOD, "mechanism": ""}, "empty mechanism"),
                           ({**GOOD, "counterparty": "someone"}, "vague counterparty")]:
            try:
                t.register_hypothesis(**bad)
                check(f"rejects {label}", False)
            except PreRegistrationError:
                check(f"rejects {label}", True)


def m3_cross_agent():
    print("\nM3  agents cannot test each other's hypotheses")
    with tempfile.TemporaryDirectory() as tmp:
        reg, led = fresh(tmp)
        a = AgentTools("athena", led, reg, returns_provider=lambda h, **k: np.zeros(10))
        b = AgentTools("oracle", led, reg, returns_provider=lambda h, **k: np.zeros(10))
        hid = a.register_hypothesis(**GOOD)
        try:
            b.get_returns(hid)
            check("borrowing another agent's registration blocked", False)
        except PreRegistrationError:
            check("borrowing another agent's registration blocked", True)


def m4_shared_budget():
    print("\nM4  the bar rises as agents consume the shared budget")
    with tempfile.TemporaryDirectory() as tmp:
        reg, led = fresh(tmp)
        rng = np.random.default_rng(0)
        bars = []
        for i in range(60):
            led.record(f"H-{i:04d}", f"agent{i % 10}", rng.normal(0, 0.01, 1000))
            bars.append(led.threshold_annual())
        check(f"threshold rises with trials ({bars[0]:.2f} -> {bars[-1]:.2f})",
              bars[-1] > bars[0])
        check("monotone non-decreasing in the long run",
              bars[-1] > bars[len(bars) // 2])
        check(f"10 agents charged the same ledger ({led.n_trials} trials)",
              led.n_trials == 60 and len(led.by_agent()) == 10)


def m5_noise_rejected():
    print("\nM5  pure noise is rejected, real edge survives the search cost")
    with tempfile.TemporaryDirectory() as tmp:
        reg, led = fresh(tmp)
        rng = np.random.default_rng(7)
        # 200 worthless strategies, mined for the luckiest
        best, best_sr = None, -9
        for i in range(200):
            r = rng.normal(0, 0.01, 1500)
            v = led.record(f"H-{i:04d}", f"agent{i % 10}", r)
            if v.sharpe_annual > best_sr:
                best_sr, best = v.sharpe_annual, v
        passes = sum(t.passed for t in led.trials)
        check(f"data-mined noise: {passes}/200 passed (want 0)", passes == 0)
        check(f"luckiest noise Sharpe {best_sr:.2f} still rejected", not best.passed)
        print(f"      bar after 200 trials: annualised Sharpe "
              f"{led.threshold_annual():.2f}")

        # The search cost is real: after 200 trials a Sharpe of 1.8 on six years
        # of daily data is NOT established at 95%. That is the correct answer,
        # not an over-strict gate.
        v_short = led.record("H-9998", "oracle",
                             series_with_sharpe(1.8, 1500, rng))
        check(f"Sharpe {v_short.sharpe_annual:.2f} on 6y rejected after 200 "
              f"trials (DSR {v_short.dsr:.3f})", not v_short.passed)

        # The same edge with ten years of evidence clears the bar, so the gate
        # discriminates rather than simply always saying no.
        v_long = led.record("H-9999", "oracle",
                            series_with_sharpe(1.8, 2600, rng))
        check(f"same edge on 10y passes (Sharpe {v_long.sharpe_annual:.2f}, "
              f"DSR {v_long.dsr:.3f})", v_long.passed)


def m6_maths():
    print("\nM6  threshold maths behaves")
    v = 1.0 / 1000
    e1, e10, e1000 = (expected_max_sharpe(n, v) for n in (1, 10, 1000))
    check(f"1 trial -> 0 bar ({e1:.4f})", abs(e1) < 1e-9)
    check(f"more trials -> higher bar ({e10:.4f} < {e1000:.4f})", e10 < e1000)
    check("grows sub-linearly (sqrt-log)", e1000 < 4 * e10)
    d_few = deflated_sharpe(0.06, 1500, 5, v)
    d_many = deflated_sharpe(0.06, 1500, 5000, v)
    check(f"same Sharpe, more trials -> lower DSR ({d_few:.3f} > {d_many:.3f})",
          d_few > d_many)
    d_fat = deflated_sharpe(0.06, 1500, 5, v, skew=-1.5, kurtosis=12.0)
    check(f"fat tails penalised ({d_fat:.3f} < {d_few:.3f})", d_fat < d_few)


def m7_persistence():
    print("\nM7  ledger survives restarts and has no reset")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "budget.json"
        led = TrialLedger(p)
        rng = np.random.default_rng(3)
        for i in range(5):
            led.record(f"H-{i:04d}", "pilot", rng.normal(0, 0.01, 500))
        again = TrialLedger(p)
        check(f"reloaded {again.n_trials} trials", again.n_trials == 5)
        check("no reset/delete method exposed",
              not any(hasattr(led, m) for m in ("reset", "clear", "delete")))
        again.record("H-0005", "pilot", rng.normal(0, 0.01, 500))
        check("continues from reloaded count", again.n_trials == 6)


def main():
    for fn in (m1_preregistration, m2_placeholders, m3_cross_agent,
               m4_shared_budget, m5_noise_rejected, m6_maths, m7_persistence):
        fn()
    print(f"\n{'=' * 70}\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
