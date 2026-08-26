"""
The VOLATILITY surface, on real data.

Reproduces the reel's `THREE WORLDS, ONE PIPELINE` panel against 5,031 days of
actual S&P 500 history, and writes everything to the store with a manifest so
the numbers can be reproduced byte-for-byte later.

    python3 run_vol_lab.py
"""

from __future__ import annotations

import numpy as np

from engine.core.storage import Store
from engine.ingest.sources import sp500_arch, to_returns
from engine.nulls.compare import compare
from engine.nulls.statistics import (independence_baseline, same_band_rate,
                                     vol_bands)

WINDOW, K, HORIZON, N_SIM = 20, 5, 5, 1000


def main():
    store = Store("data")

    px = sp500_arch()
    store.write("bronze", "sp500_daily", px,
                source="arch package (bundled real S&P 500 OHLCV)")
    rets = to_returns(px)
    store.write("silver", "sp500_returns", rets, source="bronze/sp500_daily")

    r = rets["ret"].to_numpy(float)
    print(f"S&P 500 · {len(r):,} trading days · "
          f"{rets['date'].min():%Y-%m-%d} to {rets['date'].max():%Y-%m-%d}\n")

    rep = compare(r, n_sim=N_SIM, seed=42, window=WINDOW, k=K, horizon=HORIZON)

    p = rep.garch_params
    print(f"GARCH(1,1) fit   omega={p.omega:.3e}  alpha={p.alpha:.4f}  "
          f"beta={p.beta:.4f}  persistence={p.persistence:.5f}  "
          f"vol half-life={p.half_life():.0f}d\n")

    print(rep.table())
    print()
    print(rep.pvalue_table())
    print()

    # ---- the trap the reel's own UI warns about -------------------------
    sb = same_band_rate(r, WINDOW, K, HORIZON)
    naive_base = 1.0 / K
    honest_base = rep.stats["Same-band rate"].nulls["shuffled"].mean
    garch_base = rep.stats["Same-band rate"].nulls["garch"].mean
    print("SAME-BAND RATE — WHAT THE BASELINE SHOULD BE")
    print(f"  real                         {sb:6.1%}")
    print(f"  naive 1/k independence       {naive_base:6.1%}   "
          f"-> claims +{(sb-naive_base)*100:.1f} pts of lift")
    print(f"  shuffled null (honest)       {honest_base:6.1%}   "
          f"-> real lift +{(sb-honest_base)*100:.1f} pts")
    print(f"  garch null                   {garch_base:6.1%}   "
          f"-> lift beyond clustering +{(sb-garch_base)*100:.1f} pts")
    print(f"\n  A {WINDOW}-day vol window compared {HORIZON} days apart shares "
          f"{WINDOW-HORIZON} of its {WINDOW} observations.")
    print("  That mechanical overlap alone lifts the shuffled world far above")
    print("  1/k. Measuring against 1/k would book most of it as information.")

    # ---- regime tape ---------------------------------------------------
    bands = vol_bands(r, WINDOW, K)
    valid = bands[bands >= 0]
    print(f"\nREGIME TAPE  ({len(valid):,} banded days)")
    for b in range(K):
        share = np.mean(valid == b)
        print(f"  band {b}  {'#' * int(share * 60)} {share:5.1%}")

    print("\nVERDICT")
    print(" ", rep.verdict())

    rows = [{"statistic": n, "real": s.real,
             **{f"{nn}_mean": s.nulls[nn].mean for nn in rep.null_names},
             **{f"{nn}_p": s.nulls[nn].p_two_sided for nn in rep.null_names},
             "survives_all": s.survives()}
            for n, s in rep.stats.items()]
    import pandas as pd
    store.write("gold", "vol_lab_three_worlds", pd.DataFrame(rows),
                source="silver/sp500_returns",
                note=f"n_sim={N_SIM} window={WINDOW} k={K} horizon={HORIZON}")

    print("\nSTORE")
    print(store.summary().to_string(index=False))


if __name__ == "__main__":
    main()
