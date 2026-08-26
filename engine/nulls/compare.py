"""
THREE WORLDS, ONE PIPELINE.

Runs an identical statistic set over the real series and over every surrogate
world, and reports what survives.

The contract this enforces: no statistic is evidence until it has beaten a null
that could have produced it by accident. `p` here is empirical — the fraction of
surrogate worlds that matched or exceeded the real value — never a table lookup,
because the asymptotic distributions of most of these statistics are wrong on
real financial data.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .models import build_generators
from .statistics import default_statistics


@dataclass
class NullResult:
    name: str
    mean: float
    sd: float
    p_greater: float          # P(null >= real)
    p_two_sided: float
    z: float                  # (real - null mean) / null sd


@dataclass
class StatResult:
    name: str
    real: float
    nulls: dict = field(default_factory=dict)   # name -> NullResult

    def survives(self, alpha: float = 0.05) -> bool:
        """True only if every null is rejected. Weakest link decides."""
        return all(n.p_two_sided < alpha for n in self.nulls.values())

    def hardest_null(self):
        """The null that came closest to explaining this statistic."""
        if not self.nulls:
            return None
        return max(self.nulls.values(), key=lambda n: n.p_two_sided)


@dataclass
class ComparisonReport:
    stats: dict                     # name -> StatResult
    null_names: list
    n_sim: int
    n_obs: int
    garch_params: object = None

    def table(self) -> str:
        """The `THREE WORLDS, ONE PIPELINE` panel as text."""
        w = max(len(s) for s in self.stats) + 2
        head = f"{'STATISTIC':<{w}}{'REAL':>12}"
        for nn in self.null_names:
            head += f"{nn.upper():>12}"
        head += f"{'VERDICT':>14}"
        lines = [head, "-" * len(head)]
        for name, sr in self.stats.items():
            row = f"{name:<{w}}{sr.real:>12.4g}"
            for nn in self.null_names:
                row += f"{sr.nulls[nn].mean:>12.4g}"
            hardest = sr.hardest_null()
            mark = "survives" if sr.survives() else f"~{hardest.name}"
            row += f"{mark:>14}"
            lines.append(row)
        return "\n".join(lines)

    def pvalue_table(self) -> str:
        w = max(len(s) for s in self.stats) + 2
        head = f"{'STATISTIC':<{w}}" + "".join(
            f"{('p ' + nn):>12}" for nn in self.null_names)
        lines = [head, "-" * len(head)]
        for name, sr in self.stats.items():
            row = f"{name:<{w}}" + "".join(
                f"{sr.nulls[nn].p_two_sided:>12.4f}" for nn in self.null_names)
            lines.append(row)
        return "\n".join(lines)

    def verdict(self, alpha: float = 0.05) -> str:
        survived = [n for n, s in self.stats.items() if s.survives(alpha)]
        if not survived:
            return ("NO SIGNAL — every statistic is reproducible by at least one "
                    "null. There is nothing here the nulls cannot explain.")
        return (f"{len(survived)}/{len(self.stats)} statistics survive every null: "
                + ", ".join(survived))


def _empirical_p(real: float, dist: np.ndarray) -> tuple:
    """(1 + count) / (1 + n) — never returns exactly zero.

    A p-value of 0 would claim more resolution than n_sim can support, and an
    automated loop will happily treat it as certainty.
    """
    d = dist[np.isfinite(dist)]
    if d.size == 0 or not np.isfinite(real):
        return float("nan"), float("nan")
    n = d.size
    p_ge = (1.0 + np.sum(d >= real)) / (n + 1.0)
    p_le = (1.0 + np.sum(d <= real)) / (n + 1.0)
    return float(p_ge), float(min(1.0, 2.0 * min(p_ge, p_le)))


def compare(r: np.ndarray,
            statistics: dict | None = None,
            nulls: list | None = None,
            n_sim: int = 1000,
            seed: int = 0,
            block_len: int = 20,
            mean_block: int = 20,
            window: int = 20,
            k: int = 5,
            horizon: int = 5) -> ComparisonReport:
    """Run every statistic over the real series and every surrogate world.

    Surrogates are generated in chunks and reduced to statistics immediately, so
    peak memory stays at chunk_size x n rather than n_sim x n.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    statistics = statistics or default_statistics(window, k, horizon)
    generators, fhs = build_generators(r, block_len, mean_block)
    nulls = nulls or list(generators)

    real_vals = {name: float(fn(r)) for name, fn in statistics.items()}
    dists = {name: {} for name in statistics}

    for null_name in nulls:
        gen = generators[null_name]
        # Python salts hash() on strings per process (PYTHONHASHSEED), so using
        # it here would silently give a different seed on every run and make
        # results irreproducible across machines. Use a stable digest instead.
        offset = int(hashlib.sha256(null_name.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed + offset % 100_000)
        acc = {name: [] for name in statistics}
        remaining, chunk = n_sim, 200
        while remaining > 0:
            m = min(chunk, remaining)
            sims = gen(m, rng)
            for row in sims:
                for name, fn in statistics.items():
                    acc[name].append(fn(row))
            remaining -= m
        for name in statistics:
            dists[name][null_name] = np.asarray(acc[name], dtype=float)

    stats = {}
    for name in statistics:
        sr = StatResult(name=name, real=real_vals[name])
        for null_name in nulls:
            d = dists[name][null_name]
            p_ge, p_two = _empirical_p(real_vals[name], d)
            sd = float(np.nanstd(d, ddof=1))
            mean = float(np.nanmean(d))
            z = (real_vals[name] - mean) / sd if sd > 0 else float("nan")
            sr.nulls[null_name] = NullResult(null_name, mean, sd, p_ge, p_two, z)
        stats[name] = sr

    return ComparisonReport(stats=stats, null_names=list(nulls), n_sim=n_sim,
                            n_obs=r.size, garch_params=fhs.params)
