"""
Null models — the surrogate worlds.

Each model takes a real return series and manufactures alternative histories
that keep some features and destroy others. What a statistic can still see
after a given null has done its work is what that null cannot explain.

    iid_shuffle          keeps the marginal distribution, destroys all ordering
    circular_block       keeps dependence up to the block length
    stationary_bootstrap same idea, random block lengths (Politis-Romano)
    GarchFHS             keeps vol clustering and fat tails, destroys everything else

The last one is the load-bearing null. Most apparent "edge" in a daily series is
volatility clustering wearing a hat, and GARCH reproduces clustering almost
exactly. If a statistic survives the GARCH null, it is telling you something
GARCH cannot.

All generators return an array of shape (n_sim, n) so statistics can be computed
in one pass per surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize


# --------------------------------------------------------------------------
# resampling nulls
# --------------------------------------------------------------------------

def iid_shuffle(r: np.ndarray, n_sim: int, rng: np.random.Generator) -> np.ndarray:
    """Random permutation. Preserves every marginal moment exactly; annihilates
    all time structure. The crudest null and the easiest to beat."""
    n = r.size
    out = np.empty((n_sim, n))
    for i in range(n_sim):
        out[i] = rng.permutation(r)
    return out


def circular_block(r: np.ndarray, n_sim: int, rng: np.random.Generator,
                   block_len: int = 20) -> np.ndarray:
    """Circular block bootstrap.

    Dependence shorter than `block_len` survives inside blocks; anything longer
    is broken at the joins. Choose block_len ~ 2x the horizon you claim, so that
    beating this null means your effect is genuinely longer-ranged than the
    autocorrelation you are sitting on.
    """
    n = r.size
    block_len = max(1, min(block_len, n))
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n, size=(n_sim, n_blocks))
    offs = np.arange(block_len)
    idx = (starts[:, :, None] + offs[None, None, :]) % n     # wrap around
    return r[idx.reshape(n_sim, -1)[:, :n]]


def stationary_bootstrap(r: np.ndarray, n_sim: int, rng: np.random.Generator,
                         mean_block: int = 20) -> np.ndarray:
    """Politis-Romano stationary bootstrap: geometric block lengths, p = 1/L.

    Unlike the fixed-block version the resampled series is genuinely stationary,
    which matters when the statistic is sensitive to the artificial regularity
    of equal-length blocks.
    """
    n = r.size
    p = 1.0 / max(1, mean_block)
    idx = np.empty((n_sim, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_sim)
    jump = rng.random((n_sim, n)) < p                 # restart at a new place
    fresh = rng.integers(0, n, size=(n_sim, n))
    for t in range(1, n):
        cont = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(jump[:, t], fresh[:, t], cont)
    return r[idx]


# --------------------------------------------------------------------------
# GARCH(1,1) with filtered historical simulation
# --------------------------------------------------------------------------

@dataclass
class GarchParams:
    mu: float
    omega: float
    alpha: float
    beta: float

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def uncond_var(self) -> float:
        p = self.persistence
        return self.omega / (1.0 - p) if p < 1.0 else float("nan")

    def half_life(self) -> float:
        """Days for a vol shock to decay halfway back to the long-run level."""
        p = self.persistence
        if not (0.0 < p < 1.0):
            return float("inf")
        return float(np.log(0.5) / np.log(p))


def _garch_recursion(e2: np.ndarray, omega: float, alpha: float,
                     beta: float, s2_init: float) -> np.ndarray:
    """sigma2_t = omega + alpha * e2_{t-1} + beta * sigma2_{t-1}."""
    n = e2.size
    s2 = np.empty(n)
    s2[0] = s2_init
    for t in range(1, n):
        s2[t] = omega + alpha * e2[t - 1] + beta * s2[t - 1]
    return s2


def fit_garch11(r: np.ndarray, variance_targeting: bool = True) -> GarchParams:
    """Gaussian QMLE fit of GARCH(1,1).

    Quasi-MLE: the Gaussian likelihood gives consistent parameter estimates even
    though daily returns are plainly not Gaussian, which is exactly why we then
    simulate with bootstrapped residuals instead of normal draws.

    Variance targeting pins omega to the sample variance and optimises only
    (alpha, beta). It costs a little efficiency and buys a lot of numerical
    stability, which matters when this is called inside an automated loop.
    """
    r = np.asarray(r, dtype=float)
    mu = float(r.mean())
    e = r - mu
    e2 = e ** 2
    s2_bar = float(e2.mean())
    if s2_bar <= 0:
        return GarchParams(mu, 0.0, 0.0, 0.0)

    def nll(theta):
        alpha, beta = theta
        if alpha < 0 or beta < 0 or alpha + beta >= 0.99999:
            return 1e12
        omega = s2_bar * (1.0 - alpha - beta) if variance_targeting else theta[2]
        if omega <= 0:
            return 1e12
        s2 = _garch_recursion(e2, omega, alpha, beta, s2_bar)
        if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
            return 1e12
        return 0.5 * float(np.sum(np.log(s2) + e2 / s2))

    best, best_val = None, np.inf
    # several starts: the surface is mildly multimodal near the unit root
    for a0, b0 in [(0.08, 0.90), (0.05, 0.93), (0.15, 0.80), (0.02, 0.97)]:
        res = optimize.minimize(
            nll, np.array([a0, b0]), method="L-BFGS-B",
            bounds=[(1e-8, 0.9999), (1e-8, 0.9999)],
        )
        if res.fun < best_val:
            best_val, best = res.fun, res.x

    alpha, beta = float(best[0]), float(best[1])
    omega = s2_bar * (1.0 - alpha - beta)
    return GarchParams(mu, omega, alpha, beta)


def standardised_residuals(r: np.ndarray, p: GarchParams) -> np.ndarray:
    """z_t = e_t / sigma_t under the fitted model.

    If the model is right these are iid with unit variance. Their empirical
    distribution carries whatever fat-tailedness GARCH did not explain, which is
    what makes the FHS null hard to beat.
    """
    e = r - p.mu
    s2 = _garch_recursion(e ** 2, p.omega, p.alpha, p.beta,
                          max(float(np.mean(e ** 2)), 1e-16))
    z = e / np.sqrt(s2)
    sd = np.std(z, ddof=1)
    return z / sd if sd > 0 else z


class GarchFHS:
    """GARCH(1,1) null with filtered historical simulation.

    Fit once, simulate many. Innovations are drawn with replacement from the
    fitted standardised residuals rather than from a normal, so the surrogate
    world reproduces the real series' fat tails as well as its clustering. A
    normal-innovation GARCH is a noticeably weaker null and flatters whatever
    you are testing.

    The time recursion cannot be vectorised, but the *paths* can: each step
    updates all n_sim paths at once, which keeps 1000+ surrogates cheap.
    """

    def __init__(self, r: np.ndarray):
        self.r = np.asarray(r, dtype=float)
        self.params = fit_garch11(self.r)
        self.z = standardised_residuals(self.r, self.params)

    def simulate(self, n_sim: int, rng: np.random.Generator,
                 burn: int = 500) -> np.ndarray:
        p = self.params
        n = self.r.size
        s2_0 = p.uncond_var
        if not np.isfinite(s2_0) or s2_0 <= 0:
            s2_0 = float(np.var(self.r, ddof=1))

        s2 = np.full(n_sim, s2_0)
        e_prev = np.zeros(n_sim)
        draws = rng.integers(0, self.z.size, size=(n_sim, burn + n))
        zz = self.z[draws]

        out = np.empty((n_sim, n))
        for t in range(burn + n):
            s2 = p.omega + p.alpha * e_prev ** 2 + p.beta * s2
            e = np.sqrt(s2) * zz[:, t]
            if t >= burn:
                out[:, t - burn] = e
            e_prev = e
        return out + p.mu


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def build_generators(r: np.ndarray, block_len: int = 20,
                     mean_block: int = 20) -> dict:
    """{name: callable(n_sim, rng) -> (n_sim, n) array}, ordered weakest first."""
    fhs = GarchFHS(r)
    return {
        "shuffled": lambda n, g: iid_shuffle(r, n, g),
        "block": lambda n, g: circular_block(r, n, g, block_len),
        "stationary": lambda n, g: stationary_bootstrap(r, n, g, mean_block),
        "garch": lambda n, g: fhs.simulate(n, g),
    }, fhs
