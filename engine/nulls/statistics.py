"""
Statistics for the null-comparison engine.

Every function here takes a 1-D array of returns and returns a float. They are
deliberately dependency-light and implemented from scratch so that the same code
path runs on the real series and on every surrogate, with no library-version
differences between the two worlds.

The set mirrors the `THREE WORLDS, ONE PIPELINE` panel: annualised vol, excess
kurtosis, ACF of |r| at several lags, signed ACF at lag 1, Ljung-Box Q on |r|,
vol-band persistence, lift over independence, and the vol-persistence R-squared.
"""

from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# basic moments
# --------------------------------------------------------------------------

def ann_vol(r: np.ndarray, periods: int = TRADING_DAYS) -> float:
    """Annualised standard deviation."""
    return float(np.std(r, ddof=1) * np.sqrt(periods))


def excess_kurtosis(r: np.ndarray) -> float:
    """Fisher excess kurtosis (0 for a Gaussian)."""
    d = r - r.mean()
    m2 = np.mean(d ** 2)
    m4 = np.mean(d ** 4)
    if m2 <= 0:
        return 0.0
    return float(m4 / (m2 ** 2) - 3.0)


def skewness(r: np.ndarray) -> float:
    d = r - r.mean()
    m2 = np.mean(d ** 2)
    m3 = np.mean(d ** 3)
    if m2 <= 0:
        return 0.0
    return float(m3 / (m2 ** 1.5))


def sharpe(r: np.ndarray, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe, zero risk-free rate."""
    sd = np.std(r, ddof=1)
    if sd <= 0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods))


# --------------------------------------------------------------------------
# autocorrelation
# --------------------------------------------------------------------------

def acf(x: np.ndarray, lag: int) -> float:
    """Sample autocorrelation at a single lag (biased/`1/n` estimator, as in
    Ljung-Box convention)."""
    if lag <= 0:
        return 1.0
    n = x.size
    if lag >= n:
        return 0.0
    d = x - x.mean()
    denom = np.dot(d, d)
    if denom <= 0:
        return 0.0
    return float(np.dot(d[lag:], d[:-lag]) / denom)


def acf_abs(r: np.ndarray, lag: int) -> float:
    """ACF of |r| — the standard volatility-clustering diagnostic."""
    return acf(np.abs(r), lag)


def acf_signed(r: np.ndarray, lag: int) -> float:
    """ACF of raw returns — should be near zero in any liquid market."""
    return acf(r, lag)


def ljung_box(x: np.ndarray, m: int = 22) -> float:
    """Ljung-Box Q statistic over lags 1..m.

    Returned as the raw Q rather than a p-value: the null distribution comes
    from the surrogate series, not from a chi-squared table, because the
    asymptotic distribution is wrong for |r| on real data anyway.
    """
    n = x.size
    q = 0.0
    for k in range(1, m + 1):
        rk = acf(x, k)
        q += (rk ** 2) / (n - k)
    return float(n * (n + 2) * q)


def ljung_box_abs(r: np.ndarray, m: int = 22) -> float:
    return ljung_box(np.abs(r), m)


# --------------------------------------------------------------------------
# volatility bands  (the reel's REGIME TAPE / same-band-rate machinery)
# --------------------------------------------------------------------------

def realised_vol(r: np.ndarray, window: int = 20,
                 periods: int = TRADING_DAYS) -> np.ndarray:
    """Trailing realised volatility, annualised.

    Index t uses returns (t-window+1 .. t) inclusive, so the value at t is
    knowable at the close of t. Entries before `window-1` are NaN.
    """
    n = r.size
    out = np.full(n, np.nan)
    if n < window:
        return out
    # rolling std via cumulative sums (population-consistent, ddof=1)
    c1 = np.concatenate(([0.0], np.cumsum(r)))
    c2 = np.concatenate(([0.0], np.cumsum(r ** 2)))
    idx = np.arange(window - 1, n)
    s1 = c1[idx + 1] - c1[idx + 1 - window]
    s2 = c2[idx + 1] - c2[idx + 1 - window]
    var = (s2 - s1 ** 2 / window) / (window - 1)
    var = np.maximum(var, 0.0)
    out[idx] = np.sqrt(var) * np.sqrt(periods)
    return out


def vol_bands(r: np.ndarray, window: int = 20, k: int = 5,
              periods: int = TRADING_DAYS) -> np.ndarray:
    """Label each day with its volatility band, 0..k-1, by within-series
    quantile.

    Banding is done per-series on purpose: each world (real, shuffled, GARCH)
    is banded by its own vol distribution. That makes the comparison about
    *persistence of state*, not about level differences between worlds — a
    surrogate with lower average vol should not be advantaged.
    """
    rv = realised_vol(r, window, periods)
    out = np.full(r.size, -1, dtype=np.int64)
    valid = ~np.isnan(rv)
    v = rv[valid]
    if v.size < k:
        return out
    # equal-count bins; interior cut points only
    cuts = np.quantile(v, np.linspace(0, 1, k + 1)[1:-1])
    out[valid] = np.searchsorted(cuts, v, side="right")
    return out


def _band_pairs(bands: np.ndarray, horizon: int):
    """Aligned (band_t, band_{t+h}) pairs where both are labelled."""
    if horizon >= bands.size:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    a = bands[:-horizon]
    b = bands[horizon:]
    ok = (a >= 0) & (b >= 0)
    return a[ok], b[ok]


def same_band_rate(r: np.ndarray, window: int = 20, k: int = 5,
                   horizon: int = 5, periods: int = TRADING_DAYS) -> float:
    """P(vol band in `horizon` days == vol band today)."""
    a, b = _band_pairs(vol_bands(r, window, k, periods), horizon)
    if a.size == 0:
        return float("nan")
    return float(np.mean(a == b))


def independence_baseline(r: np.ndarray, window: int = 20, k: int = 5,
                          horizon: int = 5,
                          periods: int = TRADING_DAYS) -> float:
    """What the same-band rate would be if today's band told you nothing.

    Computed from the realised marginals as sum(p_i * q_i) rather than assumed
    to be 1/k, so it stays correct if the bins are not perfectly equal-count
    after the NaN burn-in is dropped.
    """
    a, b = _band_pairs(vol_bands(r, window, k, periods), horizon)
    if a.size == 0:
        return float("nan")
    pa = np.bincount(a, minlength=k) / a.size
    pb = np.bincount(b, minlength=k) / b.size
    return float(np.dot(pa, pb))


def lift_over_independence(r: np.ndarray, window: int = 20, k: int = 5,
                           horizon: int = 5,
                           periods: int = TRADING_DAYS) -> float:
    """Same-band rate minus its independence baseline, in percentage points."""
    sb = same_band_rate(r, window, k, horizon, periods)
    ib = independence_baseline(r, window, k, horizon, periods)
    return float((sb - ib) * 100.0)


def cramers_v(r: np.ndarray, window: int = 20, k: int = 5, horizon: int = 5,
              periods: int = TRADING_DAYS) -> float:
    """Cramer's V on the band-transition contingency table.

    A scale-free measure of how much today's band tells you about the band in
    `horizon` days. 0 = nothing, 1 = deterministic.
    """
    a, b = _band_pairs(vol_bands(r, window, k, periods), horizon)
    if a.size == 0:
        return float("nan")
    obs = np.zeros((k, k))
    np.add.at(obs, (a, b), 1.0)
    n = obs.sum()
    row = obs.sum(1, keepdims=True)
    col = obs.sum(0, keepdims=True)
    exp = row @ col / n
    mask = exp > 0
    chi2 = float(np.sum((obs[mask] - exp[mask]) ** 2 / exp[mask]))
    denom = n * (k - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(chi2 / denom))


def vol_persistence_r2(r: np.ndarray, window: int = 20, horizon: int = 5,
                       periods: int = TRADING_DAYS) -> float:
    """R-squared of regressing log future realised vol on log current.

    Uses log vol because vol is right-skewed and roughly log-normal; the linear
    version is dominated by a handful of crisis days.
    """
    rv = realised_vol(r, window, periods)
    if horizon >= rv.size:
        return float("nan")
    x, y = rv[:-horizon], rv[horizon:]
    ok = ~np.isnan(x) & ~np.isnan(y) & (x > 0) & (y > 0)
    if ok.sum() < 30:
        return float("nan")
    lx, ly = np.log(x[ok]), np.log(y[ok])
    c = np.corrcoef(lx, ly)[0, 1]
    return float(c ** 2)


# --------------------------------------------------------------------------
# the registry consumed by compare.py
# --------------------------------------------------------------------------

def default_statistics(window: int = 20, k: int = 5, horizon: int = 5,
                       lb_lags: int = 22) -> dict:
    """The reel's statistic set, as {display_name: callable(r) -> float}."""
    return {
        "Ann. volatility":       lambda r: ann_vol(r),
        "Excess kurtosis":       lambda r: excess_kurtosis(r),
        "ACF |r| lag 1":         lambda r: acf_abs(r, 1),
        "ACF |r| lag 22":        lambda r: acf_abs(r, 22),
        "ACF |r| lag 66":        lambda r: acf_abs(r, 66),
        "ACF r lag 1":           lambda r: acf_signed(r, 1),
        f"Ljung-Box Q({lb_lags}) on |r|": lambda r: ljung_box_abs(r, lb_lags),
        "Same-band rate":        lambda r: same_band_rate(r, window, k, horizon),
        "Lift over independence": lambda r: lift_over_independence(r, window, k, horizon),
        "Cramer's V":            lambda r: cramers_v(r, window, k, horizon),
        "Regression R2":         lambda r: vol_persistence_r2(r, window, horizon),
    }
