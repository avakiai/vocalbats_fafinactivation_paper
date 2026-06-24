"""
inference.py — Bayesian inference of distribution parameters.

Two distributions are supported, selected via `dist=` in run_inference:

  Gamma (shape-scale):
      f(x | a, b) = x^(a-1) * exp(-x/b) / (b^a * Γ(a))
      log L = (a-1)*Σlog(x) - Σx/b - n*a*log(b) - n*log Γ(a)

  Gaussian (mean, stdev):
      f(x | μ, σ) = exp(-(x-μ)² / (2σ²)) / (σ √(2π))
      log L = -(n/2)*log(2π σ²) - Σ(x-μ)² / (2σ²)

All functions work in log-space internally for numerical stability;
scipy.special.gammaln replaces MATLAB's vpa() for arbitrary-precision gamma.
`ll_to_probability` is distribution-agnostic: it simply subtracts the max,
exponentiates, and normalises — valid for any log-likelihood grid.
"""

import numpy as np
from scipy.special import gammaln

# Default parameter grids (match MATLAB defaults)
DEFAULT_A = np.arange(0.1, 2.01, 0.01)
DEFAULT_B = np.arange(0.1, 2.01, 0.01)
DEFAULT_MU = np.arange(0.0, 10.01, 0.01)
DEFAULT_SIGMA = np.arange(0.01, 10.01, 0.01)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def gamma_log_likelihood(data, a, b):
    """
    Log-likelihood of *data* under Gamma(a, b).
    Scalars or broadcastable arrays for a and b are accepted.
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    return ((a - 1) * np.sum(np.log(data))
            - np.sum(data) / b
            - n * a * np.log(b)
            - n * gammaln(a))


def gaussian_log_likelihood(data, mu, sigma):
    """
    Log-likelihood of *data* under Gaussian(mu, sigma).
    Scalars or broadcastable arrays for mu and sigma are accepted.

    Implements the same math as logGaussianLikelihood.m:
        -(n/2)*log(2π σ²) - Σ(x-μ)² / (2σ²)
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    # Σ(x-μ)² depends only on μ; add a trailing axis to mu so data broadcasts
    # along it, then sum out that axis.
    sum_sq = np.sum((data - mu[..., np.newaxis]) ** 2, axis=-1)
    return -(n / 2) * np.log(2 * np.pi * sigma ** 2) - sum_sq / (2 * sigma ** 2)


def infer_gamma_params(data, a_values=None, b_values=None):
    """
    Compute the log-likelihood matrix over a grid of (a, b) values.

    Returns
    -------
    ll : ndarray, shape (n_a, n_b)
        ll[i, j] = log-likelihood at (a_values[i], b_values[j]).
    a_values, b_values : ndarray
        The parameter grids used.
    """
    if a_values is None:
        a_values = DEFAULT_A.copy()
    if b_values is None:
        b_values = DEFAULT_B.copy()

    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data) & (data > 0)]

    # Broadcasting: a → (n_a, 1), b → (1, n_b) → ll is (n_a, n_b)
    a = a_values[:, np.newaxis]
    b = b_values[np.newaxis, :]

    ll = gamma_log_likelihood(data, a, b)

    return ll, a_values, b_values


def infer_gaussian_params(data, mu_values=None, sigma_values=None):
    """
    Compute the log-likelihood matrix over a grid of (mu, sigma) values.

    Returns
    -------
    ll : ndarray, shape (n_mu, n_sigma)
        ll[i, j] = log-likelihood at (mu_values[i], sigma_values[j]).
    mu_values, sigma_values : ndarray
        The parameter grids used.
    """
    if mu_values is None:
        mu_values = DEFAULT_MU.copy()
    if sigma_values is None:
        sigma_values = DEFAULT_SIGMA.copy()

    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    # Broadcasting: mu → (n_mu, 1), sigma → (1, n_sigma) → ll is (n_mu, n_sigma)
    mu = mu_values[:, np.newaxis]
    sigma = sigma_values[np.newaxis, :]

    ll = gaussian_log_likelihood(data, mu, sigma)

    return ll, mu_values, sigma_values


def ll_to_probability(ll_matrix):
    """
    Convert a log-likelihood matrix to a normalised probability matrix.

    Subtracts the maximum before exponentiating (numerical stability trick
    from the original MATLAB implementation) then normalises.
    """
    ll_hat = ll_matrix - np.nanmax(ll_matrix)
    likelihood = np.exp(ll_hat)
    prob = likelihood / np.nansum(likelihood)
    return prob


# ---------------------------------------------------------------------------
# Marginalisation and credible regions
# ---------------------------------------------------------------------------

def marginal(prob_matrix, axis):
    """
    Marginalise a 2-D probability matrix by summing along *axis*.

    For prob_matrix of shape (n_a, n_b):
        axis=1  →  sum over b  →  P(a),  shape (n_a,)
        axis=0  →  sum over a  →  P(b),  shape (n_b,)
    """
    return prob_matrix.sum(axis=axis)


def marginal_mode(posterior, param_values):
    """
    Return the parameter value at the peak of a marginal posterior.

    Parameters
    ----------
    posterior : 1-D array
        Marginal probability (sums to ~1).
    param_values : 1-D array
        Parameter values corresponding to each probability entry.

    Returns
    -------
    float
        Parameter value at maximum posterior probability.
    """
    return float(param_values[np.argmax(posterior)])


def credible_region(posterior, param_values, credibility=0.95):
    """
    Highest-posterior-density (HPD) credible interval.

    Sorts probability mass in descending order, accumulates until the
    credibility level is reached, then returns the [min, max] of the
    included parameter values.

    Parameters
    ----------
    posterior : 1-D array
        Marginal probability (sums to ~1).
    param_values : 1-D array
        Parameter values corresponding to each probability entry.
    credibility : float
        Target probability content (default 0.95).

    Returns
    -------
    bounds : ndarray, shape (2,)
        [lower_bound, upper_bound] of the credible region.
    indices : ndarray, shape (2,)
        Indices of the bounds in param_values.
    """
    sort_idx = np.argsort(posterior)[::-1]
    cumsum = np.cumsum(posterior[sort_idx])
    n_include = np.searchsorted(cumsum, credibility)

    # Include at least one element
    n_include = max(n_include, 1)

    credible_params = param_values[sort_idx[:n_include]]
    lower = float(np.min(credible_params))
    upper = float(np.max(credible_params))

    lower_idx = int(np.argmin(np.abs(param_values - lower)))
    upper_idx = int(np.argmin(np.abs(param_values - upper)))

    return np.array([lower, upper]), np.array([lower_idx, upper_idx])


# ---------------------------------------------------------------------------
# Convenience wrapper: run full pipeline for one data slice
# ---------------------------------------------------------------------------

def run_inference(data, a_values=None, b_values=None,
                  credibility=0.95, dist='gamma'):
    """
    Full pipeline: data → LL → probability → marginals → credible regions.

    Parameters
    ----------
    data : array-like
        Observations. Non-finite values are removed; for dist='gamma',
        non-positive values are also removed.
    a_values, b_values : 1-D arrays, optional
        Parameter grids. For dist='gamma' these are (shape α, scale β);
        for dist='gaussian' they are (mean μ, stdev σ). Defaults come from
        DEFAULT_A/DEFAULT_B or DEFAULT_MU/DEFAULT_SIGMA respectively.
    credibility : float
        HPD credibility level (default 0.95).
    dist : {'gamma', 'gaussian'}
        Which distribution to fit.

    Returns
    -------
    dict with keys:
        ll, prob, prob_a, prob_b,
        mode_a, mode_b,
        bounds_a, bounds_b, idx_a, idx_b,
        a_values, b_values, n, dist
    Parameter-1 quantities (mode_a, prob_a, bounds_a, a_values) correspond to
    α for gamma and μ for gaussian; parameter-2 quantities correspond to β
    and σ respectively.
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    if dist == 'gamma':
        data = data[data > 0]
        ll, a_values, b_values = infer_gamma_params(data, a_values, b_values)
    elif dist == 'gaussian':
        ll, a_values, b_values = infer_gaussian_params(data, a_values, b_values)
    else:
        raise ValueError(f"Unknown dist: {dist!r} (expected 'gamma' or 'gaussian')")

    prob = ll_to_probability(ll)

    prob_a = marginal(prob, axis=1)   # sum over param-2 → P(param-1)
    prob_b = marginal(prob, axis=0)   # sum over param-1 → P(param-2)

    bounds_a, idx_a = credible_region(prob_a, a_values, credibility)
    bounds_b, idx_b = credible_region(prob_b, b_values, credibility)

    mode_a = marginal_mode(prob_a, a_values)
    mode_b = marginal_mode(prob_b, b_values)

    return dict(
        # likelihood and probability D|param1,param2
        ll=ll, prob=prob,
        # marginal probabilities
        prob_a=prob_a, prob_b=prob_b,
        # posterior modes
        mode_a=mode_a, mode_b=mode_b,
        # credible region data
        bounds_a=bounds_a, bounds_b=bounds_b,
        idx_a=idx_a, idx_b=idx_b,
        # inference search space
        a_values=a_values, b_values=b_values,
        n=len(data),
        
        dist=dist,
    )
