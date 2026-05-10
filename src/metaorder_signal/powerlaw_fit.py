"""Power-law fitting (Clauset et al. via `powerlaw`) and survival probabilities."""

from __future__ import annotations

import numpy as np

try:
    import powerlaw as pl
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install the `powerlaw` package (Clauset et al. implementation).") from exc


def fit_powerlaw_lengths(lengths: np.ndarray, discrete: bool = True, xmin: float | None = None):
    """
    Fit tail using `powerlaw`. Map to paper notation:

    Paper uses CCDF ~ L^{-alpha} for metaorder lengths (equation powerlaw / survival).

    If the fitted PDF tail is p(L) ~ L^{-alpha_pdf}, then CCDF tail exponent is alpha_pdf - 1.
    We report `alpha` matching the paper's CCDF exponent (survival / stylised facts).
    """
    x = np.asarray(lengths, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 30:
        return {"alpha": float("nan"), "alpha_pdf": float("nan"), "xmin": float("nan"), "fit": None}

    fit = pl.Fit(x, discrete=discrete, xmin=xmin)
    alpha_pdf = float(fit.power_law.alpha)
    xmin_f = float(fit.power_law.xmin)
    alpha_ccdf = alpha_pdf - 1.0
    return {"alpha": alpha_ccdf, "alpha_pdf": alpha_pdf, "xmin": xmin_f, "fit": fit}


def survival_continuous_approximation(n: int, alpha_ccdf: float) -> float:
    """Equation (survival): P(L > n | L >= n) ≈ 1 - alpha/n; clamp to [0, 1]."""
    if n <= 0 or not np.isfinite(alpha_ccdf):
        return float("nan")
    val = 1.0 - alpha_ccdf / float(n)
    return float(np.clip(val, 0.0, 1.0))


def expected_total_length_given_partial(n: int, alpha_ccdf: float, *, cap: float = 1e6) -> float:
    """
    Continuous Pareto-type heuristic for E[L | L >= n] when CCDF ~ L^{-alpha}:

    E[L | L >= n] ≈ n * alpha / (alpha - 1) for alpha > 1.
    """
    if n <= 0 or not np.isfinite(alpha_ccdf):
        return float("nan")
    if alpha_ccdf <= 1.05:
        return float(min(cap, max(n, 1.0) * 50.0))
    el = float(n) * alpha_ccdf / (alpha_ccdf - 1.0)
    return float(min(el, cap))
