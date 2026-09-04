"""Beta-divergence and the gradient split used by the multiplicative updates.

The beta-divergence unifies the three cost functions studied in the original
SPNMF paper: ``beta=2`` is the squared Euclidean distance, ``beta=1`` the
(generalised) Kullback-Leibler divergence and ``beta=0`` the Itakura-Saito
divergence.
"""

from __future__ import annotations

import numpy as np

#: Names accepted wherever this package takes a ``divergence`` argument.
DIVERGENCES = {
    "euclidean": 2.0,
    "euc": 2.0,
    "kl": 1.0,
    "is": 0.0,
    "itakura-saito": 0.0,
}

EPS = np.finfo(np.float64).eps


def as_beta(divergence):
    """Resolve a divergence name (or a raw float) to its beta exponent."""
    if isinstance(divergence, (int, float, np.floating)):
        return float(divergence)
    try:
        return DIVERGENCES[str(divergence).lower()]
    except KeyError:
        raise ValueError(
            f"unknown divergence {divergence!r}; "
            f"expected one of {sorted(set(DIVERGENCES))} or a float beta"
        ) from None


def beta_divergence(V, V_hat, beta=1.0):
    """Return ``D_beta(V | V_hat)`` summed over every entry.

    Both arguments are clipped away from zero, so the result is finite even
    for the KL and IS cases where the divergence is otherwise singular.
    """
    V = np.asarray(V, dtype=np.float64)
    V_hat = np.maximum(np.asarray(V_hat, dtype=np.float64), EPS)
    beta = float(beta)

    if beta == 2.0:
        return 0.5 * float(np.sum((V - V_hat) ** 2))
    if beta == 1.0:
        Vp = np.maximum(V, EPS)
        return float(np.sum(Vp * np.log(Vp / V_hat) - V + V_hat))
    if beta == 0.0:
        Vp = np.maximum(V, EPS)
        ratio = Vp / V_hat
        return float(np.sum(ratio - np.log(ratio) - 1.0))
    return float(
        np.sum(
            (V**beta + (beta - 1.0) * V_hat**beta - beta * V * V_hat ** (beta - 1.0))
            / (beta * (beta - 1.0))
        )
    )


def gradient_split(V, V_hat, beta):
    """Split ``dD/dV_hat`` into its non-negative ``(minus, plus)`` parts.

    The derivative of the beta-divergence with respect to the reconstruction
    is ``V_hat**(beta-1) - V * V_hat**(beta-2)``.  Multiplicative updates need
    the two terms separately: ``minus`` carries the data, ``plus`` the model.

    For ``beta=1`` the ``plus`` term is an all-ones matrix.  It is returned as
    ``None`` so callers can exploit that structure instead of materialising it;
    :func:`plus_as_array` expands it when a dense array really is needed.
    """
    V_hat = np.maximum(V_hat, EPS)
    if beta == 2.0:
        return V, V_hat
    if beta == 1.0:
        return V / V_hat, None
    if beta == 0.0:
        inv = 1.0 / V_hat
        return V * inv * inv, inv
    return V * V_hat ** (beta - 2.0), V_hat ** (beta - 1.0)


def plus_as_array(plus, shape):
    """Densify the ``plus`` part returned by :func:`gradient_split`."""
    if plus is None:
        return np.ones(shape)
    return plus
