r"""Structured Projective NMF (SPNMF) and the plain beta-NMF it builds on.

SPNMF factorises a non-negative spectrogram ``V`` (F x T) as::

    V ~= W_h @ W_h.T @ V  +  W_p @ H_p
         \-----v------/     \----v----/
          harmonic part      percussive part

The left term is a *projective* NMF: a rank-``r`` non-negative projection of
the data onto itself.  Its columns come out near-orthogonal and spectrally
sparse, which is exactly the shape of a tonal, temporally stable source.  The
right term is an ordinary NMF, free to model broadband transients.

Both parts are estimated with multiplicative updates obtained by splitting the
gradient of the beta-divergence into its non-negative parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .divergence import EPS, as_beta, beta_divergence, gradient_split

__all__ = ["SPNMFResult", "NMFResult", "spnmf", "nmf"]


# --------------------------------------------------------------------------
# gradient helpers
#
# Every update below is `factor *= negative_part / positive_part`, where the
# two parts come from `gradient_split`.  When beta == 1 the positive part is a
# matrix of ones; `gradient_split` signals that with None so the products can
# be reduced to a broadcast sum instead of an F x T temporary.
# --------------------------------------------------------------------------


def _right_product(part, Ht, ones_shape):
    """``part @ H.T``, with the all-ones shortcut."""
    if part is None:
        return np.broadcast_to(Ht.sum(axis=0), ones_shape)
    return part @ Ht


def _left_product(Wt, part, ones_shape):
    """``W.T @ part``, with the all-ones shortcut."""
    if part is None:
        return np.broadcast_to(Wt.sum(axis=1)[:, None], ones_shape)
    return Wt @ part


def _projective_terms(V, W, part, VtW):
    """``part @ V.T @ W + V @ part.T @ W``, with the all-ones shortcut.

    This is the gradient of ``tr(part.T @ W @ W.T @ V)`` with respect to ``W``,
    and it is where the projective part differs from ordinary NMF: ``W`` shows
    up twice in the model, so both contributions have to be summed.
    """
    if part is None:
        rows = np.broadcast_to(VtW.sum(axis=0), (V.shape[0], VtW.shape[1]))
        cols = V.sum(axis=1)[:, None] * W.sum(axis=0)[None, :]
        return rows + cols
    return part @ VtW + V @ (part.T @ W)


def _normalise_columns(W, H):
    """Scale ``W`` columns to unit L2 norm, compensating in ``H``.

    ``W @ H`` is unchanged, so this never perturbs the cost; it only keeps the
    two factors from drifting to very different magnitudes.
    """
    scale = np.sqrt(np.sum(W**2, axis=0)) + EPS
    return W / scale, H * scale[:, None]


def _rescale_to(factor_a, factor_b, target_mean, quadratic=False):
    """Rescale a freshly drawn factor so the model starts near the data scale."""
    current = float(np.mean(factor_a @ factor_b))
    if current <= 0.0:
        return 1.0
    ratio = target_mean / current
    return float(np.sqrt(ratio)) if quadratic else float(ratio)


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass
class SPNMFResult:
    """Factors and diagnostics returned by :func:`spnmf`."""

    W_h: np.ndarray
    """Harmonic (projective) basis, ``F x n_harmonic``."""

    W_p: np.ndarray
    """Percussive dictionary, ``F x n_percussive``."""

    H_p: np.ndarray
    """Percussive activations, ``n_percussive x T``."""

    cost: np.ndarray
    """Divergence after each iteration."""

    beta: float
    n_iter: int
    converged: bool = False
    _V: np.ndarray = field(default=None, repr=False)

    @property
    def harmonic(self):
        """Harmonic spectrogram ``W_h @ W_h.T @ V``."""
        return self.W_h @ (self.W_h.T @ self._V)

    @property
    def percussive(self):
        """Percussive spectrogram ``W_p @ H_p``."""
        return self.W_p @ self.H_p

    @property
    def reconstruction(self):
        return self.harmonic + self.percussive


@dataclass
class NMFResult:
    """Factors and diagnostics returned by :func:`nmf`."""

    W: np.ndarray
    H: np.ndarray
    cost: np.ndarray
    beta: float
    n_iter: int
    converged: bool = False

    @property
    def reconstruction(self):
        return self.W @ self.H


# --------------------------------------------------------------------------
# SPNMF
# --------------------------------------------------------------------------


def spnmf(
    V,
    n_harmonic=20,
    n_percussive=20,
    divergence="kl",
    W_p=None,
    update_dictionary=None,
    n_iter=200,
    tol=1e-5,
    w_exponent=0.5,
    random_state=None,
    verbose=False,
):
    """Factorise ``V`` as ``W_h @ W_h.T @ V + W_p @ H_p``.

    Parameters
    ----------
    V : array, shape (F, T)
        Non-negative input, typically a magnitude or power spectrogram.
    n_harmonic : int
        Rank of the projective (harmonic) part.
    n_percussive : int
        Number of percussive components.  Ignored when ``W_p`` is given.
    divergence : {'kl', 'is', 'euclidean'} or float
        Cost function; a float is taken as a raw beta exponent.
    W_p : array, shape (F, n_percussive), optional
        Pre-trained percussive dictionary.  Supplying one switches the solver
        to the semi-supervised algorithm of the paper, which is what actually
        makes harmonic/percussive separation work -- see
        :func:`spnmf.dictionary.learn_dictionary`.
    update_dictionary : bool, optional
        Whether ``W_p`` is re-estimated.  Defaults to ``False`` when ``W_p``
        was supplied and ``True`` otherwise.
    n_iter : int
        Maximum number of iterations.
    tol : float
        Stop once the relative cost decrease over one iteration drops below
        this value.  Pass ``None`` to always run ``n_iter`` iterations.
    w_exponent : float
        Exponent applied to the ``W_h`` update ratio.  ``0.5`` damps the step
        and is what the original KL implementation used; ``1.0`` is the raw
        multiplicative update.
    random_state : int or numpy.random.Generator, optional
        Seed for the initial factors.
    verbose : bool
        Print the cost every 50 iterations.

    Returns
    -------
    SPNMFResult
    """
    V = np.asarray(V, dtype=np.float64)
    if V.ndim != 2:
        raise ValueError(f"V must be 2-D, got shape {V.shape}")
    if np.any(V < 0):
        raise ValueError("V must be non-negative")
    if n_harmonic < 1:
        raise ValueError("n_harmonic must be >= 1")

    beta = as_beta(divergence)
    rng = np.random.default_rng(random_state)
    F, T = V.shape
    target = max(float(np.mean(V)), EPS)

    # ---- initialisation ---------------------------------------------------
    W_h = rng.random((F, n_harmonic)) + 0.1
    W_h /= np.sqrt(np.sum(W_h**2, axis=0)) + EPS
    # W_h @ W_h.T @ V is quadratic in W_h, hence the square root.
    W_h *= _rescale_to(W_h, W_h.T @ V, 0.5 * target, quadratic=True)

    if W_p is None:
        if n_percussive < 1:
            raise ValueError("n_percussive must be >= 1")
        W_p = rng.random((F, n_percussive)) + 0.1
        W_p /= np.sqrt(np.sum(W_p**2, axis=0)) + EPS
        learn_dict = True if update_dictionary is None else bool(update_dictionary)
    else:
        W_p = np.array(W_p, dtype=np.float64, copy=True)
        if W_p.shape[0] != F:
            raise ValueError(
                f"W_p has {W_p.shape[0]} frequency bins but V has {F}"
            )
        if np.any(W_p < 0):
            raise ValueError("W_p must be non-negative")
        learn_dict = False if update_dictionary is None else bool(update_dictionary)
    n_percussive = W_p.shape[1]

    H_p = rng.random((n_percussive, T)) + 0.1
    H_p *= _rescale_to(W_p, H_p, 0.5 * target)

    # ---- main loop --------------------------------------------------------
    cost = np.empty(n_iter)
    converged = False
    previous = np.inf
    it = 0

    for it in range(n_iter):
        # --- percussive dictionary
        if learn_dict:
            V_hat = W_h @ (W_h.T @ V) + W_p @ H_p
            minus, plus = gradient_split(V, V_hat, beta)
            Ht = H_p.T
            num = minus @ Ht
            den = _right_product(plus, Ht, (F, n_percussive))
            W_p = W_p * (num / (den + EPS))

        # --- percussive activations
        V_hat = W_h @ (W_h.T @ V) + W_p @ H_p
        minus, plus = gradient_split(V, V_hat, beta)
        Wt = W_p.T
        num = Wt @ minus
        den = _left_product(Wt, plus, (n_percussive, T))
        H_p = H_p * (num / (den + EPS))

        if learn_dict:
            W_p, H_p = _normalise_columns(W_p, H_p)

        # --- harmonic projective basis
        V_hat = W_h @ (W_h.T @ V) + W_p @ H_p
        minus, plus = gradient_split(V, V_hat, beta)
        VtW = V.T @ W_h
        num = _projective_terms(V, W_h, minus, VtW)
        den = _projective_terms(V, W_h, plus, VtW)
        ratio = num / (den + EPS)
        W_h = W_h * (ratio**w_exponent if w_exponent != 1.0 else ratio)

        # --- diagnostics
        V_hat = W_h @ (W_h.T @ V) + W_p @ H_p
        cost[it] = beta_divergence(V, V_hat, beta)

        if verbose and (it + 1) % 50 == 0:
            print(f"  iter {it + 1:4d}  cost {cost[it]:.6g}")

        if tol is not None and np.isfinite(previous):
            improvement = (previous - cost[it]) / max(abs(previous), EPS)
            if 0.0 <= improvement < tol:
                converged = True
                cost = cost[: it + 1]
                break
        previous = cost[it]

    return SPNMFResult(
        W_h=W_h,
        W_p=W_p,
        H_p=H_p,
        cost=cost[: it + 1],
        beta=beta,
        n_iter=it + 1,
        converged=converged,
        _V=V,
    )


# --------------------------------------------------------------------------
# plain NMF (used to train percussive dictionaries)
# --------------------------------------------------------------------------


def nmf(
    V,
    n_components=20,
    divergence="kl",
    W=None,
    update_W=True,
    n_iter=200,
    tol=1e-5,
    random_state=None,
    verbose=False,
):
    """Plain beta-NMF, ``V ~= W @ H``, with multiplicative updates."""
    V = np.asarray(V, dtype=np.float64)
    if V.ndim != 2:
        raise ValueError(f"V must be 2-D, got shape {V.shape}")
    if np.any(V < 0):
        raise ValueError("V must be non-negative")

    beta = as_beta(divergence)
    rng = np.random.default_rng(random_state)
    F, T = V.shape
    target = max(float(np.mean(V)), EPS)

    if W is None:
        W = rng.random((F, n_components)) + 0.1
        W /= np.sqrt(np.sum(W**2, axis=0)) + EPS
    else:
        W = np.array(W, dtype=np.float64, copy=True)
        if W.shape[0] != F:
            raise ValueError(f"W has {W.shape[0]} frequency bins but V has {F}")
    n_components = W.shape[1]

    H = rng.random((n_components, T)) + 0.1
    H *= _rescale_to(W, H, target)

    cost = np.empty(n_iter)
    converged = False
    previous = np.inf
    it = 0

    for it in range(n_iter):
        if update_W:
            V_hat = W @ H
            minus, plus = gradient_split(V, V_hat, beta)
            Ht = H.T
            W = W * ((minus @ Ht) / (_right_product(plus, Ht, (F, n_components)) + EPS))

        V_hat = W @ H
        minus, plus = gradient_split(V, V_hat, beta)
        Wt = W.T
        H = H * ((Wt @ minus) / (_left_product(Wt, plus, (n_components, T)) + EPS))

        if update_W:
            W, H = _normalise_columns(W, H)

        cost[it] = beta_divergence(V, W @ H, beta)
        if verbose and (it + 1) % 50 == 0:
            print(f"  iter {it + 1:4d}  cost {cost[it]:.6g}")

        if tol is not None and np.isfinite(previous):
            improvement = (previous - cost[it]) / max(abs(previous), EPS)
            if 0.0 <= improvement < tol:
                converged = True
                break
        previous = cost[it]

    return NMFResult(
        W=W, H=H, cost=cost[: it + 1], beta=beta, n_iter=it + 1, converged=converged
    )
