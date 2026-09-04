"""Separation quality metrics.

:func:`bss_eval_sources` is a NumPy port of the BSS_EVAL v3 criteria used in
the original experiments (the repository shipped ``bss_eval_sources.m``).
:func:`si_sdr` is the cheaper scale-invariant SDR that most recent separation
work reports.
"""

from __future__ import annotations

import itertools

import numpy as np

__all__ = ["bss_eval_sources", "si_sdr", "spectral_flatness", "BSSEvalScores"]

_EPS = 1e-20


def _toeplitz(c, r):
    """Toeplitz matrix with first column ``c`` and first row ``r``."""
    c = np.asarray(c)
    r = np.asarray(r)
    values = np.concatenate((c[::-1], r[1:]))
    idx = np.arange(len(c) - 1, -1, -1)[:, None] + np.arange(len(r))[None, :]
    return values[idx]


def _fftconvolve(a, b):
    n = len(a) + len(b) - 1
    n_fft = 1 << int(np.ceil(np.log2(max(n, 1))))
    out = np.fft.irfft(np.fft.rfft(a, n_fft) * np.fft.rfft(b, n_fft), n_fft)
    return out[:n]


def _project(references, estimate, flen):
    """Least-squares projection of ``estimate`` onto filtered references.

    Returns the best approximation of ``estimate`` by a sum of the reference
    signals each passed through an FIR filter of ``flen`` taps.
    """
    n_src, n_sampl = references.shape
    n_fft = 1 << int(np.ceil(np.log2(n_sampl + flen - 1)))

    ref_f = np.fft.rfft(references, n=n_fft, axis=1)
    est_f = np.fft.rfft(estimate, n=n_fft)

    # Block-Toeplitz Gram matrix of the delayed references.
    G = np.zeros((n_src * flen, n_src * flen))
    for i in range(n_src):
        for j in range(i, n_src):
            corr = np.real(np.fft.irfft(ref_f[i] * np.conj(ref_f[j]), n=n_fft))
            block = _toeplitz(
                np.hstack((corr[0], corr[-1 : -flen : -1])), corr[:flen]
            )
            G[i * flen : (i + 1) * flen, j * flen : (j + 1) * flen] = block
            if i != j:
                G[j * flen : (j + 1) * flen, i * flen : (i + 1) * flen] = block.T

    # Cross-correlation between references and the estimate.
    D = np.zeros(n_src * flen)
    for i in range(n_src):
        corr = np.real(np.fft.irfft(ref_f[i] * np.conj(est_f), n=n_fft))
        D[i * flen : (i + 1) * flen] = np.hstack((corr[0], corr[-1 : -flen : -1]))

    try:
        coeffs = np.linalg.solve(G, D)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(G, D, rcond=None)[0]
    coeffs = coeffs.reshape(n_src, flen)

    projection = np.zeros(n_sampl + flen - 1)
    for i in range(n_src):
        projection += _fftconvolve(coeffs[i], references[i])[: n_sampl + flen - 1]
    return projection


def _decompose(references, estimate, index, flen):
    n_sampl = estimate.size
    true = np.hstack((references[index], np.zeros(flen - 1)))
    spatial = _project(references[index : index + 1], estimate, flen) - true
    interference = _project(references, estimate, flen) - true - spatial
    artifact = -true - spatial - interference
    artifact[:n_sampl] += estimate
    return true, spatial, interference, artifact


def _criteria(true, spatial, interference, artifact):
    filtered = true + spatial
    energy = float(np.sum(filtered**2))
    sdr = 10 * np.log10(energy / (np.sum((interference + artifact) ** 2) + _EPS) + _EPS)
    sir = 10 * np.log10(energy / (np.sum(interference**2) + _EPS) + _EPS)
    sar = 10 * np.log10(
        np.sum((filtered + interference) ** 2) / (np.sum(artifact**2) + _EPS) + _EPS
    )
    return sdr, sir, sar


class BSSEvalScores(tuple):
    """``(sdr, sir, sar, permutation)`` with attribute access."""

    __slots__ = ()

    def __new__(cls, sdr, sir, sar, permutation):
        return super().__new__(cls, (sdr, sir, sar, permutation))

    sdr = property(lambda self: self[0])
    sir = property(lambda self: self[1])
    sar = property(lambda self: self[2])
    permutation = property(lambda self: self[3])

    def __repr__(self):
        return (
            f"BSSEvalScores(sdr={np.round(self.sdr, 2)}, "
            f"sir={np.round(self.sir, 2)}, sar={np.round(self.sar, 2)}, "
            f"permutation={list(self.permutation)})"
        )


def bss_eval_sources(references, estimates, filter_length=512, compute_permutation=True):
    """BSS_EVAL SDR / SIR / SAR for a set of estimated sources.

    Parameters
    ----------
    references, estimates : array, shape (n_sources, n_samples)
        Reference and estimated sources.  1-D input is treated as one source.
    filter_length : int
        Length of the allowed distortion filter, in samples (512 in the
        reference implementation).
    compute_permutation : bool
        Search over source orderings and report the best one.  Turn it off
        when the estimates are already aligned with the references.

    Returns
    -------
    BSSEvalScores
        Per-source ``sdr``, ``sir``, ``sar`` arrays plus the ``permutation``
        that maps estimates onto references.
    """
    references = np.atleast_2d(np.asarray(references, dtype=np.float64))
    estimates = np.atleast_2d(np.asarray(estimates, dtype=np.float64))
    if references.shape != estimates.shape:
        raise ValueError(
            f"references {references.shape} and estimates {estimates.shape} "
            "must have the same shape"
        )
    n_src = references.shape[0]

    scores = np.empty((n_src, n_src, 3))
    for j, estimate in enumerate(estimates):
        for i in range(n_src):
            scores[j, i] = _criteria(*_decompose(references, estimate, i, filter_length))

    if not compute_permutation or n_src == 1:
        order = np.arange(n_src)
    else:
        candidates = list(itertools.permutations(range(n_src)))
        totals = [sum(scores[j, perm[j], 0] for j in range(n_src)) for perm in candidates]
        order = np.array(candidates[int(np.argmax(totals))])

    picked = np.array([scores[j, order[j]] for j in range(n_src)])
    return BSSEvalScores(picked[:, 0], picked[:, 1], picked[:, 2], order)


def si_sdr(reference, estimate):
    """Scale-invariant signal-to-distortion ratio, in dB."""
    reference = np.asarray(reference, dtype=np.float64).ravel()
    estimate = np.asarray(estimate, dtype=np.float64).ravel()
    n = min(len(reference), len(estimate))
    reference, estimate = reference[:n], estimate[:n]

    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + _EPS)
    target = alpha * reference
    noise = estimate - target
    return float(
        10 * np.log10((np.sum(target**2) + _EPS) / (np.sum(noise**2) + _EPS))
    )


def spectral_flatness(V, eps=1e-12):
    """Energy-weighted mean spectral flatness of a magnitude spectrogram.

    The per-frame ratio of geometric to arithmetic mean across frequency,
    averaged over frames with weights proportional to frame energy.  Close to
    0 for a sparse, tonal spectrum; close to 1 for a flat, noise-like one.
    """
    V = np.maximum(np.asarray(V, dtype=np.float64), eps)
    geometric = np.exp(np.mean(np.log(V), axis=0))
    arithmetic = np.mean(V, axis=0)
    weights = arithmetic / (arithmetic.sum() + eps)
    return float(np.sum(weights * geometric / arithmetic))
