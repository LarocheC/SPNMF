"""Percussive dictionaries for the semi-supervised SPNMF.

A fully unsupervised SPNMF does not reliably assign the tonal source to the
projective part -- the paper says as much, and the 2016 experiments all use a
fixed drum dictionary learned from ENST-Drums.  These helpers build such a
dictionary from whatever percussive audio is at hand.
"""

from __future__ import annotations

import numpy as np

from .core import nmf
from .divergence import EPS
from .stft import stft

__all__ = ["learn_dictionary", "stft_dictionary", "concatenate_dictionaries"]


def _spectrogram(x, n_fft, hop_length, window, power):
    return np.abs(stft(np.asarray(x, dtype=np.float64), n_fft, hop_length, window)) ** power


def learn_dictionary(
    signals,
    n_components=20,
    n_fft=2048,
    hop_length=512,
    window="sqrt_hann",
    power=1.0,
    divergence="kl",
    n_iter=300,
    random_state=None,
    verbose=False,
):
    """Train a dictionary by running plain NMF over percussive training audio.

    Parameters
    ----------
    signals : array or sequence of arrays
        One or more mono training signals (isolated drum hits, a drum stem,
        a drum loop...).  Their spectrograms are concatenated along time.
    n_components : int
        Dictionary size.  This is the rank of the NMF, so it stays small and
        cheap regardless of how much training audio is supplied.

    Returns
    -------
    ndarray, shape (n_fft // 2 + 1, n_components)
        Column-normalised dictionary, ready to pass as ``W_p``.
    """
    if isinstance(signals, np.ndarray) and signals.ndim == 1:
        signals = [signals]

    spectrograms = [
        _spectrogram(s, n_fft, hop_length, window, power) for s in signals
    ]
    if not spectrograms:
        raise ValueError("no training signals supplied")
    V = np.concatenate(spectrograms, axis=1)

    result = nmf(
        V,
        n_components=n_components,
        divergence=divergence,
        n_iter=n_iter,
        random_state=random_state,
        verbose=verbose,
    )
    W = result.W
    return W / (np.sqrt(np.sum(W**2, axis=0)) + EPS)


def stft_dictionary(
    signals,
    n_fft=2048,
    hop_length=512,
    window="sqrt_hann",
    power=1.0,
    max_columns=None,
    energy_threshold=1e-4,
    random_state=None,
):
    """Build a dictionary directly from training spectrogram frames.

    This is the second construction discussed in the paper: no NMF, just the
    STFT frames themselves.  It is far more redundant than
    :func:`learn_dictionary` (and slower to use), but every column stays
    interpretable as one moment of one drum.

    Near-silent frames are dropped, and ``max_columns`` randomly subsamples
    what is left.
    """
    if isinstance(signals, np.ndarray) and signals.ndim == 1:
        signals = [signals]

    V = np.concatenate(
        [_spectrogram(s, n_fft, hop_length, window, power) for s in signals], axis=1
    )
    energy = np.sqrt(np.sum(V**2, axis=0))
    keep = energy > energy_threshold * (energy.max() + EPS)
    V = V[:, keep]
    if V.shape[1] == 0:
        raise ValueError("all training frames were below the energy threshold")

    if max_columns is not None and V.shape[1] > max_columns:
        rng = np.random.default_rng(random_state)
        V = V[:, np.sort(rng.choice(V.shape[1], size=max_columns, replace=False))]

    return V / (np.sqrt(np.sum(V**2, axis=0)) + EPS)


def concatenate_dictionaries(*dictionaries):
    """Stack dictionaries side by side, re-normalising the columns."""
    W = np.concatenate(dictionaries, axis=1)
    return W / (np.sqrt(np.sum(W**2, axis=0)) + EPS)
