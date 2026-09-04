"""STFT analysis/synthesis and generalised Wiener masking.

Analysis and synthesis share the same window and are combined with
weighted overlap-add, so :func:`istft` inverts :func:`stft` exactly (to
floating-point precision) for any window/hop pair whose squared window sum
stays away from zero -- no COLA condition to remember.
"""

from __future__ import annotations

import numpy as np

from .divergence import EPS

__all__ = ["stft", "istft", "wiener_mask", "get_window"]


def get_window(window, n_fft):
    """Return a window of length ``n_fft``.

    Accepts ``'hann'``, ``'sqrt_hann'``, ``'hamming'``, ``'rect'`` or a
    ready-made array.
    """
    if isinstance(window, np.ndarray):
        if window.shape != (n_fft,):
            raise ValueError(
                f"window has length {window.shape[0]}, expected {n_fft}"
            )
        return window.astype(np.float64)

    name = str(window).lower()
    n = np.arange(n_fft)
    if name in ("hann", "hanning"):
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / n_fft)
    if name in ("sqrt_hann", "hann_sqrt", "root_hann"):
        return np.sqrt(0.5 - 0.5 * np.cos(2.0 * np.pi * n / n_fft))
    if name == "hamming":
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / n_fft)
    if name in ("rect", "boxcar", "none"):
        return np.ones(n_fft)
    raise ValueError(f"unknown window {window!r}")


def stft(x, n_fft=2048, hop_length=None, window="sqrt_hann", center=True):
    """Short-time Fourier transform of a 1-D signal.

    Returns a complex array of shape ``(n_fft // 2 + 1, n_frames)``.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    hop_length = n_fft // 4 if hop_length is None else int(hop_length)
    if not 0 < hop_length <= n_fft:
        raise ValueError("hop_length must be in (0, n_fft]")

    w = get_window(window, n_fft)
    pad_left = n_fft // 2 if center else 0
    n_frames = int(np.ceil(len(x) / hop_length)) + 1
    needed = (n_frames - 1) * hop_length + n_fft
    pad_right = max(0, needed - pad_left - len(x))
    xp = np.pad(x, (pad_left, pad_right))

    frames = np.lib.stride_tricks.sliding_window_view(xp, n_fft)[::hop_length]
    frames = frames[:n_frames] * w
    return np.fft.rfft(frames, n=n_fft, axis=1).T


def istft(X, n_fft=2048, hop_length=None, window="sqrt_hann", center=True, length=None):
    """Invert :func:`stft` by weighted overlap-add."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    hop_length = n_fft // 4 if hop_length is None else int(hop_length)
    w = get_window(window, n_fft)
    w_sq = w**2

    frames = np.fft.irfft(X, n=n_fft, axis=0).T * w
    n_frames = frames.shape[0]
    total = (n_frames - 1) * hop_length + n_fft

    y = np.zeros(total)
    norm = np.zeros(total)
    for t in range(n_frames):
        start = t * hop_length
        y[start : start + n_fft] += frames[t]
        norm[start : start + n_fft] += w_sq
    y /= np.maximum(norm, EPS)

    pad_left = n_fft // 2 if center else 0
    y = y[pad_left:]
    if length is not None:
        y = y[:length]
        if len(y) < length:
            y = np.pad(y, (0, length - len(y)))
    return y


def wiener_mask(target, others, power=2.0):
    """Generalised Wiener mask ``target**p / (target**p + sum(others**p))``.

    ``power=2`` is the usual Wiener filter, ``power=1`` a ratio mask.  The
    masks built this way for a full set of sources sum to one, so the
    separated signals add back up to the mixture.
    """
    target = np.asarray(target, dtype=np.float64)
    num = target**power
    den = num.copy()
    for other in others:
        den += np.asarray(other, dtype=np.float64) ** power
    return num / (den + EPS)
