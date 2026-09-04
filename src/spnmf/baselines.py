"""Reference harmonic/percussive separators to compare SPNMF against."""

from __future__ import annotations

import numpy as np

from .separation import Separation, _as_channels
from .stft import istft, stft, wiener_mask

__all__ = ["median_hpss"]


def _median_filter(A, size, axis, block=64):
    """Median filter along one axis, with edge padding.

    Filtering is done in blocks along the other axis so the sliding-window
    view never materialises the whole ``(F, T, size)`` tensor at once.
    """
    if size <= 1:
        return A.copy()
    size = int(size) | 1  # force odd so the window is centred
    pad = size // 2

    A = np.moveaxis(A, axis, -1)
    padded = np.pad(A, [(0, 0)] * (A.ndim - 1) + [(pad, pad)], mode="edge")

    out = np.empty_like(A)
    for start in range(0, A.shape[0], block):
        stop = min(start + block, A.shape[0])
        windows = np.lib.stride_tricks.sliding_window_view(
            padded[start:stop], size, axis=-1
        )
        out[start:stop] = np.median(windows, axis=-1)
    return np.moveaxis(out, -1, axis)


def median_hpss(
    x,
    sample_rate=44100,
    n_fft=2048,
    hop_length=512,
    window="sqrt_hann",
    kernel_harmonic=31,
    kernel_percussive=31,
    mask_power=2.0,
):
    """Median-filtering harmonic/percussive separation (Fitzgerald, 2010).

    The classic baseline: a horizontal median filter smears out transients and
    leaves the tonal part, a vertical one does the opposite, and the two
    enhanced spectrograms drive a soft mask.  Provided here so SPNMF can be
    measured against something on the same footing.

    Returns a :class:`spnmf.separation.Separation` with ``factorisation=None``.
    """
    channels, was_2d = _as_channels(x)
    n_samples = channels.shape[1]
    mono = channels.mean(axis=0)

    stft_kwargs = dict(n_fft=n_fft, hop_length=hop_length, window=window)
    X_mono = stft(mono, **stft_kwargs)
    V = np.abs(X_mono)

    V_h = _median_filter(V, kernel_harmonic, axis=1)  # smooth along time
    V_p = _median_filter(V, kernel_percussive, axis=0)  # smooth along frequency

    mask_h = wiener_mask(V_h, [V_p], power=mask_power)
    mask_p = 1.0 - mask_h

    harmonic = np.empty_like(channels)
    percussive = np.empty_like(channels)
    for c, channel in enumerate(channels):
        X_c = X_mono if channels.shape[0] == 1 else stft(channel, **stft_kwargs)
        harmonic[c] = istft(mask_h * X_c, length=n_samples, **stft_kwargs)
        percussive[c] = istft(mask_p * X_c, length=n_samples, **stft_kwargs)

    if not was_2d:
        harmonic, percussive = harmonic[0], percussive[0]
    else:
        harmonic, percussive = harmonic.T, percussive.T

    return Separation(
        harmonic=harmonic,
        percussive=percussive,
        sample_rate=sample_rate,
        factorisation=None,
        stft_params=dict(stft_kwargs, mask_power=mask_power),
    )
