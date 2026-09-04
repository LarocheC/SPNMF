"""High-level harmonic/percussive separation built on SPNMF."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import SPNMFResult, spnmf
from .metrics import spectral_flatness
from .stft import istft, stft, wiener_mask

__all__ = ["Separation", "separate"]


@dataclass
class Separation:
    """Time-domain output of :func:`separate`."""

    harmonic: np.ndarray
    """Harmonic/tonal signal, same shape and length as the input."""

    percussive: np.ndarray
    """Percussive/transient signal, same shape and length as the input."""

    sample_rate: int
    factorisation: SPNMFResult = field(repr=False)
    stft_params: dict = field(default_factory=dict, repr=False)

    assignment: str = "projective"
    """How the two model parts were mapped onto the two outputs."""

    swapped: bool = False
    """True when the NMF part -- not the projective part -- became ``harmonic``."""

    @property
    def cost(self):
        return None if self.factorisation is None else self.factorisation.cost


def _as_channels(x):
    """Normalise input to ``(n_channels, n_samples)`` and remember the layout."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return x[None, :], False
    if x.ndim == 2:
        # soundfile hands back (n_samples, n_channels)
        return x.T, True
    raise ValueError(f"expected a 1-D or 2-D signal, got shape {x.shape}")


def separate(
    x,
    sample_rate=44100,
    n_fft=2048,
    hop_length=512,
    window="sqrt_hann",
    power=1.0,
    n_harmonic=20,
    n_percussive=20,
    divergence="kl",
    W_p=None,
    update_dictionary=None,
    n_iter=200,
    tol=1e-5,
    mask_power=2.0,
    assign="auto",
    random_state=None,
    verbose=False,
):
    """Split a signal into harmonic and percussive parts with SPNMF.

    The factorisation is estimated once on the mono downmix; the resulting
    masks are then applied to every channel, which keeps the stereo image
    intact and stops the two channels from converging on different component
    orderings.

    Parameters
    ----------
    x : array
        Input signal, ``(n_samples,)`` or ``(n_samples, n_channels)``.
    sample_rate : int
        Only carried through to the result; the algorithm itself is
        rate-agnostic.
    n_fft, hop_length, window : see :func:`spnmf.stft.stft`
    power : float
        Exponent of the spectrogram handed to the factorisation.  ``1.0``
        (magnitude) is what the paper uses; ``2.0`` gives a power spectrogram,
        the natural companion to the IS divergence.
    n_harmonic, n_percussive, divergence, W_p, update_dictionary, n_iter, tol
        Passed straight through to :func:`spnmf.core.spnmf`.
    mask_power : float
        Exponent of the generalised Wiener mask; ``2.0`` is the usual Wiener
        filter.
    assign : {'auto', 'projective'}
        Which model part becomes the harmonic output.

        ``'projective'`` always takes ``W_h @ W_h.T @ V``, exactly as in the
        paper.  That mapping only holds when the percussive dictionary pins
        the NMF part down; run unsupervised, SPNMF just as happily puts the
        drums in the projective part and the tonal source in the NMF part.

        ``'auto'`` (the default) compares the spectral flatness of the two
        parts and calls the flatter one percussive.  With a trained
        dictionary it agrees with ``'projective'``; without one it is what
        keeps the output labels meaningful.  :attr:`Separation.swapped`
        records whether it changed anything.
    random_state : int or numpy.random.Generator, optional

    Returns
    -------
    Separation
    """
    channels, was_2d = _as_channels(x)
    n_samples = channels.shape[1]
    mono = channels.mean(axis=0)

    stft_kwargs = dict(n_fft=n_fft, hop_length=hop_length, window=window)
    X_mono = stft(mono, **stft_kwargs)
    V = np.abs(X_mono) ** power

    result = spnmf(
        V,
        n_harmonic=n_harmonic,
        n_percussive=n_percussive,
        divergence=divergence,
        W_p=W_p,
        update_dictionary=update_dictionary,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        verbose=verbose,
    )

    # Back to magnitudes before masking, so `power` and `mask_power` stay
    # independent knobs.
    inv = 1.0 / power
    mag_h = np.maximum(result.harmonic, 0.0) ** inv
    mag_p = np.maximum(result.percussive, 0.0) ** inv

    if assign not in ("auto", "projective"):
        raise ValueError(f"assign must be 'auto' or 'projective', got {assign!r}")
    swapped = assign == "auto" and spectral_flatness(mag_h) > spectral_flatness(mag_p)
    if swapped:
        mag_h, mag_p = mag_p, mag_h

    mask_h = wiener_mask(mag_h, [mag_p], power=mask_power)
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
        factorisation=result,
        stft_params=dict(stft_kwargs, power=power, mask_power=mask_power),
        assignment=assign,
        swapped=swapped,
    )
