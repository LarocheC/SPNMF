"""Analysis tools built on the harmonic/percussive decomposition.

Two things the decomposition is good for that separation itself is not:

* :func:`characterise` places a recording on a tonal/impulsive map, which is
  how you stratify a noise corpus that has no labels.
* :func:`per_part_snr` splits the error of a processed signal into what it did
  to steady content and what it did to transients, which a single pooled score
  cannot show you.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baselines import median_hpss
from .divergence import EPS
from .separation import separate
from .stft import istft, stft, wiener_mask

__all__ = [
    "Character",
    "STEADY_THRESHOLD",
    "TONAL_THRESHOLD",
    "PartSNR",
    "characterise",
    "per_part_snr",
    "temporal_flatness",
    "tonal_fraction",
]


def _mono(x):
    x = np.asarray(x, dtype=np.float64)
    return x if x.ndim == 1 else x.mean(axis=1)


def temporal_flatness(x, sample_rate=16000, n_fft=1024, hop_length=256, eps=1e-12):
    """Geometric over arithmetic mean of frame energy.

    The time-domain dual of spectral flatness: near 1 for a steady recording,
    near 0 for one made of isolated bursts.  Independent of
    :func:`tonal_fraction`, which is what makes the pair useful -- broadband
    stationary noise and broadband impulsive noise look identical on the
    tonal axis alone.
    """
    energy = np.maximum(np.sum(np.abs(stft(_mono(x), n_fft, hop_length)) ** 2, axis=0), eps)
    return float(np.exp(np.mean(np.log(energy))) / np.mean(energy))


def tonal_fraction(
    x, sample_rate=16000, n_fft=1024, hop_length=256, method="median", **kwargs
):
    """Fraction of a recording's energy that lands in the tonal part.

    Near 1 for hum, whine and drone; near 0 for clicks, slams and broadband
    hiss.

    ``method='median'`` (the default) uses :func:`spnmf.baselines.median_hpss`.
    ``method='spnmf'`` uses this package's own factorisation and **is not
    recommended for this measurement**.  SPNMF's projective part is a residual
    sponge: the energy it takes is decided by whatever its partner cannot
    explain rather than by signal content, so the resulting split carries almost
    no information about the input and the ratio saturates near 1.0 either way.
    Measured over a labelled noise bank, the median split separates tonal from
    broadband by a clear margin (0.89-1.00 against 0.00-0.51) while the SPNMF
    ratio has no discriminative power at all.  The option exists so that is
    reproducible, not because it works.
    """
    mono = _mono(x)
    stft_kwargs = dict(n_fft=n_fft, hop_length=hop_length)
    if method == "median":
        result = median_hpss(mono, sample_rate, **stft_kwargs, **kwargs)
    elif method == "spnmf":
        kwargs.setdefault("n_harmonic", 16)
        kwargs.setdefault("n_percussive", 16)
        kwargs.setdefault("n_iter", 120)
        kwargs.setdefault("random_state", 0)
        result = separate(mono, sample_rate, **stft_kwargs, **kwargs)
    else:
        raise ValueError(f"method must be 'median' or 'spnmf', got {method!r}")
    tonal = float(np.sum(result.harmonic**2))
    transient = float(np.sum(result.percussive**2))
    return tonal / (tonal + transient + EPS)


#: Default cut between tonal and broadband.  Sits in the gap measured over a
#: labelled noise bank: tonal sources scored 0.89-1.00, broadband ones
#: 0.00-0.51.  Well clear of both, but re-check it on your own corpus.
TONAL_THRESHOLD = 0.7

#: Default cut between steady and bursty, same caveat.
STEADY_THRESHOLD = 0.35


@dataclass
class Character:
    """Where a recording sits on the tonal/impulsive map."""

    tonal_fraction: float
    """1 = purely tonal, 0 = purely broadband."""

    temporal_flatness: float
    """1 = steady throughout, 0 = isolated bursts."""

    tonal_threshold: float = TONAL_THRESHOLD
    steady_threshold: float = STEADY_THRESHOLD

    @property
    def label(self):
        """A coarse bucket, useful for stratifying a corpus."""
        steady = self.temporal_flatness > self.steady_threshold
        if self.tonal_fraction > self.tonal_threshold:
            return "stationary tonal" if steady else "impulsive tonal"
        return "stationary broadband" if steady else "impulsive broadband"


def characterise(
    x,
    sample_rate=16000,
    n_fft=1024,
    hop_length=256,
    tonal_threshold=TONAL_THRESHOLD,
    steady_threshold=STEADY_THRESHOLD,
    **kwargs,
):
    """Describe a recording on both axes at once.  See :class:`Character`.

    The two axes are independent and both are needed: broadband hiss and
    keyboard clicks are indistinguishable on the tonal axis alone, and only
    the steadiness axis tells them apart.
    """
    stft_kwargs = dict(n_fft=n_fft, hop_length=hop_length)
    return Character(
        tonal_fraction=tonal_fraction(x, sample_rate, **stft_kwargs, **kwargs),
        temporal_flatness=temporal_flatness(x, sample_rate, **stft_kwargs),
        tonal_threshold=tonal_threshold,
        steady_threshold=steady_threshold,
    )


@dataclass
class PartSNR:
    """Error of an estimate, broken out by the content it damaged."""

    overall: float
    tonal: float
    transient: float
    tonal_energy_fraction: float
    """Share of the reference's energy that is tonal -- how much of `overall`
    the tonal score is allowed to speak for."""

    def __repr__(self):
        return (
            f"PartSNR(overall={self.overall:.2f} dB, tonal={self.tonal:.2f} dB, "
            f"transient={self.transient:.2f} dB, "
            f"tonal_energy_fraction={self.tonal_energy_fraction:.3f})"
        )


def _snr(reference, estimate):
    error = reference - estimate
    return float(
        10 * np.log10((np.sum(reference**2) + EPS) / (np.sum(error**2) + EPS))
    )


def per_part_snr(
    reference,
    estimate,
    sample_rate=16000,
    n_fft=1024,
    hop_length=256,
    window="sqrt_hann",
    decomposition="median",
    mask_power=2.0,
    **kwargs,
):
    """Split the error between ``estimate`` and ``reference`` by content type.

    A single SNR is dominated by whatever carries the most energy, which for
    speech is the voiced, tonal part.  Damage to transients -- consonants,
    onsets, the things a coarsely quantised or early-exited model gives up
    first -- barely moves it.  This reports both separately.

    The tonal/transient split is taken from the **reference**, and the same
    masks are applied to both signals, so the measurement axis does not shift
    with the quality of the estimate.

    Parameters
    ----------
    reference, estimate : array
        Clean reference and the signal to score.  Mono, equal length.
    decomposition : {'median', 'spnmf'}
        How to split the reference.  ``'median'`` (Fitzgerald median filtering)
        is the default: deterministic, cheap, and it produces a balanced split.
        ``'spnmf'`` uses this package's own factorisation, whose projective
        part is greedy enough to claim ~97% of clean speech energy against the
        median split's 75% -- the two agree on the *direction* of a
        degradation, but only the median split's absolute numbers mean much.

    Returns
    -------
    PartSNR
    """
    reference = _mono(reference)
    estimate = _mono(estimate)
    n = min(len(reference), len(estimate))
    reference, estimate = reference[:n], estimate[:n]

    stft_kwargs = dict(n_fft=n_fft, hop_length=hop_length, window=window)
    if decomposition == "median":
        parts = median_hpss(reference, sample_rate, mask_power=mask_power, **stft_kwargs)
        X_ref = stft(reference, **stft_kwargs)
        V_t = np.abs(stft(parts.harmonic, **stft_kwargs))
        V_p = np.abs(stft(parts.percussive, **stft_kwargs))
    elif decomposition == "spnmf":
        kwargs.setdefault("n_harmonic", 16)
        kwargs.setdefault("n_percussive", 16)
        kwargs.setdefault("n_iter", 120)
        kwargs.setdefault("random_state", 0)
        parts = separate(
            reference, sample_rate, mask_power=mask_power, **stft_kwargs, **kwargs
        )
        X_ref = stft(reference, **stft_kwargs)
        V_t = np.abs(stft(parts.harmonic, **stft_kwargs))
        V_p = np.abs(stft(parts.percussive, **stft_kwargs))
    else:
        raise ValueError(
            f"decomposition must be 'median' or 'spnmf', got {decomposition!r}"
        )

    mask_t = wiener_mask(V_t, [V_p], power=mask_power)
    X_est = stft(estimate, **stft_kwargs)

    scores = {}
    tonal_energy = 0.0
    total_energy = 0.0
    for name, mask in (("tonal", mask_t), ("transient", 1.0 - mask_t)):
        ref_part = istft(mask * X_ref, length=n, **stft_kwargs)
        est_part = istft(mask * X_est, length=n, **stft_kwargs)
        scores[name] = _snr(ref_part, est_part)
        energy = float(np.sum(ref_part**2))
        total_energy += energy
        if name == "tonal":
            tonal_energy = energy

    return PartSNR(
        overall=_snr(reference, estimate),
        tonal=scores["tonal"],
        transient=scores["transient"],
        tonal_energy_fraction=tonal_energy / (total_energy + EPS),
    )
