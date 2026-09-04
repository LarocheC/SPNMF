"""SPNMF -- Structured Projective Non-negative Matrix Factorization.

Harmonic/percussive source separation by factorising a spectrogram ``V`` as::

    V ~= W_h @ W_h.T @ V  +  W_p @ H_p

a near-orthogonal projective part for the tonal source and an ordinary NMF
for the transients.

Quick start
-----------
>>> from spnmf import separate, synthetic_mixture
>>> mix, harmonic, percussive, sr = synthetic_mixture(duration=2.0)
>>> out = separate(mix, sr, n_harmonic=8, n_percussive=8, n_iter=60)
>>> out.harmonic.shape == mix.shape
True
"""

from .baselines import median_hpss
from .core import NMFResult, SPNMFResult, nmf, spnmf
from .diagnostics import (
    Character,
    PartSNR,
    characterise,
    per_part_snr,
    temporal_flatness,
    tonal_fraction,
)
from .dictionary import concatenate_dictionaries, learn_dictionary, stft_dictionary
from .divergence import DIVERGENCES, beta_divergence
from .metrics import bss_eval_sources, si_sdr
from .separation import Separation, separate
from .signals import harmonic_signal, percussive_signal, synthetic_mixture
from .stft import istft, stft, wiener_mask

__version__ = "0.2.0"

__all__ = [
    "tonal_fraction",
    "temporal_flatness",
    "per_part_snr",
    "characterise",
    "PartSNR",
    "Character",
    "DIVERGENCES",
    "NMFResult",
    "SPNMFResult",
    "Separation",
    "beta_divergence",
    "bss_eval_sources",
    "concatenate_dictionaries",
    "harmonic_signal",
    "istft",
    "learn_dictionary",
    "median_hpss",
    "nmf",
    "percussive_signal",
    "separate",
    "si_sdr",
    "spnmf",
    "stft",
    "stft_dictionary",
    "synthetic_mixture",
    "wiener_mask",
    "__version__",
]
