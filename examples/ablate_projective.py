"""Does the projective term earn its place?

SPNMF's distinctive claim is the projective part ``W_h W_h^T V``. This ablates
it directly: the same fixed percussive dictionary, the same iteration budget,
the same reconstruction, with the projective term swapped for a plain free
basis. Whatever is left over is what the projection is actually worth.

    python examples/ablate_projective.py
"""

from __future__ import annotations

import numpy as np

from spnmf import (
    bss_eval_sources,
    istft,
    learn_dictionary,
    median_hpss,
    semi_supervised_nmf,
    spnmf,
    stft,
    synthetic_mixture,
    wiener_mask,
)
from spnmf.signals import percussive_signal


def _score(references, harmonic, percussive, label):
    scores = bss_eval_sources(
        references, np.stack([harmonic, percussive]), filter_length=256
    )
    order = "" if list(scores.permutation) == [0, 1] else "  SWAPPED"
    print(f"  {label:46s} SDR h={scores.sdr[0]:6.2f}  p={scores.sdr[1]:6.2f}{order}")
    return scores.sdr


def compare(mixture, harmonic_ref, percussive_ref, sample_rate, W_p,
            n_fft, hop_length, rank=10, n_iter=200):
    X = stft(mixture, n_fft, hop_length)
    V = np.abs(X)
    references = np.stack([harmonic_ref, percussive_ref])
    length = len(mixture)

    def reconstruct(V_h, V_p):
        mask = wiener_mask(V_h, [V_p], power=2.0)
        return (
            istft(mask * X, n_fft, hop_length, length=length),
            istft((1.0 - mask) * X, n_fft, hop_length, length=length),
        )

    out = {}
    result = spnmf(V, n_harmonic=rank, W_p=W_p, divergence="kl",
                   n_iter=n_iter, tol=None, random_state=0)
    out["spnmf"] = _score(references, *reconstruct(result.harmonic, result.percussive),
                          "SPNMF (projective + fixed dictionary)")

    semi = semi_supervised_nmf(V, W_p, n_free=rank, divergence="kl",
                               n_iter=n_iter, tol=None, random_state=0)
    out["semi"] = _score(references, *reconstruct(semi.free, semi.fixed),
                         f"semi-supervised NMF (free {rank}-basis, SAME dict)")

    only = semi_supervised_nmf(V, W_p, n_free=1, divergence="kl",
                               n_iter=n_iter, tol=None, random_state=0)
    out["dict_only"] = _score(references,
                              *reconstruct(np.maximum(V - only.fixed, 0.0), only.fixed),
                              "dictionary only + residual (no free basis)")

    baseline = median_hpss(mixture, sample_rate, n_fft=n_fft, hop_length=hop_length)
    out["median"] = _score(references, baseline.harmonic, baseline.percussive,
                           "median HPSS (classical baseline)")
    return out


def main():
    print("=== synthetic mixture (exact ground truth) ===")
    mixture, harmonic, percussive, sample_rate = synthetic_mixture(
        4.0, 16000, random_state=0
    )
    training = percussive_signal(4.0, sample_rate, bpm=96.0, random_state=99)
    W_p = learn_dictionary(training, n_components=12, n_fft=1024, hop_length=256,
                           n_iter=200, random_state=0)
    compare(mixture, harmonic, percussive, sample_rate, W_p, 1024, 256)

    print("\n=== real music, if the stems are available ===")
    from pathlib import Path

    from spnmf.io import read_audio

    stems = Path(
        "/home/user/larochec/voice-extraction/MusicDelta_FunkJazz_STEMS"
    )
    if not stems.is_dir():
        print("  (stems not present; clone LarocheC/voice-extraction to run this half)")
        return

    def load(index):
        audio, rate = read_audio(stems / f"MusicDelta_FunkJazz_STEM_0{index}.wav")
        return audio.mean(axis=1), rate

    drums, rate = load(1)
    tonal = load(2)[0] + load(3)[0]
    train = slice(0, int(12 * rate))
    test = slice(int(12 * rate), int(30 * rate))

    # The dictionary is trained on drum audio held out from the test excerpt.
    W_p = learn_dictionary(drums[train], n_components=20, n_fft=2048,
                           hop_length=512, n_iter=300, random_state=0)
    d, h = drums[test], tonal[test]
    scale = np.abs(d + h).max()
    d, h = d / scale, h / scale
    compare(d + h, h, d, rate, W_p, 2048, 512, rank=20)


if __name__ == "__main__":
    main()
