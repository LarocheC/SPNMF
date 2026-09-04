"""Reproduce the paper's core comparison on a synthetic mixture.

Prints the same SDR/SIR/SAR table as ``spnmf demo``, and additionally shows
what happens with ``assign='projective'`` -- the mapping the paper assumes,
which only holds once a drum dictionary constrains the NMF part.
"""

from __future__ import annotations

import numpy as np

from spnmf import bss_eval_sources, learn_dictionary, separate, synthetic_mixture
from spnmf.signals import percussive_signal

STFT = dict(n_fft=1024, hop_length=256)


def main():
    mixture, harmonic, percussive, sample_rate = synthetic_mixture(
        duration=4.0, sample_rate=16000, random_state=0
    )
    references = np.stack([harmonic, percussive])

    training = percussive_signal(4.0, sample_rate, bpm=96.0, random_state=99)
    dictionary = learn_dictionary(
        training, n_components=12, n_iter=200, random_state=0, **STFT
    )

    print(f"{'configuration':46s} {'SDR h':>7s} {'SDR p':>7s}  source order")
    print("-" * 78)
    for label, kwargs in [
        ("unsupervised, assign='projective' (as published)",
         dict(n_percussive=10, assign="projective")),
        ("unsupervised, assign='auto'",
         dict(n_percussive=10, assign="auto")),
        ("drum dictionary, assign='projective'",
         dict(W_p=dictionary, assign="projective")),
        ("drum dictionary, assign='auto'",
         dict(W_p=dictionary, assign="auto")),
    ]:
        result = separate(
            mixture, sample_rate, n_harmonic=10, n_iter=150, tol=None,
            random_state=0, **STFT, **kwargs,
        )
        scores = bss_eval_sources(
            references,
            np.stack([result.harmonic, result.percussive]),
            filter_length=256,
        )
        order = "correct" if list(scores.permutation) == [0, 1] else "SWAPPED"
        print(f"{label:46s} {scores.sdr[0]:7.2f} {scores.sdr[1]:7.2f}  {order}")


if __name__ == "__main__":
    main()
