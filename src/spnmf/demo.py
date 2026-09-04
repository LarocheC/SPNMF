"""A self-contained demonstration: synthesise, separate, score.

Runs SPNMF and the median-filter baseline on a synthetic mixture whose
harmonic and percussive parts are known exactly, so every number below is a
real measurement rather than a claim.
"""

from __future__ import annotations

import numpy as np

from .baselines import median_hpss
from .dictionary import learn_dictionary
from .metrics import bss_eval_sources
from .separation import separate
from .signals import percussive_signal, synthetic_mixture

__all__ = ["run_demo"]


def run_demo(
    outdir=None,
    duration=4.0,
    sample_rate=16000,
    n_fft=1024,
    hop_length=256,
    n_iter=150,
    random_state=0,
    verbose=True,
):
    """Compare SPNMF variants against a median-filter baseline.

    Returns a list of ``dict`` rows with SDR/SIR/SAR for both sources.  When
    ``outdir`` is given, the mixture, the references and every estimate are
    written there as WAV files.
    """
    mixture, harmonic, percussive, sample_rate = synthetic_mixture(
        duration=duration, sample_rate=sample_rate, random_state=random_state
    )
    references = np.stack([harmonic, percussive])

    # The dictionary is trained on *different* drums (another tempo, another
    # seed) than the ones in the test mixture -- training on the test track
    # itself would make the numbers meaningless.
    training = percussive_signal(
        duration=duration, sample_rate=sample_rate, bpm=96.0, random_state=99
    )
    dictionary = learn_dictionary(
        training,
        n_components=12,
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=200,
        random_state=random_state,
    )

    stft_kwargs = dict(n_fft=n_fft, hop_length=hop_length)
    runs = [("median HPSS (baseline)", median_hpss(mixture, sample_rate, **stft_kwargs))]
    for divergence in ("euclidean", "kl", "is"):
        runs.append(
            (
                f"SPNMF unsupervised [{divergence}]",
                separate(
                    mixture, sample_rate, divergence=divergence, n_harmonic=10,
                    n_percussive=10, n_iter=n_iter, tol=None,
                    random_state=random_state, **stft_kwargs,
                ),
            )
        )
    for divergence in ("euclidean", "kl", "is"):
        runs.append(
            (
                f"SPNMF + drum dictionary [{divergence}]",
                separate(
                    mixture, sample_rate, divergence=divergence, W_p=dictionary,
                    n_harmonic=10, n_iter=n_iter, tol=None,
                    random_state=random_state, **stft_kwargs,
                ),
            )
        )

    rows = []
    for name, result in runs:
        estimates = np.stack([result.harmonic, result.percussive])
        scores = bss_eval_sources(references, estimates, filter_length=256)
        rows.append(
            {
                "method": name,
                "sdr_harmonic": scores.sdr[0], "sdr_percussive": scores.sdr[1],
                "sir_harmonic": scores.sir[0], "sir_percussive": scores.sir[1],
                "sar_harmonic": scores.sar[0], "sar_percussive": scores.sar[1],
                "swapped": bool(getattr(result, "swapped", False)),
            }
        )

    if verbose:
        header = f"{'method':38s} {'SDR h':>7s} {'SDR p':>7s} {'SIR h':>7s} {'SIR p':>7s}"
        print(header)
        print("-" * len(header))
        for row in rows:
            print(
                f"{row['method']:38s} {row['sdr_harmonic']:7.2f} "
                f"{row['sdr_percussive']:7.2f} {row['sir_harmonic']:7.2f} "
                f"{row['sir_percussive']:7.2f}"
            )

    if outdir is not None:
        from pathlib import Path

        from .io import write_audio

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        write_audio(outdir / "mixture.wav", mixture, sample_rate)
        write_audio(outdir / "reference_harmonic.wav", harmonic, sample_rate)
        write_audio(outdir / "reference_percussive.wav", percussive, sample_rate)
        for name, result in runs:
            slug = (
                name.lower()
                .replace(" + ", "_")
                .replace(" ", "_")
                .replace("[", "")
                .replace("]", "")
                .replace("(", "")
                .replace(")", "")
            )
            write_audio(outdir / f"{slug}_harmonic.wav", result.harmonic, sample_rate)
            write_audio(outdir / f"{slug}_percussive.wav", result.percussive, sample_rate)
        if verbose:
            print(f"\nwrote {len(list(outdir.glob('*.wav')))} WAV files to {outdir}")

    return rows
