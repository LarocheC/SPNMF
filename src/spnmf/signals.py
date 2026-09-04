"""Synthetic harmonic and percussive signals.

Enough to exercise and demonstrate the separator without shipping audio
assets, and to give the tests a mixture whose ground truth is known exactly.
"""

from __future__ import annotations

import numpy as np

__all__ = ["harmonic_signal", "percussive_signal", "synthetic_mixture"]


def harmonic_signal(
    duration=4.0,
    sample_rate=16000,
    midi_notes=(52, 55, 59, 64, 62, 59, 55, 52),
    n_partials=8,
    note_overlap=0.15,
    random_state=None,
):
    """A monophonic line of sustained, harmonically rich notes.

    Sparse in frequency and stable in time -- the component SPNMF's projective
    part is meant to capture.
    """
    rng = np.random.default_rng(random_state)
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    out = np.zeros(n)

    note_len = duration / len(midi_notes)
    for k, midi in enumerate(midi_notes):
        f0 = 440.0 * 2.0 ** ((midi - 69) / 12.0)
        start = k * note_len
        stop = min(duration, start + note_len * (1.0 + note_overlap))
        i0, i1 = int(start * sample_rate), int(stop * sample_rate)
        local = t[i0:i1] - start
        if local.size == 0:
            continue

        # Smooth attack, gentle decay: no broadband transient of its own.
        envelope = (1.0 - np.exp(-local / 0.05)) * np.exp(-local / (note_len * 1.5))
        note = np.zeros_like(local)
        for p in range(1, n_partials + 1):
            freq = f0 * p
            if freq >= sample_rate / 2:
                break
            phase = rng.uniform(0, 2 * np.pi)
            note += (1.0 / p**1.4) * np.sin(2 * np.pi * freq * local + phase)
        out[i0:i1] += envelope * note

    peak = np.max(np.abs(out))
    return out / peak if peak > 0 else out


def percussive_signal(
    duration=4.0,
    sample_rate=16000,
    bpm=120.0,
    random_state=None,
):
    """A drum-like track: broadband snare/hat bursts plus a pitched kick.

    Flat in frequency and short in time -- the component the NMF part of
    SPNMF is meant to capture.
    """
    rng = np.random.default_rng(random_state)
    n = int(duration * sample_rate)
    out = np.zeros(n)
    beat = 60.0 / bpm

    def add(onset, generator, length):
        i0 = int(onset * sample_rate)
        i1 = min(n, i0 + int(length * sample_rate))
        if i1 <= i0:
            return
        local = np.arange(i1 - i0) / sample_rate
        out[i0:i1] += generator(local)

    step = 0
    onset = 0.0
    while onset < duration:
        if step % 4 == 0:  # kick: fast downward pitch sweep
            add(
                onset,
                lambda u: 1.0
                * np.exp(-u / 0.08)
                * np.sin(2 * np.pi * (110.0 * np.exp(-u / 0.03) + 45.0) * u),
                0.35,
            )
        elif step % 4 == 2:  # snare: noise plus a body tone
            noise = rng.standard_normal(int(0.25 * sample_rate) + 1)
            add(
                onset,
                lambda u, noise=noise: 0.7
                * np.exp(-u / 0.06)
                * (noise[: len(u)] + 0.5 * np.sin(2 * np.pi * 190.0 * u)),
                0.25,
            )
        # hi-hat on every eighth
        hat = rng.standard_normal(int(0.06 * sample_rate) + 1)
        add(
            onset,
            lambda u, hat=hat: 0.25 * np.exp(-u / 0.012) * hat[: len(u)],
            0.06,
        )
        step += 1
        onset += beat / 2.0

    peak = np.max(np.abs(out))
    return out / peak if peak > 0 else out


def synthetic_mixture(
    duration=4.0,
    sample_rate=16000,
    percussive_gain=0.7,
    random_state=0,
):
    """Return ``(mixture, harmonic, percussive, sample_rate)``.

    ``mixture`` is exactly ``harmonic + percussive``, so the two parts are
    valid references for :func:`spnmf.metrics.bss_eval_sources`.
    """
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2**31 - 1, size=2)
    harmonic = harmonic_signal(duration, sample_rate, random_state=int(seeds[0]))
    percussive = percussive_gain * percussive_signal(
        duration, sample_rate, random_state=int(seeds[1])
    )
    return harmonic + percussive, harmonic, percussive, sample_rate
