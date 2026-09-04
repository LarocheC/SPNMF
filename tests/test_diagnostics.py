import numpy as np
import pytest

from spnmf.diagnostics import (
    characterise,
    per_part_snr,
    temporal_flatness,
    tonal_fraction,
)

SR = 16000
DUR = 2.0
N = int(SR * DUR)
T = np.arange(N) / SR
STFT = dict(n_fft=1024, hop_length=256)


def _tone(freq=440.0):
    return np.sin(2 * np.pi * freq * T)


def _hiss(seed=0):
    return np.random.default_rng(seed).standard_normal(N)


def _clicks(seed=0, rate=6):
    rng = np.random.default_rng(seed)
    out = np.zeros(N)
    for onset in rng.uniform(0, DUR - 0.1, int(rate * DUR)):
        i0 = int(onset * SR)
        u = np.arange(int(0.02 * SR)) / SR
        out[i0 : i0 + len(u)] += np.exp(-u / 0.004) * rng.standard_normal(len(u))
    return out


def test_temporal_flatness_separates_steady_from_bursty():
    assert temporal_flatness(_hiss(), SR, **STFT) > 0.8
    assert temporal_flatness(_clicks(), SR, **STFT) < 0.2


def test_tonal_fraction_separates_tone_from_hiss():
    assert tonal_fraction(_tone(), SR, **STFT) > 0.9
    assert tonal_fraction(_hiss(), SR, **STFT) < 0.7


def test_tonal_fraction_is_bounded():
    for signal in (_tone(), _hiss(), _clicks()):
        assert 0.0 <= tonal_fraction(signal, SR, **STFT) <= 1.0


def test_spnmf_method_runs_but_saturates():
    """Documents why 'median' is the default: the projective part is greedy."""
    tone = tonal_fraction(_tone(), SR, method="spnmf", **STFT)
    hiss = tonal_fraction(_hiss(), SR, method="spnmf", **STFT)
    assert hiss > 0.9  # white noise reads as almost entirely "tonal"
    assert abs(tone - hiss) < 0.1  # so the measure barely discriminates


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="method"):
        tonal_fraction(_tone(), SR, method="wavelet", **STFT)


@pytest.mark.parametrize(
    "signal,expected",
    [
        (_tone(), "stationary tonal"),
        (_hiss(), "stationary broadband"),
        (_clicks(), "impulsive broadband"),
    ],
)
def test_characterise_labels(signal, expected):
    assert characterise(signal, SR, **STFT).label == expected


def test_characterise_thresholds_are_configurable():
    character = characterise(_hiss(), SR, tonal_threshold=0.0, **STFT)
    assert character.label == "stationary tonal"


def test_per_part_snr_is_high_for_an_exact_copy():
    signal = _tone() + 0.5 * _clicks()
    scores = per_part_snr(signal, signal.copy(), SR, **STFT)
    assert scores.overall > 100
    assert scores.tonal > 100
    assert scores.transient > 100


def test_per_part_snr_attributes_damage_to_the_right_part():
    """Corrupt only the transients; the tonal score must barely move."""
    rng = np.random.default_rng(1)
    tonal, transient = _tone(), 0.5 * _clicks()
    reference = tonal + transient
    damaged = tonal + 0.5 * (transient + 0.4 * rng.standard_normal(N) * (transient != 0))

    scores = per_part_snr(reference, damaged, SR, **STFT)
    assert scores.transient < scores.tonal - 10
    assert 0.0 < scores.tonal_energy_fraction < 1.0


def test_per_part_snr_accepts_both_decompositions():
    signal = _tone() + 0.5 * _clicks()
    noisy = signal + 0.05 * np.random.default_rng(2).standard_normal(N)
    for decomposition in ("median", "spnmf"):
        scores = per_part_snr(signal, noisy, SR, decomposition=decomposition, **STFT)
        assert np.isfinite(scores.overall)
        assert np.isfinite(scores.transient)


def test_unknown_decomposition_is_rejected():
    signal = _tone()
    with pytest.raises(ValueError, match="decomposition"):
        per_part_snr(signal, signal, SR, decomposition="wavelet", **STFT)


def test_per_part_snr_handles_length_mismatch_and_stereo():
    signal = _tone() + 0.5 * _clicks()
    scores = per_part_snr(
        np.stack([signal, signal], axis=1), signal[: N - 500], SR, **STFT
    )
    assert np.isfinite(scores.overall)
