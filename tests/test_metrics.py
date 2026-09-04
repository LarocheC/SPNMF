import numpy as np
import pytest

from spnmf.metrics import bss_eval_sources, si_sdr, spectral_flatness
from spnmf.signals import synthetic_mixture


@pytest.fixture
def sources():
    _, harmonic, percussive, _ = synthetic_mixture(duration=1.5, sample_rate=8000)
    return np.stack([harmonic, percussive])


def test_perfect_estimates_score_very_high(sources):
    scores = bss_eval_sources(sources, sources.copy(), filter_length=64)
    assert np.all(scores.sdr > 60)
    assert list(scores.permutation) == [0, 1]


def test_permutation_is_recovered(sources):
    swapped = sources[::-1].copy()
    scores = bss_eval_sources(sources, swapped, filter_length=64)
    assert list(scores.permutation) == [1, 0]
    assert np.all(scores.sdr > 60)


def test_added_noise_lowers_sdr(sources):
    rng = np.random.default_rng(0)
    noisy = sources + 0.1 * rng.standard_normal(sources.shape)
    clean = bss_eval_sources(sources, sources.copy(), filter_length=64)
    dirty = bss_eval_sources(sources, noisy, filter_length=64)
    assert np.all(dirty.sdr < clean.sdr)
    assert np.all(np.isfinite(dirty.sdr))


def test_scores_expose_named_fields(sources):
    scores = bss_eval_sources(sources, sources.copy(), filter_length=64)
    sdr, sir, sar, perm = scores
    np.testing.assert_array_equal(sdr, scores.sdr)
    np.testing.assert_array_equal(sir, scores.sir)
    np.testing.assert_array_equal(sar, scores.sar)
    np.testing.assert_array_equal(perm, scores.permutation)
    assert "BSSEvalScores" in repr(scores)


def test_mismatched_shapes_are_rejected(sources):
    with pytest.raises(ValueError, match="same shape"):
        bss_eval_sources(sources, sources[:, :100])


def test_si_sdr_ignores_scale():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(4000)
    estimate = x + 0.5 * rng.standard_normal(4000)
    # A perfect estimate has no residual left to measure, so scale invariance
    # is checked where the residual is real.
    assert si_sdr(x, 7.5 * estimate) == pytest.approx(si_sdr(x, estimate), abs=1e-9)
    assert 0 < si_sdr(x, estimate) < 20


def test_si_sdr_of_a_perfect_estimate_is_huge():
    x = np.random.default_rng(1).standard_normal(4000)
    assert si_sdr(x, x) > 100


def test_spectral_flatness_separates_tonal_from_noisy():
    rng = np.random.default_rng(2)
    tonal = np.zeros((64, 40))
    tonal[[5, 10, 20], :] = 1.0
    noisy = rng.random((64, 40)) + 0.5
    assert spectral_flatness(tonal) < 0.1
    assert spectral_flatness(noisy) > 0.5
