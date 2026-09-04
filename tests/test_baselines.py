import numpy as np

from spnmf import median_hpss, synthetic_mixture
from spnmf.baselines import _median_filter


def test_median_filter_matches_a_naive_implementation():
    rng = np.random.default_rng(0)
    A = rng.random((9, 11))
    size = 5
    pad = size // 2
    padded = np.pad(A, ((0, 0), (pad, pad)), mode="edge")
    expected = np.stack(
        [
            [np.median(padded[i, j : j + size]) for j in range(A.shape[1])]
            for i in range(A.shape[0])
        ]
    )
    np.testing.assert_allclose(_median_filter(A, size, axis=1), expected)


def test_median_filter_blocking_does_not_change_the_result():
    A = np.random.default_rng(1).random((70, 30))
    np.testing.assert_allclose(
        _median_filter(A, 7, axis=1, block=8), _median_filter(A, 7, axis=1, block=1000)
    )


def test_median_filter_of_size_one_is_the_identity():
    A = np.random.default_rng(2).random((5, 6))
    np.testing.assert_array_equal(_median_filter(A, 1, axis=0), A)


def test_baseline_outputs_sum_back_to_the_mixture():
    mix, _, _, sample_rate = synthetic_mixture(duration=2.0, sample_rate=16000)
    result = median_hpss(mix, sample_rate, n_fft=1024, hop_length=256)
    np.testing.assert_allclose(result.harmonic + result.percussive, mix, atol=1e-10)
    assert result.factorisation is None
    assert result.cost is None


def test_baseline_handles_stereo():
    mix, _, _, sample_rate = synthetic_mixture(duration=1.5, sample_rate=16000)
    stereo = np.stack([mix, mix * 0.5], axis=1)
    result = median_hpss(stereo, sample_rate, n_fft=512, hop_length=128)
    assert result.harmonic.shape == stereo.shape
