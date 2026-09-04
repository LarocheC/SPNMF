import numpy as np
import pytest

from spnmf import bss_eval_sources, median_hpss, separate, synthetic_mixture
from spnmf.dictionary import learn_dictionary
from spnmf.signals import percussive_signal

STFT = dict(n_fft=1024, hop_length=256)


@pytest.fixture(scope="module")
def mixture():
    return synthetic_mixture(duration=3.0, sample_rate=16000, random_state=0)


@pytest.fixture(scope="module")
def drum_dictionary(mixture):
    """Trained on different drums than the test mixture contains."""
    _, _, _, sample_rate = mixture
    training = percussive_signal(3.0, sample_rate, bpm=96.0, random_state=99)
    return learn_dictionary(
        training, n_components=12, n_iter=150, random_state=0, **STFT
    )


def _sdr(mixture, result):
    _, harmonic, percussive, _ = mixture
    return bss_eval_sources(
        np.stack([harmonic, percussive]),
        np.stack([result.harmonic, result.percussive]),
        filter_length=256,
    )


def test_outputs_sum_back_to_the_mixture(mixture):
    mix, _, _, sample_rate = mixture
    result = separate(mix, sample_rate, n_harmonic=6, n_percussive=6, n_iter=25,
                      random_state=0, **STFT)
    np.testing.assert_allclose(result.harmonic + result.percussive, mix, atol=1e-10)


def test_stereo_in_stereo_out(mixture):
    mix, _, _, sample_rate = mixture
    stereo = np.stack([mix, np.roll(mix, 7)], axis=1)
    result = separate(stereo, sample_rate, n_harmonic=6, n_percussive=6, n_iter=20,
                      random_state=0, **STFT)
    assert result.harmonic.shape == stereo.shape
    np.testing.assert_allclose(result.harmonic + result.percussive, stereo, atol=1e-10)


def test_rejects_three_dimensional_input():
    with pytest.raises(ValueError, match="1-D or 2-D"):
        separate(np.zeros((2, 2, 2)))


@pytest.mark.parametrize("divergence", ["euclidean", "kl", "is"])
def test_trained_dictionary_puts_the_sources_the_right_way_round(
    mixture, drum_dictionary, divergence
):
    mix, _, _, sample_rate = mixture
    result = separate(mix, sample_rate, divergence=divergence, W_p=drum_dictionary,
                      n_harmonic=10, n_iter=120, tol=None, random_state=0, **STFT)
    scores = _sdr(mixture, result)
    assert list(scores.permutation) == [0, 1]
    assert not result.swapped
    assert scores.sdr[0] > 18.0
    assert scores.sdr[1] > 8.0


def test_trained_dictionary_beats_the_median_baseline(mixture, drum_dictionary):
    mix, _, _, sample_rate = mixture
    spnmf_scores = _sdr(
        mixture,
        separate(mix, sample_rate, W_p=drum_dictionary, n_harmonic=10,
                 n_iter=120, tol=None, random_state=0, **STFT),
    )
    baseline_scores = _sdr(mixture, median_hpss(mix, sample_rate, **STFT))
    assert spnmf_scores.sdr[0] > baseline_scores.sdr[0]
    assert spnmf_scores.sdr[1] > baseline_scores.sdr[1]


def test_auto_assignment_fixes_the_unsupervised_swap(mixture):
    """Unsupervised, SPNMF happily puts the drums in the projective part.

    The paper reports the same thing; `assign='auto'` is what keeps the
    output labels meaningful when no dictionary pins the parts down.
    """
    mix, _, _, sample_rate = mixture
    auto = separate(mix, sample_rate, n_harmonic=10, n_percussive=10, n_iter=120,
                    tol=None, assign="auto", random_state=0, **STFT)
    fixed = separate(mix, sample_rate, n_harmonic=10, n_percussive=10, n_iter=120,
                     tol=None, assign="projective", random_state=0, **STFT)

    assert auto.swapped
    assert list(_sdr(mixture, auto).permutation) == [0, 1]
    assert list(_sdr(mixture, fixed).permutation) == [1, 0]
    np.testing.assert_allclose(auto.harmonic, fixed.percussive, atol=1e-10)


def test_rejects_unknown_assignment(mixture):
    mix, _, _, sample_rate = mixture
    with pytest.raises(ValueError, match="assign"):
        separate(mix, sample_rate, n_iter=2, assign="magic", **STFT)


def test_result_carries_diagnostics(mixture):
    mix, _, _, sample_rate = mixture
    result = separate(mix, sample_rate, n_harmonic=5, n_percussive=5, n_iter=15,
                      random_state=0, **STFT)
    assert result.sample_rate == sample_rate
    assert len(result.cost) == result.factorisation.n_iter
    assert result.stft_params["n_fft"] == STFT["n_fft"]
