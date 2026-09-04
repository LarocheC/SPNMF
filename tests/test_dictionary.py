import numpy as np
import pytest

from spnmf.dictionary import concatenate_dictionaries, learn_dictionary, stft_dictionary
from spnmf.signals import percussive_signal

STFT = dict(n_fft=512, hop_length=128)
BINS = STFT["n_fft"] // 2 + 1


@pytest.fixture(scope="module")
def drums():
    return percussive_signal(duration=2.0, sample_rate=16000, random_state=0)


def test_learned_dictionary_has_unit_norm_columns(drums):
    W = learn_dictionary(drums, n_components=8, n_iter=60, random_state=0, **STFT)
    assert W.shape == (BINS, 8)
    assert np.all(W >= 0)
    np.testing.assert_allclose(np.linalg.norm(W, axis=0), 1.0, atol=1e-8)


def test_multiple_training_signals_are_concatenated(drums):
    other = percussive_signal(duration=2.0, sample_rate=16000, bpm=90, random_state=1)
    W = learn_dictionary([drums, other], n_components=6, n_iter=40, random_state=0, **STFT)
    assert W.shape == (BINS, 6)


def test_empty_training_set_is_rejected():
    with pytest.raises(ValueError, match="no training signals"):
        learn_dictionary([], n_components=4, **STFT)


def test_stft_dictionary_drops_silent_frames_and_normalises(drums):
    padded = np.concatenate([np.zeros(16000), drums])
    W = stft_dictionary(padded, **STFT)
    assert W.shape[0] == BINS
    assert W.shape[1] < padded.size // STFT["hop_length"]
    np.testing.assert_allclose(np.linalg.norm(W, axis=0), 1.0, atol=1e-8)


def test_stft_dictionary_subsamples_to_max_columns(drums):
    W = stft_dictionary(drums, max_columns=20, random_state=0, **STFT)
    assert W.shape == (BINS, 20)


def test_stft_dictionary_rejects_silence():
    with pytest.raises(ValueError, match="energy threshold"):
        stft_dictionary(np.zeros(8000), **STFT)


def test_concatenate_renormalises(drums):
    a = learn_dictionary(drums, n_components=4, n_iter=30, random_state=0, **STFT)
    b = stft_dictionary(drums, max_columns=5, random_state=0, **STFT)
    W = concatenate_dictionaries(a, 3.0 * b)
    assert W.shape == (BINS, 9)
    np.testing.assert_allclose(np.linalg.norm(W, axis=0), 1.0, atol=1e-8)
