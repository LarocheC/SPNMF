import numpy as np
import pytest

from spnmf.io import _read_wav_stdlib, _write_wav_stdlib, read_audio, write_audio


@pytest.mark.parametrize("shape", [(2000,), (2000, 2)])
def test_round_trip_preserves_shape_and_values(tmp_path, shape):
    x = np.random.default_rng(0).standard_normal(shape)
    x /= np.abs(x).max() * 1.01
    path = tmp_path / "audio.wav"
    write_audio(path, x, 16000)
    y, rate = read_audio(path)
    assert rate == 16000
    assert y.shape == x.shape
    np.testing.assert_allclose(y, x, atol=2 / 32768)


def test_stdlib_fallback_round_trips(tmp_path):
    x = np.linspace(-0.9, 0.9, 500)
    path = tmp_path / "plain.wav"
    _write_wav_stdlib(path, x, 8000)
    y, rate = _read_wav_stdlib(path)
    assert rate == 8000
    np.testing.assert_allclose(y, x, atol=2 / 32768)


def test_values_outside_the_range_are_clipped(tmp_path):
    path = tmp_path / "loud.wav"
    _write_wav_stdlib(path, np.array([-4.0, 0.0, 4.0]), 8000)
    y, _ = _read_wav_stdlib(path)
    assert y.min() >= -1.0
    assert y.max() <= 1.0
