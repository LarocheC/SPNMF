import numpy as np
import pytest

from spnmf.stft import get_window, istft, stft, wiener_mask


@pytest.mark.parametrize(
    "n_fft,hop_length,window",
    [
        (2048, 512, "sqrt_hann"),
        (1024, 256, "hann"),
        (512, 128, "hamming"),
        (256, 200, "hann"),  # hop that does not divide n_fft
        (128, 128, "rect"),
    ],
)
def test_round_trip_is_exact(n_fft, hop_length, window):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(9000)
    X = stft(x, n_fft, hop_length, window)
    y = istft(X, n_fft, hop_length, window, length=len(x))
    assert y.shape == x.shape
    np.testing.assert_allclose(y, x, atol=1e-10)


def test_output_shape():
    x = np.zeros(10000)
    X = stft(x, n_fft=1024, hop_length=256)
    assert X.shape[0] == 1024 // 2 + 1
    assert X.dtype == np.complex128


def test_signal_shorter_than_one_frame_survives_round_trip():
    x = np.random.default_rng(1).standard_normal(100)
    X = stft(x, n_fft=1024, hop_length=256)
    np.testing.assert_allclose(istft(X, 1024, 256, length=len(x)), x, atol=1e-10)


def test_windows_have_the_right_length_and_shape():
    for name in ["hann", "sqrt_hann", "hamming", "rect"]:
        w = get_window(name, 64)
        assert w.shape == (64,)
        assert np.all(w >= 0)
    np.testing.assert_allclose(get_window("sqrt_hann", 64) ** 2, get_window("hann", 64))


def test_custom_window_array_is_accepted_and_validated():
    w = np.ones(32)
    np.testing.assert_array_equal(get_window(w, 32), w)
    with pytest.raises(ValueError, match="length"):
        get_window(w, 64)
    with pytest.raises(ValueError, match="unknown window"):
        get_window("blackman-nuttall", 32)


def test_stft_rejects_bad_input():
    with pytest.raises(ValueError, match="1-D"):
        stft(np.zeros((4, 4)))
    with pytest.raises(ValueError, match="hop_length"):
        stft(np.zeros(100), n_fft=64, hop_length=0)
    with pytest.raises(ValueError, match="2-D"):
        istft(np.zeros(10))


def test_wiener_masks_partition_the_mixture():
    rng = np.random.default_rng(2)
    a = rng.random((10, 20))
    b = rng.random((10, 20))
    mask_a = wiener_mask(a, [b])
    mask_b = wiener_mask(b, [a])
    np.testing.assert_allclose(mask_a + mask_b, 1.0, atol=1e-12)
    assert np.all((mask_a >= 0) & (mask_a <= 1))


def test_wiener_mask_favours_the_dominant_source():
    loud = np.full((4, 4), 10.0)
    quiet = np.full((4, 4), 0.1)
    assert np.all(wiener_mask(loud, [quiet]) > 0.99)
