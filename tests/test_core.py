import numpy as np
import pytest

from spnmf.core import _projective_terms, nmf, spnmf
from spnmf.divergence import beta_divergence, gradient_split, plus_as_array

BETAS = [("euclidean", 2.0), ("kl", 1.0), ("is", 0.0)]


@pytest.fixture
def structured_spectrogram():
    """A spectrogram with a sparse tonal part and a broadband transient part."""
    rng = np.random.default_rng(0)
    F, T = 48, 90
    W_h = np.zeros((F, 3))
    for j in range(3):
        W_h[(np.arange(1, 7) * (j + 3)) % F, j] = 1.0
    H_h = rng.random((3, T))
    W_p = rng.random((F, 4)) * 0.3 + 0.3
    H_p = (rng.random((4, T)) < 0.15) * rng.random((4, T))
    return W_h @ H_h + W_p @ H_p + 1e-3


@pytest.mark.parametrize("name,beta", BETAS)
def test_spnmf_cost_decreases_monotonically(name, beta, structured_spectrogram):
    result = spnmf(
        structured_spectrogram, n_harmonic=3, n_percussive=4,
        divergence=name, n_iter=120, tol=None, random_state=1,
    )
    assert result.beta == beta
    assert len(result.cost) == 120
    # Allow a hair of numerical slack, but no real ascent.
    assert np.all(np.diff(result.cost) <= 1e-9 * abs(result.cost[0]))
    assert result.cost[-1] < 0.5 * result.cost[0]


@pytest.mark.parametrize("name,_", BETAS)
def test_nmf_cost_decreases_monotonically(name, _, structured_spectrogram):
    result = nmf(
        structured_spectrogram, n_components=6, divergence=name,
        n_iter=100, tol=None, random_state=1,
    )
    assert np.all(np.diff(result.cost) <= 1e-9 * abs(result.cost[0]))
    assert result.cost[-1] < 0.5 * result.cost[0]


@pytest.mark.parametrize("name,beta", BETAS)
def test_projective_gradient_matches_numerical_gradient(name, beta):
    """The W_h update ratio must be built from the true gradient of the cost.

    This is the step the 2016 MATLAB implementation got wrong for the
    Euclidean case, so it is checked directly rather than inferred from
    convergence behaviour.
    """
    rng = np.random.default_rng(3)
    F, T, r, k = 7, 9, 2, 3
    V = rng.random((F, T)) + 0.5
    W_h = rng.random((F, r)) + 0.2
    W_p = rng.random((F, k)) + 0.2
    H_p = rng.random((k, T)) + 0.2

    def cost(W):
        return beta_divergence(V, W @ (W.T @ V) + W_p @ H_p, beta)

    V_hat = W_h @ (W_h.T @ V) + W_p @ H_p
    minus, plus = gradient_split(V, V_hat, beta)
    VtW = V.T @ W_h
    analytic = _projective_terms(V, W_h, plus, VtW) - _projective_terms(
        V, W_h, minus, VtW
    )

    step = 1e-6
    numerical = np.empty_like(W_h)
    for i in range(F):
        for j in range(r):
            up, down = W_h.copy(), W_h.copy()
            up[i, j] += step
            down[i, j] -= step
            numerical[i, j] = (cost(up) - cost(down)) / (2 * step)

    np.testing.assert_allclose(analytic, numerical, rtol=1e-4, atol=1e-6)


def test_projective_terms_ones_shortcut_matches_dense_path():
    rng = np.random.default_rng(4)
    V = rng.random((6, 8)) + 0.1
    W = rng.random((6, 3)) + 0.1
    VtW = V.T @ W
    sparse = _projective_terms(V, W, None, VtW)
    dense = _projective_terms(V, W, plus_as_array(None, V.shape), VtW)
    np.testing.assert_allclose(sparse, dense, rtol=1e-12)


def test_fixed_dictionary_is_not_modified(structured_spectrogram):
    rng = np.random.default_rng(5)
    W_p = rng.random((structured_spectrogram.shape[0], 5)) + 0.1
    result = spnmf(
        structured_spectrogram, n_harmonic=3, W_p=W_p, n_iter=30, random_state=0
    )
    np.testing.assert_array_equal(result.W_p, W_p)


def test_dictionary_is_updated_when_requested(structured_spectrogram):
    rng = np.random.default_rng(5)
    W_p = rng.random((structured_spectrogram.shape[0], 5)) + 0.1
    result = spnmf(
        structured_spectrogram, n_harmonic=3, W_p=W_p, update_dictionary=True,
        n_iter=30, random_state=0,
    )
    assert not np.allclose(result.W_p, W_p)


def test_shapes_and_reconstruction(structured_spectrogram):
    F, T = structured_spectrogram.shape
    result = spnmf(
        structured_spectrogram, n_harmonic=4, n_percussive=6, n_iter=20, random_state=0
    )
    assert result.W_h.shape == (F, 4)
    assert result.W_p.shape == (F, 6)
    assert result.H_p.shape == (6, T)
    assert result.harmonic.shape == (F, T)
    assert result.percussive.shape == (F, T)
    np.testing.assert_allclose(
        result.reconstruction, result.harmonic + result.percussive
    )
    assert np.all(result.harmonic >= 0)
    assert np.all(result.percussive >= 0)


def test_same_seed_gives_same_result(structured_spectrogram):
    kwargs = dict(n_harmonic=3, n_percussive=4, n_iter=25, random_state=7)
    a = spnmf(structured_spectrogram, **kwargs)
    b = spnmf(structured_spectrogram, **kwargs)
    np.testing.assert_array_equal(a.W_h, b.W_h)
    np.testing.assert_array_equal(a.H_p, b.H_p)


def test_tolerance_stops_early(structured_spectrogram):
    result = spnmf(
        structured_spectrogram, n_harmonic=3, n_percussive=4,
        n_iter=5000, tol=1e-4, random_state=0,
    )
    assert result.converged
    assert result.n_iter < 5000
    assert len(result.cost) == result.n_iter


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (dict(n_harmonic=0), "n_harmonic"),
        (dict(n_percussive=0), "n_percussive"),
    ],
)
def test_invalid_ranks_are_rejected(structured_spectrogram, kwargs, message):
    with pytest.raises(ValueError, match=message):
        spnmf(structured_spectrogram, **kwargs)


def test_negative_input_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        spnmf(-np.ones((5, 5)))


def test_non_2d_input_is_rejected():
    with pytest.raises(ValueError, match="2-D"):
        spnmf(np.ones(10))


def test_mismatched_dictionary_is_rejected(structured_spectrogram):
    with pytest.raises(ValueError, match="frequency bins"):
        spnmf(structured_spectrogram, W_p=np.ones((5, 3)))


# --- semi-supervised NMF ---------------------------------------------------


@pytest.mark.parametrize("name,beta", BETAS)
def test_semi_supervised_cost_decreases(name, beta, structured_spectrogram):
    from spnmf.core import semi_supervised_nmf

    rng = np.random.default_rng(0)
    W_fixed = rng.random((structured_spectrogram.shape[0], 5)) + 0.1
    result = semi_supervised_nmf(
        structured_spectrogram, W_fixed, n_free=4, divergence=name,
        n_iter=100, tol=None, random_state=1,
    )
    assert result.beta == beta
    assert np.all(np.diff(result.cost) <= 1e-9 * abs(result.cost[0]))
    assert result.cost[-1] < 0.5 * result.cost[0]


def test_semi_supervised_never_touches_the_dictionary(structured_spectrogram):
    from spnmf.core import semi_supervised_nmf

    rng = np.random.default_rng(1)
    W_fixed = rng.random((structured_spectrogram.shape[0], 6)) + 0.1
    original = W_fixed.copy()
    result = semi_supervised_nmf(
        structured_spectrogram, W_fixed, n_free=3, n_iter=40, random_state=0
    )
    np.testing.assert_array_equal(result.W_fixed, original)
    np.testing.assert_array_equal(W_fixed, original)


def test_semi_supervised_shapes_and_parts(structured_spectrogram):
    from spnmf.core import semi_supervised_nmf

    F, T = structured_spectrogram.shape
    W_fixed = np.abs(np.random.default_rng(2).random((F, 5))) + 0.1
    result = semi_supervised_nmf(
        structured_spectrogram, W_fixed, n_free=4, n_iter=25, random_state=0
    )
    assert result.W_free.shape == (F, 4)
    assert result.H_free.shape == (4, T)
    assert result.H_fixed.shape == (5, T)
    assert result.free.shape == (F, T)
    np.testing.assert_allclose(result.reconstruction, result.free + result.fixed)
    assert np.all(result.free >= 0) and np.all(result.fixed >= 0)


@pytest.mark.parametrize(
    "W_fixed,message",
    [
        (np.ones((3, 2)), "W_fixed must be"),
        (-np.ones((48, 2)), "non-negative"),
    ],
)
def test_semi_supervised_rejects_bad_dictionary(structured_spectrogram, W_fixed, message):
    from spnmf.core import semi_supervised_nmf

    with pytest.raises(ValueError, match=message):
        semi_supervised_nmf(structured_spectrogram, W_fixed, n_free=2)


def test_semi_supervised_rejects_bad_rank(structured_spectrogram):
    from spnmf.core import semi_supervised_nmf

    W_fixed = np.ones((structured_spectrogram.shape[0], 3))
    with pytest.raises(ValueError, match="n_free"):
        semi_supervised_nmf(structured_spectrogram, W_fixed, n_free=0)
