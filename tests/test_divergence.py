import numpy as np
import pytest

from spnmf.divergence import (
    as_beta,
    beta_divergence,
    gradient_split,
    plus_as_array,
)

BETAS = [2.0, 1.0, 0.0]


def test_named_divergences_resolve_to_expected_betas():
    assert as_beta("euclidean") == 2.0
    assert as_beta("kl") == 1.0
    assert as_beta("is") == 0.0
    assert as_beta("IS") == 0.0
    assert as_beta(1.5) == 1.5


def test_unknown_divergence_is_rejected():
    with pytest.raises(ValueError, match="unknown divergence"):
        as_beta("cosine")


@pytest.mark.parametrize("beta", BETAS + [1.5, 0.5])
def test_divergence_vanishes_at_the_optimum(beta):
    rng = np.random.default_rng(0)
    V = rng.random((20, 30)) + 0.5
    assert beta_divergence(V, V, beta) == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("beta", BETAS + [1.5, 0.5])
def test_divergence_is_non_negative(beta):
    rng = np.random.default_rng(1)
    V = rng.random((20, 30)) + 0.5
    V_hat = rng.random((20, 30)) + 0.5
    assert beta_divergence(V, V_hat, beta) > 0.0


@pytest.mark.parametrize("beta", BETAS)
def test_gradient_split_matches_numerical_derivative(beta):
    """`plus - minus` must be the true dD/dV_hat, or every update is wrong."""
    rng = np.random.default_rng(2)
    V = rng.random((6, 5)) + 0.5
    V_hat = rng.random((6, 5)) + 0.5

    minus, plus = gradient_split(V, V_hat, beta)
    analytic = plus_as_array(plus, V.shape) - minus

    step = 1e-6
    numerical = np.empty_like(V_hat)
    for i in range(V_hat.shape[0]):
        for j in range(V_hat.shape[1]):
            up, down = V_hat.copy(), V_hat.copy()
            up[i, j] += step
            down[i, j] -= step
            numerical[i, j] = (
                beta_divergence(V, up, beta) - beta_divergence(V, down, beta)
            ) / (2 * step)

    np.testing.assert_allclose(analytic, numerical, rtol=1e-5, atol=1e-6)


def test_plus_is_all_ones_for_kl():
    V = np.full((3, 4), 2.0)
    _, plus = gradient_split(V, V, 1.0)
    assert plus is None
    np.testing.assert_array_equal(plus_as_array(plus, V.shape), np.ones((3, 4)))
