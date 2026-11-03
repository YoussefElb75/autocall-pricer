import numpy as np
import pytest

from autocall_pricer import BlackScholesModel, constant_correlation


def test_rejects_non_positive_definite_correlation():
    corr = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    with pytest.raises(ValueError, match="positive semi-definite"):
        BlackScholesModel(spots=[1, 1, 1], vols=0.2, rate=0.0, correlation=corr)


@pytest.mark.parametrize(
    "corr, msg",
    [
        (np.array([[1.0, 0.5], [0.4, 1.0]]), "symmetric"),
        (np.array([[2.0, 0.0], [0.0, 1.0]]), "unit diagonal"),
        (np.eye(3), "shape"),
    ],
)
def test_rejects_invalid_correlation(corr, msg):
    with pytest.raises(ValueError, match=msg):
        BlackScholesModel(spots=[1, 1], vols=0.2, rate=0.0, correlation=corr)


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        BlackScholesModel(spots=[0.0], vols=0.2, rate=0.0)
    with pytest.raises(ValueError):
        BlackScholesModel(spots=[1.0], vols=-0.2, rate=0.0)


def test_cholesky_reconstructs_correlation():
    corr = constant_correlation(4, 0.3)
    m = BlackScholesModel(spots=np.ones(4), vols=0.2, rate=0.0, correlation=corr)
    np.testing.assert_allclose(m.cholesky @ m.cholesky.T, corr)


def test_simulate_shape_and_time_validation():
    m = BlackScholesModel(spots=[100, 50], vols=[0.2, 0.3], rate=0.01)
    z = np.zeros((10, 3, 2))
    assert m.simulate([0.5, 1.0, 2.0], z).shape == (10, 3, 2)
    with pytest.raises(ValueError):
        m.simulate([1.0, 0.5, 2.0], z)
    with pytest.raises(ValueError):
        m.simulate([0.5, 1.0], z)


def test_zero_vol_is_deterministic_forward():
    m = BlackScholesModel(spots=[100.0], vols=[0.0], rate=0.05, dividends=[0.02])
    z = np.random.default_rng(0).standard_normal((5, 2, 1))
    paths = m.simulate([1.0, 2.0], z)
    expected = [m.forwards(1.0)[0], m.forwards(2.0)[0]]
    np.testing.assert_allclose(paths[:, :, 0], np.tile(expected, (5, 1)))


def test_martingale_and_correlation():
    rho = 0.6
    m = BlackScholesModel(
        spots=[100.0, 80.0],
        vols=[0.2, 0.35],
        rate=0.03,
        dividends=[0.01, 0.0],
        correlation=constant_correlation(2, rho),
    )
    n = 400_000
    z = np.random.default_rng(1).standard_normal((n, 2, 2))
    paths = m.simulate([1.0, 2.0], z)
    st = paths[:, -1, :]
    se = st.std(axis=0) / np.sqrt(n)
    assert np.all(np.abs(st.mean(axis=0) - m.forwards(2.0)) < 4 * se)
    log_ret = np.log(paths[:, 1, :] / paths[:, 0, :])
    assert abs(np.corrcoef(log_ret.T)[0, 1] - rho) < 0.01
    np.testing.assert_allclose(log_ret.std(axis=0), m.vols, rtol=0.01)
