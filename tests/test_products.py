import numpy as np
import pytest

from autocall_pricer import Autocall, BarrierOption, WorstOfAutocall, WorstOfOption

TIMES = [0.25, 0.5, 0.75, 1.0]
SPOT = np.array([100.0])


def path(perfs, s0=100.0):
    """One path of one asset, built from performances."""
    return (s0 * np.asarray(perfs, dtype=float))[None, :, None]


def test_autocall_called_with_memory():
    p = Autocall(TIMES, autocall_barrier=1.0, coupon=0.05, memory=True)
    cf = p.cashflows(path([0.9, 1.05, 1.2, 1.3]), SPOT)
    np.testing.assert_allclose(cf[0], [0.0, 1.10, 0.0, 0.0])


def test_autocall_called_without_memory():
    p = Autocall(TIMES, autocall_barrier=1.0, coupon=0.05, memory=False)
    cf = p.cashflows(path([0.9, 1.05, 1.2, 1.3]), SPOT)
    np.testing.assert_allclose(cf[0], [0.0, 1.05, 0.0, 0.0])


@pytest.mark.parametrize("final, redemption", [(0.65, 1.0), (0.6, 1.0), (0.5, 0.5)])
def test_protection_barrier_at_maturity(final, redemption):
    p = Autocall(TIMES, autocall_barrier=1.0, coupon=0.05, protection_barrier=0.6)
    cf = p.cashflows(path([0.9, 0.8, 0.7, final]), SPOT)
    np.testing.assert_allclose(cf[0], [0.0, 0.0, 0.0, redemption])


def test_protection_is_european_only():
    # an intermediate fall below the protection barrier does not matter
    p = Autocall(TIMES, autocall_barrier=1.0, coupon=0.0, protection_barrier=0.6)
    cf = p.cashflows(path([0.3, 0.4, 0.5, 0.7]), SPOT)
    np.testing.assert_allclose(cf[0], [0.0, 0.0, 0.0, 1.0])


def test_maturity_coupon_with_memory():
    p = Autocall(TIMES, autocall_barrier=1.0, coupon=0.05, memory=True)
    cf = p.cashflows(path([0.9, 0.8, 0.7, 1.0]), SPOT)
    np.testing.assert_allclose(cf[0], [0.0, 0.0, 0.0, 1.0 + 4 * 0.05])


def test_phoenix_coupon_barrier_and_memory():
    perfs = [0.8, 0.6, 0.75, 0.9]
    p = Autocall(TIMES, autocall_barrier=1.0, coupon=0.05, coupon_barrier=0.7, memory=True)
    np.testing.assert_allclose(p.cashflows(path(perfs), SPOT)[0], [0.05, 0.0, 0.10, 1.05])
    p = Autocall(TIMES, autocall_barrier=1.0, coupon=0.05, coupon_barrier=0.7, memory=False)
    np.testing.assert_allclose(p.cashflows(path(perfs), SPOT)[0], [0.05, 0.0, 0.05, 1.05])


def test_step_down_and_non_call_period():
    p = Autocall(TIMES, autocall_barrier=[1.0, 0.95, 0.9, 0.85], coupon=0.05, first_call_index=1)
    cf = p.cashflows(path([1.1, 0.96, 1.0, 1.0]), SPOT)
    # date 0: coupon paid but no early redemption; date 1: called at the 95% barrier
    np.testing.assert_allclose(cf[0], [0.05, 1.05, 0.0, 0.0])


def test_initial_levels_override_spot():
    p = Autocall(TIMES, autocall_barrier=1.0, coupon=0.05, initial_levels=[80.0])
    cf = p.cashflows(path([0.9, 0.9, 0.9, 0.9]), SPOT)  # 90 / 80 > 1: called at once
    np.testing.assert_allclose(cf[0], [1.05, 0.0, 0.0, 0.0])
    assert p.fixed_at(np.array([123.0])) is p


def test_fixed_at_freezes_initial_levels():
    p = Autocall(TIMES).fixed_at(np.array([100.0]))
    np.testing.assert_allclose(p.initial_levels, [100.0])


def test_worst_of_uses_minimum_performance():
    p = WorstOfAutocall(TIMES, autocall_barrier=1.0, coupon=0.05, protection_barrier=0.6, assets=(0, 1))
    spots = np.array([100.0, 50.0])
    a = 100.0 * np.array([1.2, 1.2, 1.2, 1.2])
    b = 50.0 * np.array([0.9, 0.8, 0.7, 0.55])
    paths = np.stack([a, b], axis=-1)[None]
    np.testing.assert_allclose(p.performance(paths, spots)[0], [0.9, 0.8, 0.7, 0.55])
    np.testing.assert_allclose(p.cashflows(paths, spots)[0], [0.0, 0.0, 0.0, 0.55])


def test_vectorised_paths_are_independent():
    p = Autocall(TIMES, autocall_barrier=1.0, coupon=0.05)
    paths = np.concatenate([path([1.1, 1, 1, 1]), path([0.5, 0.5, 0.5, 0.5])])
    np.testing.assert_allclose(p.cashflows(paths, SPOT), [[1.05, 0, 0, 0], [0, 0, 0, 0.5]])


def test_redemption_times():
    p = Autocall(TIMES, autocall_barrier=1.0)
    paths = np.concatenate([path([0.9, 1.1, 1, 1]), path([0.5, 0.5, 0.5, 0.5])])
    np.testing.assert_allclose(p.redemption_times(paths, SPOT), [0.5, 1.0])


@pytest.mark.parametrize("n", [1, 6])
def test_worst_of_requires_two_to_five_assets(n):
    with pytest.raises(ValueError, match="between 2 and 5"):
        WorstOfAutocall(TIMES, assets=tuple(range(n)))


def test_autocall_is_single_asset():
    with pytest.raises(ValueError):
        Autocall(TIMES, assets=(0, 1))


@pytest.mark.parametrize("times", [[0.5, 0.25], [0.0, 1.0], []])
def test_rejects_bad_observation_times(times):
    with pytest.raises(ValueError):
        Autocall(times)


def test_barrier_knock_out_plus_knock_in_is_vanilla():
    rng = np.random.default_rng(0)
    paths = 100 * np.exp(np.cumsum(0.02 * rng.standard_normal((1000, 12)), axis=1))[:, :, None]
    kw = dict(strike=100.0, barrier=95.0, maturity=1.0, kind="call", n_monitoring=12)
    out = BarrierOption(barrier_kind="down-and-out", **kw).cashflows(paths, SPOT)
    inn = BarrierOption(barrier_kind="down-and-in", **kw).cashflows(paths, SPOT)
    np.testing.assert_allclose(out[:, -1] + inn[:, -1], np.maximum(paths[:, -1, 0] - 100.0, 0.0))


def test_worst_of_option_payoff():
    paths = np.array([[[110.0, 95.0]]])
    cf = WorstOfOption(90.0, 1.0, (0, 1), "call").cashflows(paths, np.array([100.0, 100.0]))
    np.testing.assert_allclose(cf, [[5.0]])
