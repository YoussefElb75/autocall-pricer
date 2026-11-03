"""No-arbitrage identities, Monte Carlo convergence and exact autocall limit cases."""

import numpy as np
import pytest

from autocall_pricer import (
    Autocall,
    BlackScholesModel,
    EuropeanOption,
    MonteCarloEngine,
    WorstOfAutocall,
    constant_correlation,
)
from autocall_pricer.analytic import bs_digital, bs_price

S, R, Q, VOL = 100.0, 0.03, 0.01, 0.2
QUARTERLY = [0.25 * k for k in range(1, 13)]  # 3-year quarterly autocall


@pytest.fixture
def model():
    return BlackScholesModel(spots=[S], vols=[VOL], rate=R, dividends=[Q])


# --- put-call parity and forward --------------------------------------------------------
@pytest.mark.parametrize("strike, t", [(80.0, 0.5), (100.0, 1.0), (130.0, 3.0)])
@pytest.mark.parametrize("antithetic", [True, False])
def test_put_call_parity(model, strike, t, antithetic):
    engine = MonteCarloEngine(n_paths=100_000, seed=17, antithetic=antithetic)
    call = engine.price(model, EuropeanOption(strike, t, "call"))
    put = engine.price(model, EuropeanOption(strike, t, "put"))
    forward = engine.price(model, EuropeanOption(0.0, t))  # zero-strike call = PV of S_T
    # pathwise identity (C - P = S_T - K on every path, same random numbers)
    assert call.price - put.price == pytest.approx(forward.price - strike * np.exp(-R * t), abs=1e-9)
    # and the forward is unbiased
    exact = S * np.exp(-Q * t) - strike * np.exp(-R * t)
    assert call.price - put.price == pytest.approx(exact, abs=4 * forward.std_error)


def test_model_forward(model):
    assert model.forwards(2.0)[0] == pytest.approx(S * np.exp((R - Q) * 2.0), rel=1e-14)


# --- convergence in 1/sqrt(N) ---------------------------------------------------------------
@pytest.mark.slow
def test_monte_carlo_error_decreases_as_one_over_sqrt_n(model):
    exact = bs_price(S, 100.0, 1.0, R, Q, VOL)
    sizes = np.array([1_000, 4_000, 16_000, 64_000])
    rmse, mean_se = [], []
    for n in sizes:
        runs = [MonteCarloEngine(n_paths=int(n), seed=1_000 + k, antithetic=False).price(
            model, EuropeanOption(100.0, 1.0)) for k in range(30)]
        rmse.append(np.sqrt(np.mean([(r.price - exact) ** 2 for r in runs])))
        mean_se.append(np.mean([r.std_error for r in runs]))
    slope = np.polyfit(np.log(sizes), np.log(rmse), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.1)
    # the reported standard error halves when N is multiplied by 4 ...
    np.testing.assert_allclose(np.array(mean_se[1:]) / mean_se[:-1], 0.5, rtol=0.05)
    # ... and matches the observed error
    np.testing.assert_allclose(rmse, mean_se, rtol=0.3)


# --- autocall limit cases with closed forms ---------------------------------------------------
def test_never_called_autocall_is_zero_coupon_minus_european_down_and_in_put(model):
    """Autocall and coupon barriers out of reach, protection at 60%: with x = S_T / S_0,
    payoff = 1 - (1 - x) 1{x < B} = 1 - (B - x)^+ - (1 - B) 1{x < B}."""
    b, t = 0.6, QUARTERLY[-1]
    product = Autocall(QUARTERLY, autocall_barrier=np.inf, coupon=0.02, protection_barrier=b)
    res = MonteCarloEngine(n_paths=200_000, seed=7).price(model, product)
    exact = (np.exp(-R * t)
             - bs_price(S, b * S, t, R, Q, VOL, "put") / S
             - (1 - b) * bs_digital(S, b * S, t, R, Q, VOL, "put"))
    assert res.price == pytest.approx(exact, abs=4 * res.std_error)


def test_never_called_phoenix_without_memory_adds_a_strip_of_digitals(model):
    b, cb, c = 0.6, 0.7, 0.02
    product = Autocall(QUARTERLY, autocall_barrier=np.inf, coupon=c, coupon_barrier=cb,
                       memory=False, protection_barrier=b)
    res = MonteCarloEngine(n_paths=200_000, seed=7).price(model, product)
    t = QUARTERLY[-1]
    exact = (np.exp(-R * t)
             - bs_price(S, b * S, t, R, Q, VOL, "put") / S
             - (1 - b) * bs_digital(S, b * S, t, R, Q, VOL, "put")
             + sum(c * bs_digital(S, cb * S, ti, R, Q, VOL, "call") for ti in QUARTERLY))
    assert res.price == pytest.approx(exact, abs=4 * res.std_error)


def test_called_at_first_date_with_coupon_barrier_above_call_barrier(model):
    product = Autocall(QUARTERLY, autocall_barrier=0.0, coupon=0.02, coupon_barrier=1.0)
    res = MonteCarloEngine(n_paths=50_000, seed=7).price(model, product)
    t1 = QUARTERLY[0]
    exact = np.exp(-R * t1) + 0.02 * bs_digital(S, S, t1, R, Q, VOL, "call")
    assert res.price == pytest.approx(exact, abs=4 * res.std_error + 1e-12)


@pytest.mark.parametrize("n_assets", [2, 5])
def test_worst_of_with_perfect_correlation_is_single_asset(n_assets):
    spots = 100.0 * np.arange(1, n_assets + 1)  # different levels, identical dynamics
    m = BlackScholesModel(spots=spots, vols=VOL, rate=R, dividends=Q,
                          correlation=constant_correlation(n_assets, 1.0))
    engine = MonteCarloEngine(n_paths=50_000, seed=3)
    kw = dict(coupon=0.02, coupon_barrier=0.7, protection_barrier=0.6)
    wo = engine.price(m, WorstOfAutocall(QUARTERLY, assets=tuple(range(n_assets)), **kw))
    single = engine.price(m, Autocall(QUARTERLY, assets=(0,), **kw))
    assert wo.price == pytest.approx(single.price, abs=1e-9)
