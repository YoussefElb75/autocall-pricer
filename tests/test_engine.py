import numpy as np
import pytest
from autocall_pricer.analytic import bs_price

from autocall_pricer import (
    Autocall,
    BlackScholesModel,
    EuropeanOption,
    MonteCarloEngine,
    WorstOfAutocall,
    constant_correlation,
)

R, Q, VOL, T = 0.03, 0.01, 0.2, 1.0
QUARTERLY = [0.25 * k for k in range(1, 13)]  # 3-year quarterly autocall


@pytest.fixture
def model():
    return BlackScholesModel(spots=[100.0], vols=[VOL], rate=R, dividends=[Q])


def test_reproducible_with_fixed_seed(model):
    product = Autocall(QUARTERLY, coupon=0.02)
    a = MonteCarloEngine(n_paths=20_000, seed=7).price(model, product)
    b = MonteCarloEngine(n_paths=20_000, seed=7).price(model, product)
    c = MonteCarloEngine(n_paths=20_000, seed=8).price(model, product)
    assert a.price == b.price and a.std_error == b.std_error
    assert a.price != c.price


def test_batching_does_not_change_the_price(model):
    product = Autocall(QUARTERLY, coupon=0.02)
    a = MonteCarloEngine(n_paths=20_000, seed=3, batch_size=20_000).price(model, product)
    b = MonteCarloEngine(n_paths=20_000, seed=3, batch_size=3_000).price(model, product)
    assert a.price == pytest.approx(b.price, abs=1e-12)


def test_ci_contains_black_scholes_price(model):
    res = MonteCarloEngine(n_paths=200_000, seed=11).price(model, EuropeanOption(100.0, T))
    lo, hi = res.ci
    assert lo <= bs_price(100.0, 100.0, T, R, Q, VOL) <= hi
    assert hi - lo == pytest.approx(2 * 1.959964 * res.std_error, rel=1e-5)


def test_antithetic_reduces_variance(model):
    product = EuropeanOption(100.0, T)
    anti = MonteCarloEngine(n_paths=100_000, seed=5, antithetic=True).price(model, product)
    plain = MonteCarloEngine(n_paths=100_000, seed=5, antithetic=False).price(model, product)
    assert anti.std_error < 0.8 * plain.std_error


def test_zero_strike_call_is_discounted_forward(model):
    res = MonteCarloEngine(n_paths=100_000, seed=42).price(model, EuropeanOption(0.0, T))
    assert res.price == pytest.approx(100.0 * np.exp(-Q * T), abs=3 * res.std_error)


def test_engine_input_validation(model):
    with pytest.raises(ValueError):
        MonteCarloEngine(n_paths=11, antithetic=True)
    with pytest.raises(ValueError):
        MonteCarloEngine(seed=42).price(model, EuropeanOption(100.0, T, asset=1))


# --- autocall limit cases ---------------------------------------------------
def test_autocall_never_called_never_lossy_is_zero_coupon(model):
    product = Autocall(QUARTERLY, autocall_barrier=np.inf, coupon=0.0, protection_barrier=0.0)
    res = MonteCarloEngine(n_paths=10_000, seed=42).price(model, product)
    assert res.price == pytest.approx(np.exp(-R * 3.0), abs=1e-12)
    assert res.std_error == pytest.approx(0.0, abs=1e-12)


def test_autocall_always_called_at_first_date(model):
    product = Autocall(QUARTERLY, autocall_barrier=0.0, coupon=0.02)
    res = MonteCarloEngine(n_paths=10_000, seed=42).price(model, product)
    assert res.price == pytest.approx(1.02 * np.exp(-R * 0.25), abs=1e-12)


def test_autocall_with_full_protection_is_never_below_zero_coupon(model):
    product = Autocall(QUARTERLY, coupon=0.0, protection_barrier=0.0)
    res = MonteCarloEngine(n_paths=50_000, seed=42).price(model, product)
    # with r > 0, early redemption at par is worth more than par at maturity
    assert res.price > np.exp(-R * 3.0)
    assert res.price < 1.0


def test_autocall_price_is_monotonic_in_its_parameters(model):
    engine = MonteCarloEngine(n_paths=50_000, seed=1)

    def price(**kw):
        base = dict(autocall_barrier=1.0, coupon=0.02, protection_barrier=0.6)
        return engine.price(model, Autocall(QUARTERLY, **{**base, **kw})).price

    assert price(coupon=0.03) > price(coupon=0.02)
    assert price(protection_barrier=0.5) > price(protection_barrier=0.7)
    p_mem = engine.price(model, Autocall(QUARTERLY, coupon=0.02, coupon_barrier=0.7, memory=True))
    p_nomem = engine.price(model, Autocall(QUARTERLY, coupon=0.02, coupon_barrier=0.7, memory=False))
    assert p_mem.price > p_nomem.price


def test_worst_of_is_cheaper_than_each_single_asset_autocall():
    corr = constant_correlation(3, 0.5)
    m = BlackScholesModel(spots=[100.0, 50.0, 20.0], vols=[0.2, 0.25, 0.3], rate=R, correlation=corr)
    engine = MonteCarloEngine(n_paths=50_000, seed=42)
    kw = dict(coupon=0.02, protection_barrier=0.6)
    wo = engine.price(m, WorstOfAutocall(QUARTERLY, assets=(0, 1, 2), **kw)).price
    singles = [engine.price(m, Autocall(QUARTERLY, assets=(i,), **kw)).price for i in range(3)]
    assert wo < min(singles)


def test_worst_of_identical_assets_full_correlation_equals_single():
    rho = 1.0  # positive semi-definite: accepted since the eigen-decomposition fallback
    m = BlackScholesModel(spots=[100.0, 100.0], vols=VOL, rate=R, dividends=Q,
                          correlation=constant_correlation(2, rho))
    engine = MonteCarloEngine(n_paths=50_000, seed=42)
    kw = dict(coupon=0.02, protection_barrier=0.6)
    wo = engine.price(m, WorstOfAutocall(QUARTERLY, assets=(0, 1), **kw))
    single = engine.price(m, Autocall(QUARTERLY, assets=(0,), **kw))
    assert wo.price == pytest.approx(single.price, abs=1e-9)


@pytest.mark.parametrize("n_assets", [2, 3, 4, 5])
def test_worst_of_decreases_with_number_of_assets(n_assets):
    m = BlackScholesModel(spots=np.full(5, 100.0), vols=0.25, rate=R,
                          correlation=constant_correlation(5, 0.5))
    engine = MonteCarloEngine(n_paths=40_000, seed=42)
    kw = dict(coupon=0.02, protection_barrier=0.6)
    p_n = engine.price(m, WorstOfAutocall(QUARTERLY, assets=tuple(range(n_assets)), **kw)).price
    if n_assets == 2:
        p_prev = engine.price(m, Autocall(QUARTERLY, assets=(0,), **kw)).price
    else:
        p_prev = engine.price(m, WorstOfAutocall(QUARTERLY, assets=tuple(range(n_assets - 1)), **kw)).price
    assert p_n < p_prev
