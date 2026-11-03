import numpy as np
import pytest
from autocall_pricer.analytic import bs_delta, bs_gamma, bs_vega

from autocall_pricer import (
    Autocall,
    BlackScholesModel,
    EuropeanOption,
    MonteCarloEngine,
    WorstOfAutocall,
    compute_greeks,
    constant_correlation,
    correlation_sensitivity,
)

S, R, Q, VOL, T = 100.0, 0.03, 0.01, 0.2, 1.0
QUARTERLY = [0.25 * k for k in range(1, 13)]
ENGINE = MonteCarloEngine(n_paths=200_000, seed=123)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_vanilla_greeks_match_black_scholes(kind):
    model = BlackScholesModel(spots=[S], vols=[VOL], rate=R, dividends=[Q])
    g = compute_greeks(model, EuropeanOption(100.0, T, kind), ENGINE)
    assert g.delta[0] == pytest.approx(bs_delta(S, 100.0, T, R, Q, VOL, kind), abs=5e-3)
    assert g.gamma[0] == pytest.approx(bs_gamma(S, 100.0, T, R, Q, VOL), rel=0.05)
    assert g.vega[0] == pytest.approx(bs_vega(S, 100.0, T, R, Q, VOL), rel=0.02)
    assert g.correlation is None


def test_common_random_numbers_make_delta_stable():
    """Two seeds give deltas far closer than the price standard error."""
    model = BlackScholesModel(spots=[S], vols=[VOL], rate=R, dividends=[Q])
    product = EuropeanOption(100.0, T)
    d = [compute_greeks(model, product, MonteCarloEngine(n_paths=50_000, seed=s)).delta[0] for s in (1, 2)]
    assert abs(d[0] - d[1]) < 0.01


def test_autocall_greeks_signs():
    model = BlackScholesModel(spots=[S], vols=[VOL], rate=R, dividends=[Q])
    product = Autocall(QUARTERLY, coupon=0.02, protection_barrier=0.6)
    g = compute_greeks(model, product, MonteCarloEngine(n_paths=100_000, seed=42))
    assert g.delta[0] > 0  # long the underlying (short a down-and-in put)
    assert g.vega[0] < 0  # short volatility
    assert 0.0 < g.price < 1.2


def test_autocall_delta_is_not_zero_when_struck_today():
    """Initial fixings are frozen before bumping spots."""
    model = BlackScholesModel(spots=[S], vols=[VOL], rate=R)
    product = Autocall(QUARTERLY, coupon=0.02)
    assert compute_greeks(model, product, MonteCarloEngine(n_paths=20_000, seed=42)).delta[0] > 1e-4


def test_worst_of_greeks():
    corr = constant_correlation(3, 0.5)
    model = BlackScholesModel(spots=[S, S, S], vols=[0.2, 0.2, 0.2], rate=R, correlation=corr)
    product = WorstOfAutocall(QUARTERLY, assets=(0, 1, 2), coupon=0.03, protection_barrier=0.6)
    g = compute_greeks(model, product, MonteCarloEngine(n_paths=100_000, seed=42))
    assert np.all(g.delta > 0)
    assert np.all(g.vega < 0)
    # symmetric assets: symmetric deltas
    assert np.ptp(g.delta) < 0.1 * g.delta.mean()
    # higher correlation makes the worst-of less bad: the investor is short correlation dispersion
    assert g.correlation > 0


def test_unreferenced_asset_has_zero_greeks():
    model = BlackScholesModel(spots=[S, 50.0], vols=[0.2, 0.3], rate=R)
    g = compute_greeks(model, Autocall(QUARTERLY, assets=(0,)), MonteCarloEngine(n_paths=10_000, seed=42))
    assert g.delta[1] == g.gamma[1] == g.vega[1] == 0.0
    assert g.correlation is None


def test_pairwise_correlation_sensitivities_sum_to_parallel():
    corr = constant_correlation(3, 0.4)
    model = BlackScholesModel(spots=[S, S, S], vols=[0.2, 0.25, 0.3], rate=R, correlation=corr)
    product = WorstOfAutocall(QUARTERLY, assets=(0, 1, 2), coupon=0.03)
    engine = MonteCarloEngine(n_paths=100_000, seed=42)
    parallel = correlation_sensitivity(model, product, engine)
    pairs = sum(correlation_sensitivity(model, product, engine, pair=p) for p in [(0, 1), (0, 2), (1, 2)])
    assert parallel == pytest.approx(pairs, rel=0.05)


def test_correlation_bump_outside_valid_range_raises():
    model = BlackScholesModel(spots=[S, S], vols=0.2, rate=R, correlation=constant_correlation(2, 0.995))
    product = WorstOfAutocall(QUARTERLY, assets=(0, 1))
    with pytest.raises(ValueError):
        correlation_sensitivity(model, product, MonteCarloEngine(n_paths=1_000, seed=42), abs_bump=0.01)
