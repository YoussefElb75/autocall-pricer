"""Regression tests for the issues found in the code review (one section per issue).

Each test failed on the code before the corresponding fix.
"""

import warnings

import numpy as np
import pytest
from scipy.stats import norm

from autocall_pricer import (
    Autocall,
    BarrierOption,
    BlackScholesModel,
    DigitalOption,
    EuropeanOption,
    MonteCarloEngine,
    WorstOfAutocall,
    WorstOfOption,
    compute_greeks,
    constant_correlation,
    correlation_sensitivity,
)

S, R, Q, VOL = 100.0, 0.03, 0.01, 0.2
TIMES = [0.25, 0.5, 0.75, 1.0]
QUARTERLY = [0.25 * k for k in range(1, 13)]


# --- 1. inputs are copied and stored read-only ---------------------------------------
def test_model_does_not_alias_its_inputs():
    spots, vols, divs = np.array([100.0, 50.0]), np.array([0.2, 0.3]), np.array([0.01, 0.0])
    corr = constant_correlation(2, 0.5)
    m = BlackScholesModel(spots=spots, vols=vols, rate=R, dividends=divs, correlation=corr)
    spots[0], vols[0], divs[0], corr[0, 1] = -1.0, -1.0, 9.0, 0.99
    np.testing.assert_array_equal(m.spots, [100.0, 50.0])
    np.testing.assert_array_equal(m.vols, [0.2, 0.3])
    np.testing.assert_array_equal(m.dividends, [0.01, 0.0])
    np.testing.assert_allclose(m.cholesky @ m.cholesky.T, m.correlation)
    assert m.correlation[0, 1] == 0.5


def test_model_arrays_are_read_only():
    m = BlackScholesModel(spots=[100.0], vols=[0.2], rate=R)
    for arr in (m.spots, m.vols, m.dividends, m.correlation, m.cholesky):
        with pytest.raises(ValueError):
            arr[0] = 0.0
    # the bump helpers still work on copies
    assert m.with_spot(0, 101.0).spots[0] == 101.0 and m.spots[0] == 100.0


def test_product_does_not_alias_its_inputs():
    t, levels, ac = np.array(TIMES), np.array([80.0]), np.array([1.0, 0.95, 0.9, 0.85])
    p = Autocall(t, autocall_barrier=ac, initial_levels=levels)
    t[0], levels[0], ac[0] = 0.9, 1.0, 0.0
    np.testing.assert_array_equal(p.observation_times, TIMES)
    np.testing.assert_array_equal(p.initial_levels, [80.0])
    np.testing.assert_array_equal(p.autocall_barrier, [1.0, 0.95, 0.9, 0.85])
    for arr in (p.observation_times, p.autocall_barrier, p.coupon_barrier, p.initial_levels):
        with pytest.raises(ValueError):
            arr[0] = 0.0


# --- 2. positive semi-definite correlations (rho = 1) are accepted ----------------------
def test_perfect_correlation_is_accepted():
    m = BlackScholesModel(spots=[S, 50.0], vols=VOL, rate=R, correlation=constant_correlation(2, 1.0))
    np.testing.assert_allclose(m.cholesky @ m.cholesky.T, m.correlation, atol=1e-12)
    z = np.random.default_rng(0).standard_normal((1000, 3, 2))
    paths = m.simulate([0.5, 1.0, 2.0], z)
    np.testing.assert_allclose(paths[:, :, 0] / S, paths[:, :, 1] / 50.0, rtol=1e-12)


def test_singular_correlation_block_is_accepted():
    corr = np.array([[1.0, 1.0, 0.3], [1.0, 1.0, 0.3], [0.3, 0.3, 1.0]])
    m = BlackScholesModel(spots=[1, 1, 1], vols=0.2, rate=0.0, correlation=corr)
    np.testing.assert_allclose(m.cholesky @ m.cholesky.T, corr, atol=1e-12)


def test_indefinite_correlation_is_still_rejected():
    corr = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    with pytest.raises(ValueError, match="positive semi-definite"):
        BlackScholesModel(spots=[1, 1, 1], vols=0.2, rate=0.0, correlation=corr)


# --- 3. option / barrier kinds are validated at construction -----------------------------
@pytest.mark.parametrize("barrier_kind", ["dwn-and-out", "down-out", "", "DOWN-AND-OUT"])
def test_invalid_barrier_kind_is_rejected(barrier_kind):
    with pytest.raises(ValueError, match="barrier_kind"):
        BarrierOption(100.0, 90.0, 1.0, barrier_kind)


@pytest.mark.parametrize(
    "make",
    [
        lambda: EuropeanOption(100.0, 1.0, "cal"),
        lambda: DigitalOption(100.0, 1.0, "Put"),
        lambda: BarrierOption(100.0, 90.0, 1.0, "down-and-out", "pt"),
        lambda: WorstOfOption(100.0, 1.0, (0, 1), "c"),
    ],
)
def test_invalid_option_kind_is_rejected_at_construction(make):
    with pytest.raises(ValueError, match="kind"):
        make()


# --- 4. value equality and hashing --------------------------------------------------------
def test_products_compare_and_hash_by_value():
    a, b = Autocall(TIMES, coupon=0.05), Autocall(list(TIMES), coupon=0.05)
    assert a == b and hash(a) == hash(b)
    assert a != Autocall(TIMES, coupon=0.06)
    assert a != WorstOfAutocall(TIMES, coupon=0.05)
    assert len({a, b, Autocall(TIMES, coupon=0.06)}) == 2
    assert EuropeanOption(100.0, 1.0) == EuropeanOption(100.0, 1.0)


def test_models_and_greeks_compare_and_hash_by_value():
    m1 = BlackScholesModel(spots=[S], vols=[VOL], rate=R, dividends=[Q])
    m2 = BlackScholesModel(spots=np.array([S]), vols=VOL, rate=R, dividends=Q)
    assert m1 == m2 and hash(m1) == hash(m2)
    assert m1 != m1.with_vol(0, 0.25)
    e = MonteCarloEngine(n_paths=2_000, seed=1)
    g = compute_greeks(m1, EuropeanOption(100.0, 1.0), e)
    assert g == compute_greeks(m1, EuropeanOption(100.0, 1.0), e)


# --- 5. vega bump stays small relative to the volatility ----------------------------------
def _bs_vega(v):
    d1 = (np.log(S / 100.0) + (R - Q + 0.5 * v * v)) / v
    return S * np.exp(-Q) * norm.pdf(d1)


@pytest.mark.parametrize("vol, rel", [(0.01, 0.05), (0.02, 0.02)])
def test_vega_is_accurate_at_low_volatility(vol, rel):
    m = BlackScholesModel(spots=[S], vols=[vol], rate=R, dividends=[Q])
    g = compute_greeks(m, EuropeanOption(100.0, 1.0), MonteCarloEngine(n_paths=200_000, seed=123))
    assert g.vega[0] == pytest.approx(_bs_vega(vol), rel=rel)


# --- 6. input validation ----------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(first_call_index=-1),
        dict(first_call_index=4),
        dict(nominal=0.0),
        dict(nominal=-1.0),
        dict(autocall_barrier=[1.0, 0.9]),
        dict(autocall_barrier=np.nan),
        dict(autocall_barrier=-0.1),
        dict(coupon_barrier=[0.7, np.nan, 0.7, 0.7]),
        dict(protection_barrier=np.nan),
        dict(coupon=np.inf),
    ],
)
def test_autocall_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError, match=r"first_call_index|nominal|barrier|coupon"):
        Autocall(TIMES, **kwargs)


def test_first_call_index_on_last_date_is_allowed():
    p = Autocall(TIMES, coupon=0.05, first_call_index=3)
    cf = p.cashflows((100 * np.array([1.2, 1.2, 1.2, 1.2]))[None, :, None], np.array([S]))
    np.testing.assert_allclose(cf[0], [0.05, 0.05, 0.05, 1.05])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(spots=[np.nan], vols=[0.2], rate=R),
        dict(spots=[np.inf], vols=[0.2], rate=R),
        dict(spots=[100.0], vols=[np.nan], rate=R),
        dict(spots=[100.0], vols=[0.2], rate=np.nan),
        dict(spots=[100.0], vols=[0.2], rate=R, dividends=[np.nan]),
        dict(spots=[[100.0, 100.0]], vols=0.2, rate=R),
        dict(spots=[], vols=0.2, rate=R),
        dict(spots=[100.0, 100.0], vols=0.2, rate=R, correlation=[[1.0, np.nan], [np.nan, 1.0]]),
    ],
)
def test_model_rejects_non_finite_or_malformed_inputs(kwargs):
    with pytest.raises(ValueError, match=r"finite|1-D|at least one"):
        BlackScholesModel(**kwargs)


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        (dict(confidence=1.5), ValueError),
        (dict(confidence=0.0), ValueError),
        (dict(n_paths=1e5), TypeError),
        (dict(n_paths=True), TypeError),
        (dict(batch_size=5e4), TypeError),
        (dict(seed=1.5), TypeError),
        (dict(seed=-1), ValueError),
    ],
)
def test_engine_rejects_invalid_settings(kwargs, exc):
    with pytest.raises(exc):
        MonteCarloEngine(**kwargs)


@pytest.mark.parametrize("assets", [(), (0, 0)])
def test_worst_of_option_requires_distinct_underlyings(assets):
    with pytest.raises(ValueError):
        WorstOfOption(100.0, 1.0, assets)


@pytest.mark.parametrize(
    "make",
    [
        lambda: EuropeanOption(100.0, 0.0),
        lambda: EuropeanOption(-1.0, 1.0),
        lambda: DigitalOption(100.0, -1.0),
        lambda: BarrierOption(100.0, 90.0, 1.0, n_monitoring=0),
        lambda: BarrierOption(100.0, -90.0, 1.0),
    ],
)
def test_simple_products_reject_invalid_parameters(make):
    with pytest.raises(ValueError):
        make()


@pytest.mark.parametrize("pair", [(1, 2), (0, 7), (0, -1)])
def test_correlation_pair_must_belong_to_the_product(pair):
    model = BlackScholesModel(spots=[S, S, S], vols=VOL, rate=R, correlation=constant_correlation(3, 0.3))
    with pytest.raises(ValueError, match="pair"):
        correlation_sensitivity(model, WorstOfAutocall(QUARTERLY, assets=(0, 1)),
                                MonteCarloEngine(n_paths=1_000, seed=0), pair=pair)


# --- 7. typing --------------------------------------------------------------------------
def test_normalised_attributes_are_typed_as_arrays():
    import typing

    from autocall_pricer import products

    hints = typing.get_type_hints(BlackScholesModel)
    for name in ("spots", "vols", "dividends", "correlation"):
        assert hints[name] is np.ndarray, name
    hints = typing.get_type_hints(WorstOfAutocall)
    for name in ("observation_times", "autocall_barrier", "coupon_barrier"):
        assert hints[name] is np.ndarray, name
    for cls in (products.EuropeanOption, products.DigitalOption, products.BarrierOption,
                products.WorstOfOption, products.WorstOfAutocall):
        assert set(typing.get_type_hints(cls.cashflows)) == {"paths", "spots", "return"}, cls


# --- 8. seeds ---------------------------------------------------------------------------
def test_unseeded_engines_are_independent_but_reproducible():
    model = BlackScholesModel(spots=[S], vols=[VOL], rate=R, dividends=[Q])
    product = EuropeanOption(100.0, 1.0)
    e1, e2 = MonteCarloEngine(n_paths=10_000), MonteCarloEngine(n_paths=10_000)
    assert e1.seed != e2.seed
    assert e1.price(model, product).price != e2.price(model, product).price
    # one engine reuses its seed on every call (common random numbers for the greeks) ...
    assert e1.price(model, product).price == e1.price(model, product).price
    # ... and the drawn seed reproduces the run
    replay = MonteCarloEngine(n_paths=10_000, seed=e1.seed)
    assert replay.price(model, product).price == e1.price(model, product).price


# --- 9. QuantLib validation --------------------------------------------------------------
def test_validation_restores_quantlib_evaluation_date():
    ql = pytest.importorskip("QuantLib")
    from autocall_pricer.validation import _QLMarket

    before = ql.Date(15, ql.June, 2031)
    ql.Settings.instance().evaluationDate = before
    try:
        mkt = _QLMarket()
        price = mkt.vanilla(100.0, 1.0, "call")
        assert ql.Settings.instance().evaluationDate == before
        assert price == pytest.approx(8.82732, abs=1e-5)
    finally:
        ql.Settings.instance().evaluationDate = ql.Date()


def test_validation_barrier_helper_has_no_unused_parameter():
    pytest.importorskip("QuantLib")
    import inspect

    from autocall_pricer.validation import _QLMarket

    assert list(inspect.signature(_QLMarket.barrier).parameters) == [
        "self", "strike", "barrier", "t", "barrier_kind", "kind"]


def test_validation_tolerances_are_parameters():
    pytest.importorskip("QuantLib")
    from autocall_pricer.validation import run_validation, validation_cases

    n_cases = len(validation_cases())
    rows = run_validation(n_paths=2_000, seed=1, family_alpha=0.01, barrier_rel_tol=0.005)
    expected = norm.ppf(1 - 0.01 / (2 * n_cases))
    assert all(r.n_sigma == pytest.approx(expected) for r in rows)
    barrier_rows = [r for r in rows if "obs." in r.case]
    assert barrier_rows and all(r.model_tolerance == pytest.approx(0.005 * r.ql_price) for r in barrier_rows)


# --- 10. readable QuantLib test ids --------------------------------------------------------
def test_validation_cases_have_unique_readable_ids():
    pytest.importorskip("QuantLib")
    from autocall_pricer.validation import validation_cases

    ids = [c.id for c in validation_cases()]
    assert len(set(ids)) == len(ids) and all(ids)
    assert "european-call-K100" in ids


def test_closed_forms_live_in_the_package():
    from autocall_pricer import analytic

    assert analytic.bs_price(S, 100.0, 1.0, R, Q, VOL) == pytest.approx(8.82732, abs=1e-5)
    assert analytic.bs_price(S, 0.0, 1.0, R, Q, VOL) == pytest.approx(S * np.exp(-Q))


# --- 11. misc ----------------------------------------------------------------------------
def test_redemption_times_replaces_expected_life():
    p = Autocall(TIMES, autocall_barrier=1.0)
    paths = (100 * np.array([[0.9, 1.1, 1, 1], [0.5, 0.5, 0.5, 0.5]]))[:, :, None]
    np.testing.assert_allclose(p.redemption_times(paths, np.array([S])), [0.5, 1.0])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        np.testing.assert_allclose(p.expected_life(paths, np.array([S])), [0.5, 1.0])
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
