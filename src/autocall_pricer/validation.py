"""Validation of the Monte Carlo engine against QuantLib.

Requires the optional dependency ``QuantLib`` (``pip install autocall-pricer[validation]``).

Each case is priced by the Monte Carlo engine and by a QuantLib engine (analytic
whenever possible). A case passes when

    |MC - QuantLib| <= z * SE + model_tolerance   (+ 1e-10 for round-off)

where SE is the Monte Carlo standard error and ``model_tolerance`` covers a
known approximation of the reference (0 for exact references, a fraction
``barrier_rel_tol`` of the price for the continuity-corrected barriers).
``z`` is chosen so that, if the engine is right, the probability that *any*
of the n cases fails is ``family_alpha`` (Bonferroni):
z = Phi^-1(1 - family_alpha / (2 n)); about 3.29 for 10 cases at 1%. The
result therefore does not hinge on a lucky seed.

Run ``python -m autocall_pricer.validation`` to print the table in Markdown.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

import numpy as np
from scipy.stats import norm

from .engine import MonteCarloEngine
from .models import BlackScholesModel
from .products import (
    Autocall,
    BarrierOption,
    DigitalOption,
    EuropeanOption,
    BarrierKind,
    OptionKind,
    WorstOfOption,
)

# Broadie-Glasserman-Kou continuity correction constant: -zeta(1/2)/sqrt(2*pi)
BGK_BETA = 0.5825971579390106

#: Market of the validation cases (flat Black-Scholes); maturities in years.
S0, R, Q, VOL, T = 100.0, 0.03, 0.01, 0.20, 1.0

#: Default criterion: family-wise false-alarm probability and barrier tolerance.
FAMILY_ALPHA = 0.01
BARRIER_REL_TOL = 0.002


@dataclass(frozen=True)
class ValidationRow:
    case: str
    reference: str
    mc_price: float
    std_error: float
    ql_price: float
    model_tolerance: float = 0.0
    n_sigma: float = 3.0

    @property
    def abs_error(self) -> float:
        return abs(self.mc_price - self.ql_price)

    @property
    def tolerance(self) -> float:
        return self.n_sigma * self.std_error + self.model_tolerance + 1e-10

    @property
    def n_std_errors(self) -> float:
        return self.abs_error / self.std_error if self.std_error > 0 else float("inf")

    @property
    def passed(self) -> bool:
        return self.abs_error <= self.tolerance


def bonferroni_n_sigma(n_cases: int, family_alpha: float = FAMILY_ALPHA) -> float:
    """Two-sided normal quantile giving a family-wise error rate ``family_alpha``."""
    if n_cases < 1 or not 0.0 < family_alpha < 1.0:
        raise ValueError("need n_cases >= 1 and family_alpha in (0, 1)")
    return float(norm.ppf(1.0 - family_alpha / (2 * n_cases)))


# ---------------------------------------------------------------------------
# QuantLib helpers
# ---------------------------------------------------------------------------
def _ql() -> Any:
    try:
        import QuantLib as ql
    except ImportError as exc:  # pragma: no cover
        raise ImportError("validation requires QuantLib: pip install QuantLib") from exc
    return ql


class _QLMarket:
    """Flat Black-Scholes market in QuantLib; times are Actual/365 year fractions.

    QuantLib's evaluation date is a global setting: it is set to ``today``
    only while a price is computed, then restored, so that using this class
    has no side effect on other QuantLib code.
    """

    def __init__(self, spot: float = S0, rate: float = R, div: float = Q, vol: float = VOL,
                 today: tuple[int, int, int] = (2, 1, 2026)) -> None:
        ql = self.ql = _ql()
        day, month, year = today
        self.today = ql.Date(day, month, year)
        self.dc = ql.Actual365Fixed()
        self.cal = ql.NullCalendar()
        self.rate, self.div = rate, div
        self.process = self.make_process(spot, vol)

    @contextmanager
    def _evaluation_date(self) -> Iterator[None]:
        settings = self.ql.Settings.instance()
        previous = settings.evaluationDate
        settings.evaluationDate = self.today
        try:
            yield
        finally:
            settings.evaluationDate = previous

    def _npv(self, instrument: Any) -> float:
        with self._evaluation_date():
            return float(instrument.NPV())

    def make_process(self, spot: float, vol: float) -> Any:
        ql = self.ql
        return ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(spot)),
            ql.YieldTermStructureHandle(ql.FlatForward(self.today, self.div, self.dc)),
            ql.YieldTermStructureHandle(ql.FlatForward(self.today, self.rate, self.dc)),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(self.today, self.cal, vol, self.dc)),
        )

    def date(self, t: float) -> Any:
        days = round(t * 365)
        if abs(days - t * 365) > 1e-9:
            raise ValueError("use maturities that are whole numbers of days (Actual/365)")
        return self.today + days

    def option_type(self, kind: str) -> Any:
        return self.ql.Option.Call if kind == "call" else self.ql.Option.Put

    def vanilla(self, strike: float, t: float, kind: str) -> float:
        ql = self.ql
        opt = ql.VanillaOption(
            ql.PlainVanillaPayoff(self.option_type(kind), strike), ql.EuropeanExercise(self.date(t))
        )
        opt.setPricingEngine(ql.AnalyticEuropeanEngine(self.process))
        return self._npv(opt)

    def digital(self, strike: float, t: float, kind: str, cash: float = 1.0) -> float:
        ql = self.ql
        opt = ql.VanillaOption(
            ql.CashOrNothingPayoff(self.option_type(kind), strike, cash),
            ql.EuropeanExercise(self.date(t)),
        )
        opt.setPricingEngine(ql.AnalyticEuropeanEngine(self.process))
        return self._npv(opt)

    def barrier(self, strike: float, barrier: float, t: float, barrier_kind: str, kind: str) -> float:
        ql = self.ql
        kinds = {
            "down-and-out": ql.Barrier.DownOut,
            "down-and-in": ql.Barrier.DownIn,
            "up-and-out": ql.Barrier.UpOut,
            "up-and-in": ql.Barrier.UpIn,
        }
        opt = ql.BarrierOption(
            kinds[barrier_kind],
            barrier,
            0.0,
            ql.PlainVanillaPayoff(self.option_type(kind), strike),
            ql.EuropeanExercise(self.date(t)),
        )
        opt.setPricingEngine(ql.AnalyticBarrierEngine(self.process))
        return self._npv(opt)

    def min_call_two_assets(self, strike: float, t: float, spots: list[float], vols: list[float],
                            rho: float) -> float:
        ql = self.ql
        p1 = self.make_process(spots[0], vols[0])
        p2 = self.make_process(spots[1], vols[1])
        opt = ql.BasketOption(
            ql.MinBasketPayoff(ql.PlainVanillaPayoff(ql.Option.Call, strike)),
            ql.EuropeanExercise(self.date(t)),
        )
        opt.setPricingEngine(ql.StulzEngine(p1, p2, rho))
        return self._npv(opt)


# ---------------------------------------------------------------------------
# Validation cases
# ---------------------------------------------------------------------------
CaseFn = Callable[[MonteCarloEngine, _QLMarket], ValidationRow]


class ValidationCase(NamedTuple):
    id: str
    run: CaseFn


def _single_asset_model(vol: float = VOL) -> BlackScholesModel:
    return BlackScholesModel(spots=[S0], vols=[vol], rate=R, dividends=[Q])


def _case_vanilla(engine: MonteCarloEngine, mkt: _QLMarket, kind: OptionKind, strike: float) -> ValidationRow:
    model = _single_asset_model()
    res = engine.price(model, EuropeanOption(strike, T, kind))
    return ValidationRow(
        f"European {kind} K={strike:g}", "AnalyticEuropeanEngine",
        res.price, res.std_error, mkt.vanilla(strike, T, kind),
    )


def _case_digital(engine: MonteCarloEngine, mkt: _QLMarket, kind: OptionKind, strike: float) -> ValidationRow:
    model = _single_asset_model()
    res = engine.price(model, DigitalOption(strike, T, kind))
    return ValidationRow(
        f"Digital cash-or-nothing {kind} K={strike:g}", "AnalyticEuropeanEngine",
        res.price, res.std_error, mkt.digital(strike, T, kind),
    )


def _case_barrier(engine: MonteCarloEngine, mkt: _QLMarket, barrier_kind: BarrierKind, kind: OptionKind,
                  strike: float, barrier: float, rel_tol: float, n_monitoring: int = 252) -> ValidationRow:
    """Discretely monitored barrier vs continuous analytic formula with the
    Broadie-Glasserman-Kou shifted barrier H * exp(+/- beta sigma sqrt(dt))."""
    model = _single_asset_model()
    product = BarrierOption(strike, barrier, T, barrier_kind, kind, n_monitoring)
    res = engine.price(model, product)
    sign = -1.0 if barrier_kind.startswith("down") else 1.0
    shifted = barrier * np.exp(sign * BGK_BETA * VOL * np.sqrt(T / n_monitoring))
    ref = mkt.barrier(strike, shifted, T, barrier_kind, kind)
    # The BGK correction is accurate to o(1/sqrt(n)): allow a fraction rel_tol of the price.
    return ValidationRow(
        f"{barrier_kind} {kind} K={strike:g} H={barrier:g} ({n_monitoring} obs.)",
        "AnalyticBarrierEngine + BGK",
        res.price, res.std_error, ref, model_tolerance=rel_tol * ref,
    )


def _case_worst_of_call(engine: MonteCarloEngine, mkt: _QLMarket, rho: float,
                        strike: float = 95.0) -> ValidationRow:
    spots, vols = [100.0, 100.0], [0.20, 0.30]
    corr = np.array([[1.0, rho], [rho, 1.0]])
    model = BlackScholesModel(spots=spots, vols=vols, rate=R, dividends=[Q, Q], correlation=corr)
    res = engine.price(model, WorstOfOption(strike, T, (0, 1), "call"))
    return ValidationRow(
        f"Worst-of call 2 assets K={strike:g} rho={rho:g}", "StulzEngine",
        res.price, res.std_error, mkt.min_call_two_assets(strike, T, spots, vols, rho),
    )


def _case_autocall_one_date(engine: MonteCarloEngine, mkt: _QLMarket, coupon: float = 0.08,
                            call: float = 1.0, protection: float = 0.7) -> ValidationRow:
    """Autocall with a single observation (= maturity) replicated by QuantLib
    instruments. With x = S_T / S_0:

        payoff = 1 + c 1{x >= R} - (B - x)^+ - (1 - B) 1{x < B}

    i.e. zero-coupon + c digital calls - puts (K = B S0) / S0 - (1 - B) digital puts.
    """
    model = _single_asset_model()
    product = Autocall([T], autocall_barrier=call, coupon=coupon, protection_barrier=protection)
    res = engine.price(model, product)
    ref = (
        np.exp(-R * T)
        + coupon * mkt.digital(call * S0, T, "call")
        - mkt.vanilla(protection * S0, T, "put") / S0
        - (1 - protection) * mkt.digital(protection * S0, T, "put")
    )
    return ValidationRow(
        f"Autocall 1 date, cpn {coupon:.0%}, protection {protection:.0%}",
        "QL replication (ZC + digitals + put)",
        res.price, res.std_error, float(ref),
    )


def validation_cases(barrier_rel_tol: float = BARRIER_REL_TOL) -> list[ValidationCase]:
    def case(id_: str, fn: CaseFn) -> ValidationCase:
        return ValidationCase(id_, fn)

    return [
        case("european-call-K100", lambda e, m: _case_vanilla(e, m, "call", 100.0)),
        case("european-put-K90", lambda e, m: _case_vanilla(e, m, "put", 90.0)),
        case("digital-call-K105", lambda e, m: _case_digital(e, m, "call", 105.0)),
        case("digital-put-K80", lambda e, m: _case_digital(e, m, "put", 80.0)),
        case("down-and-out-call-H85",
             lambda e, m: _case_barrier(e, m, "down-and-out", "call", 100.0, 85.0, barrier_rel_tol)),
        case("up-and-out-put-H120",
             lambda e, m: _case_barrier(e, m, "up-and-out", "put", 100.0, 120.0, barrier_rel_tol)),
        case("down-and-in-put-H80",
             lambda e, m: _case_barrier(e, m, "down-and-in", "put", 100.0, 80.0, barrier_rel_tol)),
        case("worst-of-call-rho+0.5", lambda e, m: _case_worst_of_call(e, m, 0.5)),
        case("worst-of-call-rho-0.3", lambda e, m: _case_worst_of_call(e, m, -0.3)),
        case("autocall-one-date", lambda e, m: _case_autocall_one_date(e, m)),
    ]


def run_validation(
    n_paths: int = 400_000,
    seed: int = 2026,
    family_alpha: float = FAMILY_ALPHA,
    barrier_rel_tol: float = BARRIER_REL_TOL,
) -> list[ValidationRow]:
    engine = MonteCarloEngine(n_paths=n_paths, seed=seed, antithetic=True)
    mkt = _QLMarket()
    cases = validation_cases(barrier_rel_tol)
    n_sigma = bonferroni_n_sigma(len(cases), family_alpha)
    return [replace(c.run(engine, mkt), n_sigma=n_sigma) for c in cases]


def to_markdown(rows: list[ValidationRow]) -> str:
    header = (
        "| Case | QuantLib reference | MC | Std error | QuantLib | Abs. diff | Tolerance | Diff / SE | OK |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|"
    )
    body = [
        f"| {r.case} | {r.reference} | {r.mc_price:.5f} | {r.std_error:.5f} | {r.ql_price:.5f} "
        f"| {r.abs_error:.5f} | {r.tolerance:.5f} | {r.n_std_errors:.2f} | {'✅' if r.passed else '❌'} |"
        for r in rows
    ]
    return "\n".join([header, *body])


if __name__ == "__main__":  # pragma: no cover
    import sys

    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    print(to_markdown(run_validation()))
