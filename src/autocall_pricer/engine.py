"""Vectorised Monte Carlo engine with antithetic variates and confidence intervals."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .models import BlackScholesModel
from .products import Product


def _check_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence!r}")


def _is_integer(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


@dataclass(frozen=True)
class PricingResult:
    price: float
    std_error: float
    n_paths: int
    confidence: float = 0.95

    def __post_init__(self) -> None:
        _check_confidence(self.confidence)

    @property
    def half_width(self) -> float:
        return float(norm.ppf(0.5 + self.confidence / 2) * self.std_error)

    @property
    def ci(self) -> tuple[float, float]:
        """Confidence interval for the price (normal approximation)."""
        return (self.price - self.half_width, self.price + self.half_width)

    def __str__(self) -> str:
        lo, hi = self.ci
        return (
            f"{self.price:.6f} ± {self.half_width:.6f} "
            f"({self.confidence:.0%} CI [{lo:.6f}, {hi:.6f}], {self.n_paths} paths)"
        )


@dataclass(frozen=True)
class MonteCarloEngine:
    """Monte Carlo pricer.

    Parameters
    ----------
    n_paths : total number of simulated paths (antithetic pairs count as 2)
    seed : seed of the NumPy ``Generator`` (PCG64). Every call to ``price``
        restarts from this seed, so one engine gives reproducible prices and
        common random numbers to the greeks. With ``seed=None`` (default) a
        fresh 63-bit seed is drawn from OS entropy when the engine is created and
        stored in ``engine.seed``: distinct engines are then independent, and
        a run can be replayed with ``MonteCarloEngine(seed=engine.seed)``.
    antithetic : use antithetic variates (Z, -Z)
    batch_size : paths simulated at once, to bound memory on long grids
    confidence : level of the reported confidence interval, in (0, 1)
    """

    n_paths: int = 100_000
    seed: int | None = None
    antithetic: bool = True
    batch_size: int = 50_000
    confidence: float = 0.95

    def __post_init__(self) -> None:
        for name in ("n_paths", "batch_size"):
            if not _is_integer(getattr(self, name)):
                raise TypeError(f"{name} must be an integer, got {getattr(self, name)!r}")
        if self.seed is None:
            object.__setattr__(self, "seed", int(np.random.default_rng().integers(2**63)))
        elif not _is_integer(self.seed):
            raise TypeError(f"seed must be a non-negative integer or None, got {self.seed!r}")
        elif self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.n_paths < 2:
            raise ValueError("n_paths must be at least 2")
        if self.antithetic and self.n_paths % 2:
            raise ValueError("n_paths must be even with antithetic variates")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2")
        _check_confidence(self.confidence)

    def _batches(self) -> Iterator[int]:
        """Sizes of the independent draws per batch (half sizes if antithetic)."""
        n = self.n_paths // 2 if self.antithetic else self.n_paths
        step = max(self.batch_size // 2 if self.antithetic else self.batch_size, 1)
        while n > 0:
            m = min(step, n)
            yield m
            n -= m

    def discounted_payoffs(self, model: BlackScholesModel, product: Product) -> np.ndarray:
        """Independent discounted payoff samples.

        With antithetic variates each sample is the average over an antithetic
        pair, so the returned array has ``n_paths / 2`` entries that are i.i.d.
        """
        for a in product.underlyings:
            if not 0 <= a < model.n_assets:
                raise ValueError(f"product references asset {a}, model has {model.n_assets}")
        times = product.times
        df = model.discount(times)
        rng = np.random.default_rng(self.seed)
        out = []
        for m in self._batches():
            z = rng.standard_normal((m, times.size, model.n_assets))
            pv = product.cashflows(model.simulate(times, z), model.spots) @ df
            if self.antithetic:
                pv_anti = product.cashflows(model.simulate(times, -z), model.spots) @ df
                pv = 0.5 * (pv + pv_anti)
            out.append(pv)
        return np.concatenate(out)

    def price(self, model: BlackScholesModel, product: Product) -> PricingResult:
        samples = self.discounted_payoffs(model, product)
        std_error = samples.std(ddof=1) / np.sqrt(samples.size)
        return PricingResult(
            price=float(samples.mean()),
            std_error=float(std_error),
            n_paths=self.n_paths,
            confidence=self.confidence,
        )
