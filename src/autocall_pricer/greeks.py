"""Bump-and-reprice greeks with common random numbers.

All derivatives are in natural units:

* delta_i = dP / dS_i
* gamma_i = d2P / dS_i^2
* vega_i  = dP / d sigma_i          (multiply by 0.01 for "per vol point")
* corr    = dP / d rho              (parallel shift of all off-diagonal
                                     correlations, or of a single pair)

Every bumped price is computed with the same engine, hence the same seed and
the same normal draws (common random numbers), which removes most of the
Monte Carlo noise from the finite differences. Structured products have their
initial fixings frozen at the unbumped spots before bumping, so that a spot
bump moves the performance (otherwise an autocall struck "today" would have a
zero delta by construction).

Payoffs with discontinuities (autocall/coupon barriers, digitals) make
pathwise finite differences noisy, gamma in particular: use relative spot
bumps of about 1% or more and enough paths.

The vol bump is ``min(abs_bump, max_rel_bump * sigma)``: a fixed 1-point bump
would be as large as the volatility itself for very low vols, where the price
is strongly convex in sigma and a wide central difference is badly biased.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._utils import ValueSemantics
from .engine import MonteCarloEngine
from .models import BlackScholesModel
from .products import Product


@dataclass(frozen=True, eq=False)
class Greeks(ValueSemantics):
    price: float
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    correlation: float | None = None

    def as_table(self) -> str:
        lines = [f"price = {self.price:.6f}"]
        lines.append(f"{'asset':>5} {'delta':>12} {'gamma':>12} {'vega':>12}")
        for i, (d, g, v) in enumerate(zip(self.delta, self.gamma, self.vega)):
            lines.append(f"{i:>5} {d:>12.6f} {g:>12.6f} {v:>12.6f}")
        if self.correlation is not None:
            lines.append(f"dP/drho = {self.correlation:.6f}")
        return "\n".join(lines)


def _price(engine: MonteCarloEngine, model: BlackScholesModel, product: Product) -> float:
    return engine.price(model, product).price


def delta_gamma(
    model: BlackScholesModel,
    product: Product,
    engine: MonteCarloEngine,
    asset: int,
    rel_bump: float = 0.01,
    base_price: float | None = None,
) -> tuple[float, float]:
    """Central-difference delta and gamma with respect to one spot."""
    if not 0.0 < rel_bump < 1.0:
        raise ValueError(f"rel_bump must lie in (0, 1), got {rel_bump!r}")
    product = product.fixed_at(model.spots)
    s = model.spots[asset]
    h = rel_bump * s
    up = _price(engine, model.with_spot(asset, s + h), product)
    down = _price(engine, model.with_spot(asset, s - h), product)
    mid = _price(engine, model, product) if base_price is None else base_price
    return (up - down) / (2 * h), (up - 2 * mid + down) / h**2


def vega(
    model: BlackScholesModel,
    product: Product,
    engine: MonteCarloEngine,
    asset: int,
    abs_bump: float = 0.01,
    max_rel_bump: float = 0.1,
) -> float:
    """Central-difference dP/dsigma_i with bump ``min(abs_bump, max_rel_bump * sigma_i)``.

    A forward difference with ``abs_bump`` is used when sigma_i = 0.
    """
    if abs_bump <= 0 or not 0.0 < max_rel_bump < 1.0:
        raise ValueError("abs_bump must be positive and max_rel_bump in (0, 1)")
    product = product.fixed_at(model.spots)
    v = float(model.vols[asset])
    if v == 0.0:
        up = _price(engine, model.with_vol(asset, abs_bump), product)
        return (up - _price(engine, model, product)) / abs_bump
    h = min(abs_bump, max_rel_bump * v)
    up = _price(engine, model.with_vol(asset, v + h), product)
    down = _price(engine, model.with_vol(asset, v - h), product)
    return (up - down) / (2 * h)


def correlation_sensitivity(
    model: BlackScholesModel,
    product: Product,
    engine: MonteCarloEngine,
    abs_bump: float = 0.01,
    pair: tuple[int, int] | None = None,
) -> float:
    """dP/drho by central differences.

    ``pair=None`` shifts every off-diagonal correlation between the product's
    underlyings in parallel; ``pair=(i, j)`` bumps rho_ij only, and both
    assets must be underlyings of the product. Each bumped matrix must remain
    a valid (positive semi-definite) correlation matrix, checked by the model.
    """
    if abs_bump <= 0:
        raise ValueError("abs_bump must be positive")
    product = product.fixed_at(model.spots)
    if pair is None:
        idx = list(product.underlyings)
        mask = np.zeros_like(model.correlation, dtype=bool)
        mask[np.ix_(idx, idx)] = True
        np.fill_diagonal(mask, False)
    else:
        i, j = pair
        if i == j:
            raise ValueError("pair must reference two different assets")
        if i not in product.underlyings or j not in product.underlyings:
            raise ValueError(f"pair {pair} must reference underlyings of the product {product.underlyings}")
        mask = np.zeros_like(model.correlation, dtype=bool)
        mask[i, j] = mask[j, i] = True
    if not mask.any():
        raise ValueError("no correlation to bump (single underlying)")

    def bumped(h: float) -> float:
        corr = model.correlation.copy()
        corr[mask] += h
        return _price(engine, model.with_correlation(corr), product)

    return (bumped(abs_bump) - bumped(-abs_bump)) / (2 * abs_bump)


def compute_greeks(
    model: BlackScholesModel,
    product: Product,
    engine: MonteCarloEngine | None = None,
    spot_bump: float = 0.01,
    vol_bump: float = 0.01,
    corr_bump: float = 0.01,
    max_rel_vol_bump: float = 0.1,
) -> Greeks:
    """Price, delta, gamma and vega for every underlying of ``product``.

    The correlation sensitivity is added when the product has more than one
    underlying. Greeks of assets the product does not reference are zero.
    """
    engine = engine or MonteCarloEngine()
    product = product.fixed_at(model.spots)
    base = _price(engine, model, product)
    n = model.n_assets
    delta, gamma, vegas = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in product.underlyings:
        delta[i], gamma[i] = delta_gamma(model, product, engine, i, spot_bump, base)
        vegas[i] = vega(model, product, engine, i, vol_bump, max_rel_vol_bump)
    corr = None
    if len(product.underlyings) > 1:
        corr = correlation_sensitivity(model, product, engine, corr_bump)
    return Greeks(price=base, delta=delta, gamma=gamma, vega=vegas, correlation=corr)
