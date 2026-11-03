"""Closed-form Black-Scholes prices and greeks (single asset, continuous dividend yield).

Used as references by the tests and the validation; ``kind`` is "call" or "put".
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _check(strike: float, t: float, vol: float, kind: str) -> None:
    if kind not in ("call", "put"):
        raise ValueError(f"unknown option kind {kind!r}")
    if t <= 0 or vol <= 0 or strike < 0:
        raise ValueError("closed forms need t > 0, vol > 0 and strike >= 0")


def d1_d2(spot: float, strike: float, t: float, rate: float, div: float, vol: float) -> tuple[float, float]:
    sd = vol * np.sqrt(t)
    if strike == 0.0:  # the option is exercised with probability 1
        return float("inf"), float("inf")
    d1 = (np.log(spot / strike) + (rate - div + 0.5 * vol**2) * t) / sd
    return float(d1), float(d1 - sd)


def bs_price(spot: float, strike: float, t: float, rate: float, div: float, vol: float,
             kind: str = "call") -> float:
    _check(strike, t, vol, kind)
    d1, d2 = d1_d2(spot, strike, t, rate, div, vol)
    fwd_pv, k_pv = spot * np.exp(-div * t), strike * np.exp(-rate * t)
    if kind == "call":
        return float(fwd_pv * norm.cdf(d1) - k_pv * norm.cdf(d2))
    return float(k_pv * norm.cdf(-d2) - fwd_pv * norm.cdf(-d1))


def bs_digital(spot: float, strike: float, t: float, rate: float, div: float, vol: float,
               kind: str = "call", cash: float = 1.0) -> float:
    """Cash-or-nothing: ``cash`` paid at t if S_t > K (call) or S_t < K (put)."""
    _check(strike, t, vol, kind)
    _, d2 = d1_d2(spot, strike, t, rate, div, vol)
    sign = 1.0 if kind == "call" else -1.0
    return float(cash * np.exp(-rate * t) * norm.cdf(sign * d2))


def bs_delta(spot: float, strike: float, t: float, rate: float, div: float, vol: float,
             kind: str = "call") -> float:
    _check(strike, t, vol, kind)
    d1, _ = d1_d2(spot, strike, t, rate, div, vol)
    sign = 1.0 if kind == "call" else -1.0
    return float(sign * np.exp(-div * t) * norm.cdf(sign * d1))


def bs_gamma(spot: float, strike: float, t: float, rate: float, div: float, vol: float) -> float:
    _check(strike, t, vol, "call")
    d1, _ = d1_d2(spot, strike, t, rate, div, vol)
    return float(np.exp(-div * t) * norm.pdf(d1) / (spot * vol * np.sqrt(t)))


def bs_vega(spot: float, strike: float, t: float, rate: float, div: float, vol: float) -> float:
    """dP/dsigma (natural units; multiply by 0.01 for one vol point)."""
    _check(strike, t, vol, "call")
    d1, _ = d1_d2(spot, strike, t, rate, div, vol)
    return float(spot * np.exp(-div * t) * norm.pdf(d1) * np.sqrt(t))
