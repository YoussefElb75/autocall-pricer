"""autocall-pricer: Monte Carlo pricing of autocalls and worst-of autocalls."""

from .engine import MonteCarloEngine, PricingResult
from .greeks import Greeks, compute_greeks, correlation_sensitivity, delta_gamma, vega
from .models import BlackScholesModel, constant_correlation
from .products import (
    Autocall,
    BarrierOption,
    DigitalOption,
    EuropeanOption,
    Product,
    WorstOfAutocall,
    WorstOfOption,
)

__version__ = "0.1.0"

__all__ = [
    "Autocall",
    "BarrierOption",
    "BlackScholesModel",
    "DigitalOption",
    "EuropeanOption",
    "Greeks",
    "MonteCarloEngine",
    "PricingResult",
    "Product",
    "WorstOfAutocall",
    "WorstOfOption",
    "compute_greeks",
    "constant_correlation",
    "correlation_sensitivity",
    "delta_gamma",
    "vega",
]
