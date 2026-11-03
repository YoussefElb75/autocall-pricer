"""Payoff definitions.

Every product exposes

* ``times``: the simulation / observation dates (year fractions, increasing);
* ``cashflows(paths, spots)``: the undiscounted cash flows paid at each of
  those dates, shape (n_paths, n_times).

``paths`` has shape (n_paths, n_times, n_assets) and contains the levels of
*all* the model's assets; each product picks its own ``underlyings``.
``spots`` are the model's current levels; structured products use them as
initial fixings unless explicit ``initial_levels`` are given.

Products are immutable: array parameters are copied and stored read-only,
and two products compare (and hash) equal when all their parameters are equal.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import ClassVar, Literal, get_args

import numpy as np
import numpy.typing as npt

from ._utils import ValueSemantics, frozen_array

OptionKind = Literal["call", "put"]
BarrierKind = Literal["down-and-out", "down-and-in", "up-and-out", "up-and-in"]

OPTION_KINDS: tuple[str, ...] = get_args(OptionKind)
BARRIER_KINDS: tuple[str, ...] = get_args(BarrierKind)


class Product(ValueSemantics, ABC):
    @property
    @abstractmethod
    def times(self) -> np.ndarray: ...

    @abstractmethod
    def cashflows(self, paths: np.ndarray, spots: np.ndarray) -> np.ndarray: ...

    @property
    def underlyings(self) -> tuple[int, ...]:
        return (0,)

    def fixed_at(self, spots: np.ndarray) -> Product:
        """Product with its initial fixings frozen at ``spots`` (no-op by default)."""
        return self


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _check_kind(kind: str) -> None:
    if kind not in OPTION_KINDS:
        raise ValueError(f"unknown option kind {kind!r}, expected one of {OPTION_KINDS}")


def _check_finite(name: str, value: float, *, positive: bool = False, non_negative: bool = False) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be strictly positive, got {value!r}")
    if non_negative and value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _check_assets(
    assets: tuple[int, ...], min_assets: int, max_assets: int | None, owner: str
) -> tuple[int, ...]:
    assets = tuple(int(a) for a in assets)
    if max_assets is None and len(assets) < min_assets:
        raise ValueError(f"{owner} needs at least {min_assets} underlying(s), got {len(assets)}")
    if max_assets is not None and not min_assets <= len(assets) <= max_assets:
        raise ValueError(
            f"{owner} needs between {min_assets} and {max_assets} underlyings, got {len(assets)}"
        )
    if len(set(assets)) != len(assets):
        raise ValueError("underlyings must be distinct")
    if any(a < 0 for a in assets):
        raise ValueError("asset indices must be non-negative")
    return assets


def _vanilla(x: np.ndarray, strike: float, kind: str) -> np.ndarray:
    if kind == "call":
        return np.maximum(x - strike, 0.0)
    if kind == "put":
        return np.maximum(strike - x, 0.0)
    raise ValueError(f"unknown option kind {kind!r}")


def _terminal_only(values: np.ndarray, n_times: int) -> np.ndarray:
    cf = np.zeros((values.shape[0], n_times))
    cf[:, -1] = values
    return cf


# ---------------------------------------------------------------------------
# Simple products, used mainly for validation against QuantLib
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class EuropeanOption(Product):
    strike: float
    maturity: float
    kind: OptionKind = "call"
    asset: int = 0

    def __post_init__(self) -> None:
        _check_finite("strike", self.strike, non_negative=True)
        _check_finite("maturity", self.maturity, positive=True)
        _check_kind(self.kind)
        _check_assets((self.asset,), 1, 1, type(self).__name__)

    @property
    def times(self) -> np.ndarray:
        return np.array([self.maturity])

    @property
    def underlyings(self) -> tuple[int, ...]:
        return (self.asset,)

    def cashflows(self, paths: np.ndarray, spots: np.ndarray) -> np.ndarray:
        return _terminal_only(_vanilla(paths[:, -1, self.asset], self.strike, self.kind), 1)


@dataclass(frozen=True, eq=False)
class DigitalOption(Product):
    """Cash-or-nothing option paying ``cash`` if S_T > K (call) or S_T < K (put)."""

    strike: float
    maturity: float
    kind: OptionKind = "call"
    cash: float = 1.0
    asset: int = 0

    def __post_init__(self) -> None:
        _check_finite("strike", self.strike, non_negative=True)
        _check_finite("maturity", self.maturity, positive=True)
        _check_finite("cash", self.cash)
        _check_kind(self.kind)
        _check_assets((self.asset,), 1, 1, type(self).__name__)

    @property
    def times(self) -> np.ndarray:
        return np.array([self.maturity])

    @property
    def underlyings(self) -> tuple[int, ...]:
        return (self.asset,)

    def cashflows(self, paths: np.ndarray, spots: np.ndarray) -> np.ndarray:
        s = paths[:, -1, self.asset]
        hit = s > self.strike if self.kind == "call" else s < self.strike
        return _terminal_only(self.cash * hit, 1)


@dataclass(frozen=True, eq=False)
class BarrierOption(Product):
    """Vanilla option with a discretely monitored knock-in / knock-out barrier.

    The barrier is checked on ``n_monitoring`` equally spaced dates, the last
    one being the maturity. No rebate.
    """

    strike: float
    barrier: float
    maturity: float
    barrier_kind: BarrierKind = "down-and-out"
    kind: OptionKind = "call"
    n_monitoring: int = 252
    asset: int = 0

    def __post_init__(self) -> None:
        _check_finite("strike", self.strike, non_negative=True)
        _check_finite("barrier", self.barrier, positive=True)
        _check_finite("maturity", self.maturity, positive=True)
        if self.barrier_kind not in BARRIER_KINDS:
            raise ValueError(f"unknown barrier_kind {self.barrier_kind!r}, expected one of {BARRIER_KINDS}")
        _check_kind(self.kind)
        if isinstance(self.n_monitoring, bool) or not isinstance(self.n_monitoring, (int, np.integer)):
            raise TypeError("n_monitoring must be an integer")
        if self.n_monitoring < 1:
            raise ValueError("n_monitoring must be at least 1")
        _check_assets((self.asset,), 1, 1, type(self).__name__)

    @property
    def times(self) -> np.ndarray:
        return self.maturity * np.arange(1, self.n_monitoring + 1) / self.n_monitoring

    @property
    def underlyings(self) -> tuple[int, ...]:
        return (self.asset,)

    def cashflows(self, paths: np.ndarray, spots: np.ndarray) -> np.ndarray:
        s = paths[:, :, self.asset]
        if self.barrier_kind.startswith("down"):
            touched = (s <= self.barrier).any(axis=1)
        else:
            touched = (s >= self.barrier).any(axis=1)
        alive = ~touched if self.barrier_kind.endswith("out") else touched
        payoff = _vanilla(s[:, -1], self.strike, self.kind) * alive
        return _terminal_only(payoff, s.shape[1])


@dataclass(frozen=True, eq=False)
class WorstOfOption(Product):
    """European option on the minimum of several assets: (min_i S_i(T) - K)^+ for a call."""

    strike: float
    maturity: float
    assets: tuple[int, ...] = (0, 1)
    kind: OptionKind = "call"

    def __post_init__(self) -> None:
        _check_finite("strike", self.strike, non_negative=True)
        _check_finite("maturity", self.maturity, positive=True)
        _check_kind(self.kind)
        object.__setattr__(self, "assets", _check_assets(self.assets, 1, None, type(self).__name__))

    @property
    def times(self) -> np.ndarray:
        return np.array([self.maturity])

    @property
    def underlyings(self) -> tuple[int, ...]:
        return tuple(self.assets)

    def cashflows(self, paths: np.ndarray, spots: np.ndarray) -> np.ndarray:
        worst = paths[:, -1, list(self.assets)].min(axis=1)
        return _terminal_only(_vanilla(worst, self.strike, self.kind), 1)


# ---------------------------------------------------------------------------
# Autocalls
# ---------------------------------------------------------------------------
def _per_date(name: str, value: npt.ArrayLike, n_times: int) -> np.ndarray:
    """Broadcast a scalar or per-date barrier to shape (n_times,), validated and read-only."""
    arr = np.asarray(value, dtype=float)
    try:
        arr = np.broadcast_to(arr, (n_times,))
    except ValueError:
        raise ValueError(
            f"{name} must be a scalar or have one value per observation date "
            f"({n_times}), got shape {arr.shape}"
        ) from None
    if np.any(np.isnan(arr)):
        raise ValueError(f"{name} must not contain NaN")
    if np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative")
    return frozen_array(arr)


@dataclass(frozen=True, init=False, eq=False)
class WorstOfAutocall(Product):
    """Worst-of autocallable note on 2 to 5 underlyings.

    All levels are expressed as fractions of the initial fixings: the
    performance on date t_k is ``perf_k = min_i S_i(t_k) / S_i(0)``.

    On each observation date t_k, as long as the note is alive:

    * coupon: if ``perf_k >= coupon_barrier[k]`` the note pays
      ``nominal * coupon * (1 + m)``, where ``m`` is the number of previously
      missed coupons when ``memory`` is on (0 otherwise);
    * early redemption (k < last date, from ``first_call_index`` on): if
      ``perf_k >= autocall_barrier[k]`` the note pays ``nominal`` and stops;
    * maturity (last date): the note pays ``nominal`` if
      ``perf >= protection_barrier`` (European observation), otherwise
      ``nominal * perf`` (capital loss).

    Barriers may be scalars or one value per date (step-down); ``inf`` makes
    a barrier unreachable. ``coupon_barrier`` defaults to the autocall
    barrier. ``first_call_index`` ranges from 0 to ``n_dates - 1``; the latter
    means no early redemption at all.

    With ``coupon_barrier = autocall_barrier`` and ``memory = True`` this is
    the classic "snowball" / Athena autocall; with a lower coupon barrier it
    is a Phoenix. Cash flows are paid on the observation dates.
    """

    observation_times: np.ndarray
    autocall_barrier: np.ndarray
    coupon: float
    coupon_barrier: np.ndarray
    memory: bool
    protection_barrier: float
    nominal: float
    assets: tuple[int, ...]
    initial_levels: np.ndarray | None
    first_call_index: int

    min_assets: ClassVar[int] = 2
    max_assets: ClassVar[int] = 5
    default_assets: ClassVar[tuple[int, ...]] = (0, 1)

    def __init__(
        self,
        observation_times: npt.ArrayLike,
        autocall_barrier: npt.ArrayLike = 1.0,
        coupon: float = 0.05,
        coupon_barrier: npt.ArrayLike | None = None,
        memory: bool = True,
        protection_barrier: float = 0.6,
        nominal: float = 1.0,
        assets: tuple[int, ...] | None = None,
        initial_levels: npt.ArrayLike | None = None,
        first_call_index: int = 0,
    ) -> None:
        t = np.asarray(observation_times, dtype=float)
        if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)) or t[0] <= 0 or np.any(np.diff(t) <= 0):
            raise ValueError("observation_times must be strictly increasing, positive and finite")
        n = t.size
        assets_ = _check_assets(self.default_assets if assets is None else assets,
                                self.min_assets, self.max_assets, type(self).__name__)
        ac = _per_date("autocall_barrier", autocall_barrier, n)
        cb = ac if coupon_barrier is None else _per_date("coupon_barrier", coupon_barrier, n)
        _check_finite("coupon", coupon, non_negative=True)
        _check_finite("protection_barrier", protection_barrier, non_negative=True)
        _check_finite("nominal", nominal, positive=True)
        if isinstance(first_call_index, bool) or not isinstance(first_call_index, (int, np.integer)):
            raise TypeError("first_call_index must be an integer")
        if not 0 <= first_call_index <= n - 1:
            raise ValueError(f"first_call_index must lie in [0, {n - 1}], got {first_call_index}")
        levels = None
        if initial_levels is not None:
            levels = frozen_array(initial_levels)
            if levels.shape != (len(assets_),) or not np.all(np.isfinite(levels)) or np.any(levels <= 0):
                raise ValueError("initial_levels must be positive and finite, one per underlying")

        set_ = object.__setattr__
        set_(self, "observation_times", frozen_array(t))
        set_(self, "autocall_barrier", ac)
        set_(self, "coupon", float(coupon))
        set_(self, "coupon_barrier", cb)
        set_(self, "memory", bool(memory))
        set_(self, "protection_barrier", float(protection_barrier))
        set_(self, "nominal", float(nominal))
        set_(self, "assets", assets_)
        set_(self, "initial_levels", levels)
        set_(self, "first_call_index", int(first_call_index))

    @property
    def times(self) -> np.ndarray:
        return self.observation_times

    @property
    def underlyings(self) -> tuple[int, ...]:
        return self.assets

    def fixed_at(self, spots: np.ndarray) -> WorstOfAutocall:
        if self.initial_levels is not None:
            return self
        return replace(self, initial_levels=np.asarray(spots, dtype=float)[list(self.assets)])

    def performance(self, paths: np.ndarray, spots: np.ndarray) -> np.ndarray:
        """Worst performance on each observation date, shape (n_paths, n_times)."""
        idx = list(self.assets)
        ref = self.initial_levels if self.initial_levels is not None else np.asarray(spots)[idx]
        return (paths[:, :, idx] / ref).min(axis=2)

    def cashflows(self, paths: np.ndarray, spots: np.ndarray) -> np.ndarray:
        perf = self.performance(paths, spots)
        n_paths, n_times = perf.shape
        cf = np.zeros((n_paths, n_times))
        alive = np.ones(n_paths, dtype=bool)
        missed = np.zeros(n_paths)
        last = n_times - 1
        for k in range(n_times):
            p = perf[:, k]
            pays_coupon = alive & (p >= self.coupon_barrier[k])
            n_coupons = 1.0 + (missed if self.memory else 0.0)
            cf[:, k] += np.where(pays_coupon, self.nominal * self.coupon * n_coupons, 0.0)
            missed = np.where(pays_coupon, 0.0, missed + alive)
            if k < last:
                if k >= self.first_call_index:
                    called = alive & (p >= self.autocall_barrier[k])
                    cf[:, k] += np.where(called, self.nominal, 0.0)
                    alive &= ~called
            else:
                protected = p >= self.protection_barrier
                redemption = np.where(protected, self.nominal, self.nominal * p)
                cf[:, k] += np.where(alive, redemption, 0.0)
        return cf

    def redemption_times(self, paths: np.ndarray, spots: np.ndarray) -> np.ndarray:
        """Redemption time of each path, shape (n_paths,) (diagnostics: average it
        for the expected life of the note)."""
        perf = self.performance(paths, spots)
        n_times = perf.shape[1]
        callable_ = np.zeros(n_times, dtype=bool)
        callable_[self.first_call_index : n_times - 1] = True
        called = (perf >= self.autocall_barrier[None, :]) & callable_[None, :]
        first = np.where(called.any(axis=1), called.argmax(axis=1), n_times - 1)
        return self.observation_times[first]

    def expected_life(self, paths: np.ndarray, spots: np.ndarray) -> np.ndarray:
        """Deprecated alias of :meth:`redemption_times` (it never returned an expectation)."""
        warnings.warn(
            "expected_life() is deprecated, use redemption_times()", DeprecationWarning, stacklevel=2
        )
        return self.redemption_times(paths, spots)


class Autocall(WorstOfAutocall):
    """Single-underlying autocall (see :class:`WorstOfAutocall` for the payoff)."""

    min_assets: ClassVar[int] = 1
    max_assets: ClassVar[int] = 1
    default_assets: ClassVar[tuple[int, ...]] = (0,)
