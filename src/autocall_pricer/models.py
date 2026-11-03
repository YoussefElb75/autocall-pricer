"""Multi-asset Black-Scholes model with constant rate, dividends and correlation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

from ._utils import ValueSemantics, frozen_array

#: Eigenvalues of the correlation matrix above ``-PSD_TOLERANCE`` are treated as
#: zero (round-off); anything more negative means the matrix is not a correlation.
PSD_TOLERANCE = 1e-10


def correlation_factor(corr: np.ndarray, tol: float = PSD_TOLERANCE) -> np.ndarray:
    """Matrix ``L`` with ``L @ L.T == corr`` for a positive semi-definite ``corr``.

    The Cholesky factor (lower triangular) is returned whenever ``corr`` is
    positive definite. Singular matrices (e.g. a correlation of exactly 1) have
    no Cholesky factor; ``L = V sqrt(max(w, 0))`` is then built from the
    eigen-decomposition ``corr = V diag(w) V.T``.
    """
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        pass
    w, v = np.linalg.eigh(corr)
    if w.min() < -tol:
        raise ValueError(
            f"correlation matrix is not positive semi-definite (smallest eigenvalue {w.min():.3g})"
        )
    return v * np.sqrt(np.clip(w, 0.0, None))


@dataclass(frozen=True, init=False, eq=False)
class BlackScholesModel(ValueSemantics):
    """Correlated geometric Brownian motions under the risk-neutral measure.

    dS_i / S_i = (r - q_i) dt + sigma_i dW_i,   d<W_i, W_j> = rho_ij dt

    Parameters
    ----------
    spots : current levels S_i(0), shape (n_assets,)
    vols : constant volatilities sigma_i, scalar or shape (n_assets,)
    rate : constant continuously compounded risk-free rate r
    dividends : constant continuous dividend yields q_i, scalar or shape
        (n_assets,) (default 0)
    correlation : correlation matrix (default identity); must be symmetric
        positive semi-definite (perfect correlations are allowed).

    All inputs are copied and stored as read-only arrays.
    """

    spots: np.ndarray
    vols: np.ndarray
    rate: float
    dividends: np.ndarray
    correlation: np.ndarray
    _chol: np.ndarray = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        spots: npt.ArrayLike,
        vols: npt.ArrayLike,
        rate: float,
        dividends: npt.ArrayLike | None = None,
        correlation: npt.ArrayLike | None = None,
    ) -> None:
        spots_ = np.array(spots, dtype=float)
        if spots_.ndim == 0:
            spots_ = spots_.reshape(1)
        if spots_.ndim != 1:
            raise ValueError(f"spots must be a 1-D array, got shape {spots_.shape}")
        n = spots_.size
        if n == 0:
            raise ValueError("the model needs at least one asset")
        vols_ = np.broadcast_to(np.asarray(vols, dtype=float), (n,))
        divs = np.zeros(n) if dividends is None else np.broadcast_to(np.asarray(dividends, dtype=float), (n,))
        corr = np.eye(n) if correlation is None else np.asarray(correlation, dtype=float)
        rate = float(rate)

        for name, value in (("spots", spots_), ("vols", vols_), ("dividends", divs), ("correlation", corr)):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
        if not np.isfinite(rate):
            raise ValueError("rate must be finite")
        if np.any(spots_ <= 0):
            raise ValueError("spots must be strictly positive")
        if np.any(vols_ < 0):
            raise ValueError("volatilities must be non-negative")
        if corr.shape != (n, n):
            raise ValueError(f"correlation must have shape ({n}, {n}), got {corr.shape}")
        if not np.allclose(corr, corr.T):
            raise ValueError("correlation matrix must be symmetric")
        if not np.allclose(np.diag(corr), 1.0):
            raise ValueError("correlation matrix must have a unit diagonal")
        if np.any(np.abs(corr) > 1.0 + 1e-12):
            raise ValueError("correlations must lie in [-1, 1]")

        object.__setattr__(self, "spots", frozen_array(spots_))
        object.__setattr__(self, "vols", frozen_array(vols_))
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "dividends", frozen_array(divs))
        object.__setattr__(self, "correlation", frozen_array(corr))
        object.__setattr__(self, "_chol", frozen_array(correlation_factor(self.correlation)))

    @property
    def n_assets(self) -> int:
        return int(self.spots.size)

    @property
    def cholesky(self) -> np.ndarray:
        """Factor L with L @ L.T == correlation.

        Lower triangular (Cholesky) when the correlation is positive definite,
        eigen-decomposition factor when it is only semi-definite.
        """
        return self._chol

    def discount(self, times: npt.ArrayLike) -> np.ndarray:
        """Discount factors exp(-r t)."""
        return np.exp(-self.rate * np.asarray(times, dtype=float))

    def forwards(self, t: float) -> np.ndarray:
        """Forward levels S_i(0) exp((r - q_i) t)."""
        return self.spots * np.exp((self.rate - self.dividends) * t)

    def simulate(self, times: npt.ArrayLike, normals: np.ndarray) -> np.ndarray:
        """Exact simulation of the asset levels on a time grid.

        Parameters
        ----------
        times : strictly increasing positive times, shape (n_times,)
        normals : independent standard normals, shape (n_paths, n_times, n_assets)

        Returns
        -------
        Asset levels, shape (n_paths, n_times, n_assets). The log-normal
        transition is sampled exactly, so there is no discretisation bias
        between grid dates.
        """
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times.size == 0 or times[0] <= 0 or np.any(np.diff(times) <= 0):
            raise ValueError("times must be a non-empty, strictly increasing array of positive values")
        if normals.ndim != 3 or normals.shape[1:] != (times.size, self.n_assets):
            raise ValueError(
                f"normals must have shape (n_paths, {times.size}, {self.n_assets}), got {normals.shape}"
            )
        dt = np.diff(times, prepend=0.0)[None, :, None]
        correlated = normals @ self._chol.T  # (n_paths, n_times, n_assets)
        drift = (self.rate - self.dividends - 0.5 * self.vols**2)[None, None, :] * dt
        diffusion = self.vols[None, None, :] * np.sqrt(dt) * correlated
        log_paths = np.cumsum(drift + diffusion, axis=1)
        return self.spots[None, None, :] * np.exp(log_paths)

    # --- helpers used by the bump-and-reprice greeks -----------------------
    def with_spot(self, i: int, value: float) -> BlackScholesModel:
        spots = self.spots.copy()
        spots[i] = value
        return replace(self, spots=spots)

    def with_vol(self, i: int, value: float) -> BlackScholesModel:
        vols = self.vols.copy()
        vols[i] = value
        return replace(self, vols=vols)

    def with_correlation(self, correlation: npt.ArrayLike) -> BlackScholesModel:
        return replace(self, correlation=np.asarray(correlation, dtype=float))


def constant_correlation(n_assets: int, rho: float) -> np.ndarray:
    """Correlation matrix with every off-diagonal entry equal to ``rho``."""
    corr = np.full((n_assets, n_assets), float(rho))
    np.fill_diagonal(corr, 1.0)
    return corr
