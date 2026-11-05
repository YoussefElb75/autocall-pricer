"""Generate the figures shown in the README (saved as PNG in ``docs/figures/``).

Usage::

    python scripts/make_figures.py              # full quality, about a minute
    python scripts/make_figures.py --quick      # few paths, a few seconds (smoke test)
    python scripts/make_figures.py --out DIR    # another output directory

Every figure function takes a :class:`Config` (path counts and grids) and
returns a matplotlib ``Figure``; all random draws use fixed seeds, so the
figures are reproducible. The pricing itself is done by the library
(``MonteCarloEngine``, ``delta_gamma``, ``vega``, ``run_validation``); the
script only post-processes samples and draws.

The figures use an opaque light background so that they stay readable on
both the light and the dark GitHub themes.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from autocall_pricer import (
    Autocall,
    BlackScholesModel,
    MonteCarloEngine,
    WorstOfAutocall,
    constant_correlation,
    delta_gamma,
    vega,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "figures"

# ---------------------------------------------------------------------------
# Style: validated categorical slots (blue, orange, aqua) on a light surface
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e2e1dc"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#8a8984"

STYLE: dict[Any, Any] = {  # keys are rcParams names
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 9.5,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "text.color": INK,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "font.size": 9.5,
}

# ---------------------------------------------------------------------------
# Products and market (the autocall of the README / notebook)
# ---------------------------------------------------------------------------
OBS = [0.25 * k for k in range(1, 13)]  # 3 years, quarterly
COUPON = 0.02
PROTECTION = 0.6
S0, VOL, RATE, DIV = 100.0, 0.20, 0.03, 0.01


def base_model(vol: float = VOL, spot: float = S0) -> BlackScholesModel:
    return BlackScholesModel(spots=[spot], vols=[vol], rate=RATE, dividends=[DIV])


def base_autocall() -> Autocall:
    return Autocall(OBS, autocall_barrier=1.0, coupon=COUPON, memory=True, protection_barrier=PROTECTION)


@dataclass(frozen=True)
class Config:
    """Number of paths and grid sizes of each figure."""

    conv_paths: int = 1_000_000
    conv_ref_paths: int = 4_000_000
    validation_paths: int = 400_000
    sim_paths: int = 200_000
    greek_paths: int = 200_000
    greek_spots: int = 45
    corr_paths: int = 200_000
    corr_points: int = 15


FULL = Config()
QUICK = Config(conv_paths=20_000, conv_ref_paths=40_000, validation_paths=10_000, sim_paths=10_000,
               greek_paths=4_000, greek_spots=7, corr_paths=4_000, corr_points=4)


def _new_figure(width: float, height: float) -> Figure:
    # A bare Figure (not pyplot-managed): no global state, nothing left open.
    return Figure(figsize=(width, height), dpi=150, layout="constrained")


def _styled(fn: Callable[..., Figure]) -> Callable[..., Figure]:
    """Run a figure builder with the style active (artists read rcParams at creation)."""
    import functools

    import matplotlib as mpl

    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> Figure:
        with mpl.rc_context(STYLE):
            return fn(*args, **kwargs)

    return wrapper


def _ref_line(ax: Axes, x: float, label: str, *, y_text: float = 0.97) -> None:
    ax.axvline(x, color=NEUTRAL, lw=1, ls=(0, (3, 3)))
    ax.text(x, y_text, f" {label}", transform=ax.get_xaxis_transform(), color=INK_2, fontsize=8, va="top")


# ---------------------------------------------------------------------------
# 1. Convergence
# ---------------------------------------------------------------------------
def _running_estimates(samples: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard error of the first ``counts`` i.i.d. samples."""
    s1 = np.cumsum(samples)[counts - 1]
    s2 = np.cumsum(samples**2)[counts - 1]
    mean = s1 / counts
    var = (s2 - counts * mean**2) / (counts - 1)
    return mean, np.sqrt(np.maximum(var, 0.0) / counts)


@_styled
def fig_convergence(cfg: Config = FULL) -> Figure:
    model, product = base_model(), base_autocall()
    ref = MonteCarloEngine(n_paths=cfg.conv_ref_paths, seed=2026).price(model, product)
    n_paths = np.unique(np.geomspace(1_000, cfg.conv_paths, 40).astype(int) // 2 * 2)

    plain = MonteCarloEngine(n_paths=cfg.conv_paths, seed=11, antithetic=False).discounted_payoffs(model, product)
    anti = MonteCarloEngine(n_paths=cfg.conv_paths, seed=11, antithetic=True).discounted_payoffs(model, product)
    est = {
        "plain": (*_running_estimates(plain, n_paths), BLUE),
        "antithetic": (*_running_estimates(anti, n_paths // 2), ORANGE),  # one sample per pair
    }

    fig = _new_figure(10, 3.6)
    axes = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, (name, (mean, se, color)) in zip(axes[:2], est.items()):
        ax.fill_between(n_paths, mean - 1.96 * se, mean + 1.96 * se, color=color, alpha=0.22, lw=0,
                        label="95% confidence band")
        ax.plot(n_paths, mean, color=color, lw=1.6, label="MC estimate")
        ax.axhline(ref.price, color=INK_2, lw=1, ls=(0, (3, 3)),
                   label=f"reference {ref.price:.4f} ({cfg.conv_ref_paths:,} paths)")
        ax.set_xscale("log")
        ax.set_xlabel("number of paths")
        ax.set_title(name.capitalize())
        ax.legend(loc="lower right")
    axes[0].set_ylabel("price (fraction of nominal)")
    axes[1].sharey(axes[0])
    axes[1].tick_params(labelleft=False)

    ax = axes[2]
    for name, (_, se, color) in est.items():
        ax.plot(n_paths, se, color=color, label=name)
    guide = est["plain"][1][0] * np.sqrt(n_paths[0] / n_paths)
    ax.plot(n_paths, guide, color=NEUTRAL, lw=1, ls=(0, (3, 3)), label=r"$\propto 1/\sqrt{N}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of paths")
    ax.set_ylabel("standard error")
    ax.set_title("Standard error")
    ax.legend(loc="upper right")
    fig.suptitle("Monte Carlo convergence of the 3-year autocall", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    return fig


# ---------------------------------------------------------------------------
# 2. Validation against QuantLib
# ---------------------------------------------------------------------------
@_styled
def fig_validation(cfg: Config = FULL) -> Figure:
    from autocall_pricer.validation import run_validation

    rows = run_validation(n_paths=cfg.validation_paths, seed=2026)
    z = rows[0].n_sigma
    err = np.array([(r.mc_price - r.ql_price) / r.std_error for r in rows])
    tol = np.array([(r.tolerance - 1e-10) / r.std_error for r in rows])
    y = np.arange(len(rows))[::-1]

    fig = _new_figure(9, 4.4)
    ax = fig.subplots()
    ax.axvspan(-z, z, color=BLUE, alpha=0.08, lw=0)
    ax.axvline(0, color=INK_2, lw=1)
    for v in (-z, z):
        ax.axvline(v, color=NEUTRAL, lw=1, ls=(0, (3, 3)))
    ax.text(z, 0.01, f"±{z:.2f} SE (Bonferroni, 1%) ", transform=ax.get_xaxis_transform(),
            color=INK_2, fontsize=8, va="bottom", ha="right")
    widened = tol > z + 1e-9
    ax.scatter(np.r_[-tol[widened], tol[widened]], np.r_[y[widened], y[widened]], marker="|", s=120,
               color=NEUTRAL, lw=1.5, label="barrier cases: tolerance incl. 0.2% model error", zorder=2)
    passed = np.array([r.passed for r in rows])
    ax.scatter(err[passed], y[passed], s=48, color=BLUE, edgecolor=SURFACE, lw=1.5, zorder=3,
               label="(MC − QuantLib) / SE — pass")
    if (~passed).any():
        ax.scatter(err[~passed], y[~passed], s=48, color=ORANGE, marker="X", zorder=3, label="fail")
    ax.set_yticks(y, [r.case for r in rows])
    lim = max(float(np.abs(err).max()), float(tol.max())) * 1.12
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("signed error in Monte Carlo standard errors")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Validation against QuantLib — {len(rows)} cases, {cfg.validation_paths:,} antithetic paths")
    fig.legend(loc="outside lower center", ncol=2)
    return fig


# ---------------------------------------------------------------------------
# 3 & 4. Path statistics of the autocall
# ---------------------------------------------------------------------------
def simulate_autocall(n_paths: int, vol: float = VOL, seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Redemption time, final worst performance and discounted payoff of each path."""
    model, product = base_model(vol), base_autocall()
    times = product.times
    rng = np.random.default_rng(seed)
    paths = model.simulate(times, rng.standard_normal((n_paths, times.size, model.n_assets)))
    redemption = product.redemption_times(paths, model.spots)
    final_perf = product.performance(paths, model.spots)[:, -1]
    pv = product.cashflows(paths, model.spots) @ model.discount(times)
    return redemption, final_perf, pv


@_styled
def fig_call_probability(cfg: Config = FULL) -> Figure:
    times = np.asarray(OBS)[:-1]  # early-redemption dates (the last date is the maturity)
    red, _, _ = simulate_autocall(cfg.sim_paths)
    prob = np.array([np.mean(red == t) for t in times])  # P(called at t_k)
    alive = np.array([np.mean(red >= t) for t in times])  # P(still alive at t_k)
    conditional = prob / alive
    cumulative = np.cumsum(prob)

    fig = _new_figure(8.5, 4.0)
    ax = fig.subplots()
    ax.bar(times, 100 * prob, width=0.17, color=BLUE, label="P(called on this date)")
    ax.plot(times, 100 * conditional, "o-", color=ORANGE, markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.2, label="P(called on this date | still alive)")
    ax.step(times, 100 * cumulative, where="mid", color=AQUA, label="P(called on or before this date)")
    for t, v in ((times[0], prob[0]), (times[-1], cumulative[-1])):
        ax.annotate(f"{v:.0%}", (t, 100 * v), xytext=(0, 5), textcoords="offset points",
                    ha="center", color=INK_2, fontsize=8)
    ax.set_xticks(times, [f"{t:g}" for t in times])
    ax.set_ylim(0, 100)
    ax.set_xlabel("observation date (years)")
    ax.set_ylabel("probability (%)")
    ax.set_title("Early redemption of the 3-year autocall (call barrier 100%, σ = 20%)")
    ax.legend(loc="center right")
    return fig


@_styled
def fig_distributions(cfg: Config = FULL) -> Figure:
    times = np.asarray(OBS)
    red, final_perf, pv = simulate_autocall(cfg.sim_paths)
    at_maturity = red == times[-1]
    prob_called = np.array([np.mean(red == t) for t in times[:-1]])
    prob_protected = np.mean(at_maturity & (final_perf >= PROTECTION))
    prob_loss = np.mean(at_maturity & (final_perf < PROTECTION))

    fig = _new_figure(10, 3.8)
    left, right = fig.subplots(1, 2)

    w = 0.17
    left.bar(times[:-1], 100 * prob_called, width=w, color=BLUE, label="called early")
    left.bar(times[-1], 100 * prob_protected, width=w, color=AQUA, label="maturity, capital repaid")
    left.bar(times[-1], 100 * prob_loss, width=w, bottom=100 * prob_protected, color=ORANGE,
             edgecolor=SURFACE, lw=1.5, label="maturity, capital loss")
    life = red.mean()
    _ref_line(left, life, f"expected life {life:.2f} y")
    left.set_xticks(times, [f"{t:g}" for t in times])
    left.set_xlabel("redemption date (years)")
    left.set_ylabel("probability (%)")
    left.set_title("Redemption time")
    left.legend(loc="upper right", bbox_to_anchor=(1, 0.88))

    bins = np.linspace(0, max(1.3, float(pv.max()) + 0.02), 131)
    loss = pv < 0.9  # payoffs without any coupon or capital repaid lie at or above ~0.92
    right.hist(pv[~loss], bins=bins, color=BLUE, label="capital repaid (+ coupons)")
    right.hist(pv[loss], bins=bins, color=ORANGE, label="capital loss (perf < 60%)")
    right.set_yscale("log")
    right.axvline(pv.mean(), color=NEUTRAL, lw=1, ls=(0, (3, 3)))
    right.text(pv.mean(), 0.6, f"mean = price {pv.mean():.4f} ", transform=right.get_xaxis_transform(),
               color=INK_2, fontsize=8, ha="right", va="top")
    right.set_xlabel("discounted payoff (coupons + redemption, fraction of nominal)")
    right.set_ylabel("number of paths (log scale)")
    right.set_title(f"Payoff distribution ({cfg.sim_paths:,} paths)")
    right.legend(loc="upper left")
    fig.suptitle("Life and payoff of the 3-year autocall (σ = 20%)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    return fig


# ---------------------------------------------------------------------------
# 5. Greeks vs spot
# ---------------------------------------------------------------------------
@_styled
def fig_greeks_vs_spot(cfg: Config = FULL) -> Figure:
    engine = MonteCarloEngine(n_paths=cfg.greek_paths, seed=7)
    spots = np.linspace(40, 150, cfg.greek_spots)
    cases = (
        ("3 years left (12 dates)", OBS, BLUE),
        ("6 months left (2 dates)", OBS[:2], ORANGE),
    )

    fig = _new_figure(9, 5.4)
    top, bottom = fig.subplots(2, 1, sharex=True)
    for label, obs, color in cases:
        # Issued at 100 (initial level frozen), the spot moves afterwards.
        product = Autocall(obs, autocall_barrier=1.0, coupon=COUPON, memory=True,
                           protection_barrier=PROTECTION).fixed_at(np.array([S0]))
        deltas, vegas = [], []
        for s in spots:
            model = base_model(spot=float(s))
            price = engine.price(model, product).price
            deltas.append(delta_gamma(model, product, engine, 0, base_price=price)[0])
            vegas.append(vega(model, product, engine, 0))
        # dP/dS x 100: price change in % of nominal for a 1-point move of the spot (1% of S0);
        # dP/dsigma x 0.01 x 100 = dP/dsigma: price change in % of nominal per vol point.
        top.plot(spots, 100 * np.asarray(deltas), color=color, label=label)
        bottom.plot(spots, np.asarray(vegas), color=color, label=label)
    top.set_ylabel("delta\n(% of nominal per spot pt)")
    top.set_title("Autocall greeks vs spot (initial level 100, 2% quarterly coupon with memory)")
    top.legend(loc="upper right", bbox_to_anchor=(1, 0.85))
    bottom.axhline(0, color=INK_2, lw=1)
    bottom.set_ylabel("vega\n(% of nominal per vol pt)")
    bottom.set_xlabel("spot")
    for ax in (top, bottom):
        _ref_line(ax, 100 * PROTECTION, "protection 60%")
        _ref_line(ax, 100.0, "autocall 100%")
    return fig


# ---------------------------------------------------------------------------
# 6. Worst-of vs correlation
# ---------------------------------------------------------------------------
@_styled
def fig_worst_of_correlation(cfg: Config = FULL) -> Figure:
    engine = MonteCarloEngine(n_paths=cfg.corr_paths, seed=5)
    single = engine.price(base_model(), base_autocall())
    rhos = np.linspace(-0.4, 1.0, cfg.corr_points)

    fig = _new_figure(8, 4.0)
    ax = fig.subplots()
    ax.axhspan(*single.ci, color=NEUTRAL, alpha=0.18, lw=0)
    ax.axhline(single.price, color=INK_2, lw=1, ls=(0, (3, 3)), label="single-underlying autocall")
    for n, color in ((2, BLUE), (3, ORANGE)):
        res = []
        for rho in rhos:
            model = BlackScholesModel(spots=np.full(n, S0), vols=VOL, rate=RATE, dividends=DIV,
                                      correlation=constant_correlation(n, float(rho)))
            product = WorstOfAutocall(OBS, assets=tuple(range(n)), autocall_barrier=1.0, coupon=COUPON,
                                      memory=True, protection_barrier=PROTECTION)
            res.append(engine.price(model, product))
        price = np.array([r.price for r in res])
        hw = np.array([r.half_width for r in res])
        ax.fill_between(rhos, price - hw, price + hw, color=color, alpha=0.22, lw=0)
        ax.plot(rhos, price, "o-", color=color, markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.2, label=f"worst-of, {n} underlyings")
        ax.annotate(f"{n} underlyings", (rhos[0], price[0]), xytext=(4, -12), textcoords="offset points",
                    color=INK_2, fontsize=8)
    ax.set_xlabel("pairwise correlation ρ (identical underlyings, σ = 20%)")
    ax.set_ylabel("price (fraction of nominal)")
    ax.set_title("Worst-of autocall vs correlation: converges to the single-underlying price at ρ = 1")
    ax.legend(loc="lower right")
    return fig


FIGURES: dict[str, Callable[[Config], Figure]] = {
    "convergence": fig_convergence,
    "validation_quantlib": fig_validation,
    "call_probability": fig_call_probability,
    "redemption_and_payoff": fig_distributions,
    "greeks_vs_spot": fig_greeks_vs_spot,
    "worst_of_correlation": fig_worst_of_correlation,
}


def main(out_dir: Path = DEFAULT_OUT, cfg: Config = FULL, verbose: bool = True) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, build in FIGURES.items():
        start = time.perf_counter()
        fig = build(cfg)
        path = out_dir / f"{name}.png"
        # No software/date metadata: the PNGs only change when the pixels do.
        fig.savefig(path, facecolor=SURFACE, metadata={"Software": None})
        written.append(path)
        if verbose:
            print(f"{path.name}  ({time.perf_counter() - start:.1f}s)")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    parser.add_argument("--quick", action="store_true", help="few paths (smoke test), not for the README")
    args = parser.parse_args()
    main(args.out, QUICK if args.quick else FULL)
