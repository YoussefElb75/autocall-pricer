# autocall-pricer

[![CI](https://github.com/YoussefElb75/autocall-pricer/actions/workflows/ci.yml/badge.svg)](https://github.com/YoussefElb75/autocall-pricer/actions/workflows/ci.yml)

Python library for **Monte Carlo** pricing of single-underlying **autocalls** and **worst-of autocalls**
(2 to 5 underlyings) under correlated multi-asset Black-Scholes, with bump-and-reprice greeks and
**validation against QuantLib**. Written from scratch, for educational and demonstration purposes.

## Goals

- Vectorised Monte Carlo engine (NumPy), reproducible (explicit seed, or a seed drawn at creation and
  stored in `engine.seed`), with antithetic variates and a confidence interval for the price.
- Products: autocall (early redemption, coupon with or without memory, Phoenix-style coupon barrier,
  step-down, non-call period, capital protection at maturity) and its worst-of version.
- Bump-and-reprice greeks: delta, gamma, vega per underlying, correlation sensitivity.
- Systematic validation against QuantLib on the cases that have a reliable reference.

## Model

Under the risk-neutral measure, for assets $`i = 1..n`$:

```math
\frac{dS_i(t)}{S_i(t)} = (r - q_i)\,dt + \sigma_i\,dW_i(t), \qquad d\langle W_i, W_j\rangle_t = \rho_{ij}\,dt
```

with constant $`r`$, $`q_i`$, $`\sigma_i`$, $`\rho`$. Between two grid dates, the log-normal transition is
simulated **exactly** (no discretisation bias):

```math
S_i(t_{k+1}) = S_i(t_k)\exp\!\Big(\big(r - q_i - \tfrac12\sigma_i^2\big)\Delta t_k + \sigma_i\sqrt{\Delta t_k}\,(LZ_k)_i\Big),
\qquad LL^\top = \rho,\; Z_k \sim \mathcal N(0, I_n)
```

where $`L`$ is the **Cholesky** factor of the correlation matrix when it is positive definite.
A matrix that is only positive semi-definite (correlation of 1, singular block) is accepted: $`L = V\sqrt{\max(w, 0)}`$
is then built from the eigendecomposition $`\rho = V\,\mathrm{diag}(w)\,V^\top`$.
A matrix that is not positive semi-definite is rejected.

Parameters are validated at construction (finite values, option and barrier types, indices) and stored
in copied read-only arrays: models and products are immutable, comparable and hashable by value.

**Price and confidence interval.** With $`Y_j`$ the discounted payoffs $`\sum_k e^{-rt_k}\,CF_k`$ (averaged
over each $`(Z, -Z)`$ pair when antithetic, which makes the $`Y_j`$ i.i.d.):

```math
\hat P = \frac1N\sum_j Y_j, \qquad \mathrm{SE} = \frac{s_Y}{\sqrt N}, \qquad CI_{95\%} = \hat P \pm 1.96\,\mathrm{SE}
```

**Autocall payoff.** (Worst-of) performance on date $`t_k`$:

```math
\mathrm{perf}_k = \min_i \frac{S_i(t_k)}{S_i(0)}
```

As long as the product is alive, on each date:

- coupon $`N\,c\,(1 + m)`$ if $`\mathrm{perf}_k \ge B_{\text{coupon}}`$, where $`m`$ is the number of missed coupons
  (memory effect) or $`m = 0`$ (no memory);
- early redemption ($`k`$ < last date): repayment of $`N`$ if $`\mathrm{perf}_k \ge B_{\text{call},k}`$;
- maturity: $`N`$ if $`\mathrm{perf}_T \ge B_{\text{protection}}`$ (European observation), otherwise $`N \cdot \mathrm{perf}_T`$.

By default $`B_{\text{coupon}} = B_{\text{call}}`$ with memory on, i.e. the classic "Athena / snowball" autocall.

**Greeks** (central differences, common random numbers — the same seed for every bumped price):

```math
\Delta_i = \frac{P(S_i+h)-P(S_i-h)}{2h},\quad \Gamma_i = \frac{P(S_i+h)-2P+P(S_i-h)}{h^2},\quad
\mathcal V_i = \frac{P(\sigma_i+\epsilon)-P(\sigma_i-\epsilon)}{2\epsilon},\quad
\frac{\partial P}{\partial\rho} = \frac{P(\rho+\delta)-P(\rho-\delta)}{2\delta}
```

with $`h = 1\%\cdot S_i`$, $`\epsilon = \min(1\text{ vol point},\ 10\%\cdot\sigma_i)`$ (a fixed one-point bump would
be as large as the vol itself for low vols, where the price is very convex in $`\sigma`$), $`\delta = 0.01`$ (parallel shift of all
off-diagonal correlations, or of a single pair). The product's initial levels are frozen before bumping
(otherwise an autocall issued "today" would have a zero delta by construction).

## Installation

```bash
pip install -e ".[dev]"   # library + pytest, QuantLib, matplotlib, nbconvert
```

## Example

```python
from autocall_pricer import (BlackScholesModel, MonteCarloEngine, Autocall,
                             WorstOfAutocall, compute_greeks, constant_correlation)

obs = [0.25 * k for k in range(1, 13)]                  # 3 years, quarterly
model = BlackScholesModel(spots=[100.0], vols=[0.20], rate=0.03, dividends=[0.01])
autocall = Autocall(obs, autocall_barrier=1.0, coupon=0.02, memory=True, protection_barrier=0.6)

engine = MonteCarloEngine(n_paths=200_000, seed=42, antithetic=True)
print(engine.price(model, autocall))   # 0.990685 ± 0.000509 (95% CI [...], 200000 paths)

model3 = BlackScholesModel(spots=[100, 250, 40], vols=[0.2, 0.25, 0.3], rate=0.03,
                           correlation=constant_correlation(3, 0.6))
wo = WorstOfAutocall(obs, assets=(0, 1, 2), coupon=0.025, coupon_barrier=0.7)
print(compute_greeks(model3, wo, engine).as_table())
```

The notebook [`notebooks/demo.ipynb`](notebooks/demo.ipynb) shows convergence and the CI, the effect of
antithetic variates, the price profile as a function of spot, the greeks, the correlation sensitivity and
the effect of the number of underlyings.

## Validation against QuantLib

`python -m autocall_pricer.validation` (400,000 antithetic paths, seed 2026,
$`S_0 = 100`$, $`r = 3\%`$, $`q = 1\%`$, $`\sigma = 20\%`$, $`T = 1`$ year). Criterion:
$`|\text{MC} - \text{QuantLib}| \le z\,\mathrm{SE} + \text{model tolerance}`$, with
$`z = \Phi^{-1}(1 - \alpha / 2n) \approx 3.29`$ for $`n = 10`$ cases and $`\alpha = 1\%`$ (Bonferroni: if the engine is
correct, the probability that at least one case fails is 1%, whatever the seed), the model tolerance
being 0 for exact references and 0.2% of the price for barriers (Broadie-Glasserman-Kou continuity
correction: continuous barrier shifted by $`e^{\pm 0.5826\,\sigma\sqrt{\Delta t}}`$ to approximate daily
discrete monitoring).

| Case | QuantLib reference | MC | Std error | QuantLib | Abs. diff | Tolerance | Diff / SE | OK |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| European call K=100 | AnalyticEuropeanEngine | 8.81481 | 0.01645 | 8.82732 | 0.01251 | 0.05412 | 0.76 | ✅ |
| European put K=90 | AnalyticEuropeanEngine | 2.98606 | 0.00849 | 2.99429 | 0.00823 | 0.02793 | 0.97 | ✅ |
| Digital cash-or-nothing call K=105 | AnalyticEuropeanEngine | 0.39150 | 0.00043 | 0.39171 | 0.00021 | 0.00141 | 0.48 | ✅ |
| Digital cash-or-nothing put K=80 | AnalyticEuropeanEngine | 0.12754 | 0.00048 | 0.12836 | 0.00082 | 0.00157 | 1.72 | ✅ |
| down-and-out call K=100 H=85 (252 obs.) | AnalyticBarrierEngine + BGK | 8.44346 | 0.01705 | 8.44300 | 0.00046 | 0.07299 | 0.03 | ✅ |
| up-and-out put K=100 H=120 (252 obs.) | AnalyticBarrierEngine + BGK | 6.66428 | 0.01098 | 6.66464 | 0.00037 | 0.04947 | 0.03 | ✅ |
| down-and-in put K=100 H=80 (252 obs.) | AnalyticBarrierEngine + BGK | 4.89463 | 0.01328 | 4.88653 | 0.00809 | 0.05348 | 0.61 | ✅ |
| Worst-of call 2 assets K=95 rho=0.5 | StulzEngine | 6.55927 | 0.01562 | 6.55654 | 0.00272 | 0.05141 | 0.17 | ✅ |
| Worst-of call 2 assets K=95 rho=-0.3 | StulzEngine | 2.83909 | 0.00929 | 2.84620 | 0.00711 | 0.03056 | 0.77 | ✅ |
| Autocall 1 date, cpn 8%, protection 70% | QL replication (ZC + digitals + put) | 0.99653 | 0.00010 | 0.99653 | 0.00000 | 0.00033 | 0.05 | ✅ |

The 2-asset worst-of call (Stulz formula) validates the correlation and the Cholesky factorisation. The
autocall with a single observation date is replicated exactly with QuantLib instruments ($`x = S_T/S_0`$):

```math
\text{payoff} = 1 + c\,\mathbf 1_{x \ge B_r} - (B_p - x)^+ - (1 - B_p)\,\mathbf 1_{x < B_p}
```

i.e. zero-coupon + $`c`$ digital calls − $`\mathrm{put}(K = B_p S_0)/S_0`$ − $`(1-B_p)`$ digital puts.

The tests (`pytest`; `pytest -m "not slow"` skips the longest ones) also cover: payoffs on hand-built
paths (memory, Phoenix, step-down, non-call period, European protection, worst-of), put-call and forward
parity, $`1/\sqrt N`$ error convergence, exact limit cases (always called ⇒ $`1.02\,e^{-r/4}`$; never called ⇒
zero-coupon − European down-and-in put, plus a strip of digitals for Phoenix coupons without memory),
worst-of with correlation 1 equal to the single-underlying autocall, martingale property and empirical
correlation, monotonicity (coupon, protection, memory, number of underlyings), vanilla greeks against the
Black-Scholes closed forms (`autocall_pricer.analytic`), signs and symmetry of autocall greeks, consistency
of parallel vs pairwise correlation sensitivity, input validation and typing (`mypy`).

## V1 limitations

- Black-Scholes model with constant parameters: no smile/skew (to which autocalls are very sensitive,
  through the protection barrier), no term structure of rates, dividends as a continuous yield.
- Dates as year fractions, no calendar or payment lag; coupons paid on the observation date; no issuer
  credit risk, no quanto.
- Finite-difference greeks on discontinuous payoffs: noisy gamma near the barriers, cost of
  $`1 + 4n + 2`$ pricings for $`n`$ underlyings.
- Antithetic variates barely reduce the variance of autocalls (payoff dominated by indicators).
- Validation against QuantLib limited to cases with a closed-form or quasi closed-form reference; the
  multi-date autocall has no independent reference (tested through limit cases and properties).

## Roadmap

- **V2 — C++ core with AAD**: rewrite of the engine in C++ (Python bindings), adjoint algorithmic
  differentiation to get all sensitivities in a single backward pass; payoff smoothing
  ("fuzzy barriers") for stable greeks.
- **Other volatility models**: local volatility, stochastic volatility, to capture the smile to which
  autocalls are sensitive.

## License

MIT, see [LICENSE](LICENSE).
