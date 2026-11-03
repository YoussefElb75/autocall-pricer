# autocall-pricer

[![CI](https://github.com/YoussefElb75/autocall-pricer/actions/workflows/ci.yml/badge.svg)](https://github.com/YoussefElb75/autocall-pricer/actions/workflows/ci.yml)

Librairie Python de pricing **Monte Carlo** pour **autocalls** mono-sous-jacent et **worst-of autocalls**
(2 à 5 sous-jacents) sous Black-Scholes multi-actifs corrélé, avec grecques par perturbation et
**validation contre QuantLib**. Code écrit de zéro, à but pédagogique et de démonstration.

## Objectif

- Moteur Monte Carlo vectorisé (NumPy), reproductible (graine explicite, ou graine tirée à la création
  et conservée dans `engine.seed`), avec variables antithétiques et intervalle de confiance du prix.
- Produits : autocall (rappel, coupon avec ou sans mémoire, barrière de coupon type Phoenix, step-down,
  période de non-rappel, protection du capital à maturité) et sa version worst-of.
- Grecques par bump : delta, gamma, vega par sous-jacent, sensibilité à la corrélation.
- Validation systématique contre QuantLib sur les cas qui ont une référence fiable.

## Modèle

Sous la probabilité risque-neutre, pour les actifs $i = 1..n$ :

$$\frac{dS_i(t)}{S_i(t)} = (r - q_i)\,dt + \sigma_i\,dW_i(t), \qquad d\langle W_i, W_j\rangle_t = \rho_{ij}\,dt$$

avec $r$, $q_i$, $\sigma_i$, $\rho$ constants. Entre deux dates de la grille, la transition log-normale est
simulée **exactement** (pas de biais de discrétisation) :

$$S_i(t_{k+1}) = S_i(t_k)\exp\!\Big(\big(r - q_i - \tfrac12\sigma_i^2\big)\Delta t_k + \sigma_i\sqrt{\Delta t_k}\,(LZ_k)_i\Big),
\qquad LL^\top = \rho,\; Z_k \sim \mathcal N(0, I_n)$$

où $L$ est la factorisation de **Cholesky** de la matrice de corrélation lorsqu'elle est définie positive.
Une matrice seulement semi-définie positive (corrélation de 1, bloc singulier) est acceptée : $L = V\sqrt{\max(w, 0)}$
est alors construit à partir de la décomposition en valeurs propres $\rho = V\,\mathrm{diag}(w)\,V^\top$.
Une matrice non semi-définie positive est rejetée.

Les paramètres sont validés à la construction (valeurs finies, types d'option et de barrière, indices) et
stockés dans des tableaux copiés en lecture seule : modèles et produits sont immuables, comparables et
hachables par valeur.

**Prix et intervalle de confiance.** Avec $Y_j$ les payoffs actualisés $\sum_k e^{-rt_k}\,CF_k$ (moyennés
par paire $(Z, -Z)$ en antithétique, ce qui rend les $Y_j$ i.i.d.) :

$$\hat P = \frac1N\sum_j Y_j, \qquad \mathrm{SE} = \frac{s_Y}{\sqrt N}, \qquad IC_{95\%} = \hat P \pm 1{,}96\,\mathrm{SE}$$

**Payoff de l'autocall.** Performance (worst-of) à la date $t_k$ : $\mathrm{perf}_k = \min_i S_i(t_k)/S_i(0)$.
Tant que le produit est vivant, à chaque date :

- coupon $N\,c\,(1 + m)$ si $\mathrm{perf}_k \ge B_{\text{coupon}}$, avec $m$ le nombre de coupons manqués
  (effet mémoire) ou $m = 0$ (sans mémoire) ;
- rappel anticipé ($k$ < dernière date) : remboursement de $N$ si $\mathrm{perf}_k \ge B_{\text{rappel},k}$ ;
- maturité : $N$ si $\mathrm{perf}_T \ge B_{\text{protection}}$ (observation européenne), sinon $N\cdot\mathrm{perf}_T$.

Par défaut $B_{\text{coupon}} = B_{\text{rappel}}$ et mémoire activée, soit l'autocall « Athena / snowball » classique.

**Grecques** (différences centrées, nombres aléatoires communs — même graine pour tous les prix bumpés) :

$$\Delta_i = \frac{P(S_i+h)-P(S_i-h)}{2h},\quad \Gamma_i = \frac{P(S_i+h)-2P+P(S_i-h)}{h^2},\quad
\mathcal V_i = \frac{P(\sigma_i+\epsilon)-P(\sigma_i-\epsilon)}{2\epsilon},\quad
\frac{\partial P}{\partial\rho} = \frac{P(\rho+\delta)-P(\rho-\delta)}{2\delta}$$

avec $h = 1\%\cdot S_i$, $\epsilon = \min(1\text{ point de vol},\ 10\%\cdot\sigma_i)$ (un choc fixe d'un point serait
aussi grand que la vol elle-même pour les vols faibles, où le prix est très convexe en $\sigma$), $\delta = 0{,}01$ (décalage parallèle de toutes les
corrélations hors diagonale, ou d'une seule paire). Les niveaux initiaux du produit sont figés avant le bump
(sinon un autocall émis « aujourd'hui » aurait un delta nul par construction).

## Installation

```bash
pip install -e ".[dev]"   # librairie + pytest, QuantLib, matplotlib, nbconvert
```

## Exemple

```python
from autocall_pricer import (BlackScholesModel, MonteCarloEngine, Autocall,
                             WorstOfAutocall, compute_greeks, constant_correlation)

obs = [0.25 * k for k in range(1, 13)]                  # 3 ans, trimestriel
model = BlackScholesModel(spots=[100.0], vols=[0.20], rate=0.03, dividends=[0.01])
autocall = Autocall(obs, autocall_barrier=1.0, coupon=0.02, memory=True, protection_barrier=0.6)

engine = MonteCarloEngine(n_paths=200_000, seed=42, antithetic=True)
print(engine.price(model, autocall))   # 0.990685 ± 0.000509 (95% CI [...], 200000 paths)

model3 = BlackScholesModel(spots=[100, 250, 40], vols=[0.2, 0.25, 0.3], rate=0.03,
                           correlation=constant_correlation(3, 0.6))
wo = WorstOfAutocall(obs, assets=(0, 1, 2), coupon=0.025, coupon_barrier=0.7)
print(compute_greeks(model3, wo, engine).as_table())
```

Le notebook [`notebooks/demo.ipynb`](notebooks/demo.ipynb) montre la convergence et l'IC, l'effet des
antithétiques, le profil de prix en fonction du spot, les grecques, la sensibilité à la corrélation et
l'effet du nombre de sous-jacents.

## Validation contre QuantLib

`python -m autocall_pricer.validation` (400 000 chemins antithétiques, graine 2026,
$S_0 = 100$, $r = 3\%$, $q = 1\%$, $\sigma = 20\%$, $T = 1$ an). Critère :
$|\text{MC} - \text{QuantLib}| \le z\,\mathrm{SE} + \text{tolérance de modèle}$, avec
$z = \Phi^{-1}(1 - \alpha / 2n) \approx 3{,}29$ pour $n = 10$ cas et $\alpha = 1\%$ (Bonferroni : si le moteur est
juste, la probabilité qu'un cas au moins échoue vaut 1 %, quelle que soit la graine), la tolérance de modèle
valant 0 pour les références exactes et 0,2 % du prix pour les barrières (correction de continuité
Broadie-Glasserman-Kou : barrière continue décalée de $e^{\pm 0{,}5826\,\sigma\sqrt{\Delta t}}$ pour approcher la
surveillance discrète quotidienne).

| Cas | Référence QuantLib | MC | Erreur std | QuantLib | Écart abs. | Tolérance | Écart / SE | OK |
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

Le worst-of call à 2 actifs (formule de Stulz) valide la corrélation et la factorisation de Cholesky. L'autocall
à une seule date d'observation se réplique exactement avec des instruments QuantLib ($x = S_T/S_0$) :

$$\text{payoff} = 1 + c\,\mathbf 1_{x \ge B_r} - (B_p - x)^+ - (1 - B_p)\,\mathbf 1_{x < B_p}$$

soit zéro-coupon + $c$ digitales call − put($K = B_pS_0$)$/S_0$ − $(1-B_p)$ digitales put.

Les tests (`pytest`, `pytest -m "not slow"` pour sauter les plus longs) couvrent aussi : payoffs sur
chemins construits à la main (mémoire, Phoenix, step-down, non-rappel, protection européenne, worst-of),
parité call-put et forward, convergence de l'erreur en $1/\sqrt N$, cas limites exacts (toujours rappelé ⇒
$1{,}02\,e^{-r/4}$ ; jamais rappelé ⇒ zéro-coupon − put down-and-in européen, plus une bande de digitales pour
les coupons Phoenix sans mémoire), worst-of à corrélation 1 égal au mono-sous-jacent, martingalité et
corrélation empirique, monotonies (coupon, protection, mémoire, nombre de sous-jacents), grecques vanilles
contre les formules fermées de Black-Scholes (`autocall_pricer.analytic`), signes et symétrie des grecques
d'autocall, cohérence corrélation parallèle / par paire, validation des entrées et typage (`mypy`).

## Limites de la V1

- Modèle Black-Scholes à paramètres constants : pas de smile/skew (or les autocalls y sont très sensibles,
  via la barrière de protection), pas de structure par terme des taux, dividendes en rendement continu.
- Dates en fractions d'année, pas de calendrier ni de décalage de paiement ; coupons payés à la date
  d'observation ; pas de risque de crédit de l'émetteur, pas de quanto.
- Grecques par différences finies sur payoffs discontinus : gamma bruité près des barrières, coût
  en $1 + 4n + 2$ pricings pour $n$ sous-jacents.
- Les antithétiques réduisent peu la variance des autocalls (payoff dominé par des indicatrices).
- Validation contre QuantLib limitée aux cas ayant une référence fermée ou quasi fermée ; l'autocall
  multi-dates n'a pas de référence indépendante (testé par cas limites et propriétés).

## Feuille de route

- **V2 — cœur C++ avec AAD** : réécriture du moteur en C++ (bindings Python), différentiation automatique
  adjointe pour obtenir toutes les sensibilités en un seul passage arrière ; lissage des payoffs
  (« fuzzy barriers ») pour des grecques stables.
- **Autres modèles de volatilité** : volatilité locale, volatilité stochastique, pour capturer le smile auquel
  les autocalls sont sensibles.

## Licence

MIT, voir [LICENSE](LICENSE).
