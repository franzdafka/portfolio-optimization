# Portfolio Optimization — Markowitz Mean-Variance Theory

Python implementation of Markowitz mean-variance optimization: efficient frontier, GMV and tangency portfolios, and Monte Carlo feasible set — validated on real S&P 500 data (2024–2025).

![Efficient Frontier — Real Data](efficient_frontier_real.png)

---

## Results (Real Data — S&P 500 Top-5, 2024–2025)

| Portfolio | Annual Return | Volatility | Sharpe |
|-----------|--------------|------------|--------|
| Min Variance | 7.3% | 22.6% | 0.10 |
| Max Sharpe (Tangency) | 15.4% | 30.9% | 0.34 |
| Equal Weight | ~10.5% | ~35.0% | ~0.15 |

**Max Sharpe concentrates 93.5% in MSFT, 6.5% in AAPL** — replicating the known Markowitz overconcentration problem (Michaud, 1989): a 0.5% perturbation in input returns produces 50pp+ weight reallocation across the frontier.

Risk-free rate: 5.0% (US T-bill). Solver: `scipy.optimize.minimize` SLSQP.

---

## Mathematical Foundation

### Setup

Given $n$ assets and $T$ observations, we estimate:

$$\mu = \frac{1}{T} R^\top \mathbf{1} \in \mathbb{R}^n \qquad \Sigma = \frac{1}{T-1}(R - \mathbf{1}\mu^\top)^\top(R - \mathbf{1}\mu^\top) \in \mathbb{R}^{n \times n}$$

$\Sigma$ is symmetric positive semi-definite: $\forall w,\ w^\top \Sigma w \geq 0$.

### Portfolio Statistics

| Quantity | Formula |
|---|---|
| Expected return | $\mu_p = w^\top \mu$ |
| Variance | $\sigma_p^2 = w^\top \Sigma w$ |
| Sharpe ratio | $S = (\mu_p - r_f) / \sigma_p$ |

### Optimization Problems

**Minimum Variance** (for target return $r^*$):

$$\min_{w}\ w^\top \Sigma w \quad \text{s.t.}\ w^\top \mu = r^*,\ \mathbf{1}^\top w = 1,\ w \geq 0$$

Quadratic program; solved numerically via SLSQP.

**Global Minimum Variance** — closed-form (unconstrained):

$$w^*_{GMV} = \frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^\top \Sigma^{-1} \mathbf{1}}$$

Derived from Lagrangian $\mathcal{L} = w^\top \Sigma w - \lambda(\mathbf{1}^\top w - 1)$; FOC gives $2\Sigma w = \lambda \mathbf{1}$, normalized by $\mathbf{1}^\top w = 1$.

**Tangency Portfolio (Max Sharpe)** — closed-form (unconstrained):

$$w^*_{tan} \propto \Sigma^{-1}(\mu - r_f \mathbf{1})$$

Tangent point of the Capital Market Line to the efficient frontier:

$$\mu_p = r_f + \frac{\mu_{tan} - r_f}{\sigma_{tan}} \cdot \sigma_p$$

### Efficient Frontier

Solving the min-variance QP across $r^* \in [\mu_{min},\ \mu_{max}]$ traces the efficient frontier — a **hyperbola** in $(\sigma, \mu)$ space. The 8,000-point Monte Carlo cloud (Dirichlet-sampled) visualises the full feasible set.

### Two-Fund Separation

Any efficient portfolio is a linear combination of GMV and tangency:

$$w^* = \alpha\, w_{GMV} + (1-\alpha)\, w_{tan}, \qquad \alpha \in \mathbb{R}$$

### Sensitivity Analysis

One of the most striking findings: a 0.5% annual change in expected 
returns — well within normal estimation error — can completely reshuffle 
the optimal weights.

I ran this systematically: perturbed each asset's μ by ±0.5% and tracked 
how the tangency portfolio reacted. Some assets caused 50%+ weight shifts 
from
---

## Markowitz Instability

The optimizer is hyper-sensitive to input estimates. On real S&P 500 data:

- **Min Variance**: 66.9% AMZN, 19.7% AAPL, 13.4% GOOGL
- **Max Sharpe**: 93.5% MSFT, 6.5% AAPL

A 0.5% perturbation in $\mu$ produces 50pp+ weight reallocation — replicating Michaud (1989). This is estimation error amplification, not a solver bug. Practitioners address it via Black-Litterman, resampling, or hard weight caps (e.g. $w_i \leq 30\%$). The long-only constraint ($w \geq 0$) is implemented; concentration limits are a natural extension.

---

## Project Structure

```
portfolio-optimization/
├── portfolio_optimizer.py   — MarkowitzOptimizer class (GMV, Tangency, Frontier)
├── visualize.py             — Efficient frontier dashboard (Matplotlib)
├── analysis.ipynb           — Real data analysis (yfinance, S&P 500 2024–2025)
├── efficient_frontier.png   — Synthetic data output
├── efficient_frontier_real.png — Real market data output
└── requirements.txt
```

---

## Usage

```python
import pandas as pd
from portfolio_optimizer import MarkowitzOptimizer

# Daily returns DataFrame (rows = dates, columns = tickers)
returns = pd.read_csv("returns.csv", index_col=0)

opt = MarkowitzOptimizer(returns, risk_free_rate=0.05/252)

gmv      = opt.global_minimum_variance()   # Min variance
tangency = opt.maximize_sharpe()           # Max Sharpe
frontier = opt.efficient_frontier(n_points=300)
```

```bash
python visualize.py   # generates efficient_frontier.png
```

---

## API

### `MarkowitzOptimizer(returns, risk_free_rate=0.02)`

| Method | Description |
|---|---|
| `minimize_variance(target_return, long_only)` | Min-variance QP for given $r^*$ |
| `maximize_sharpe(long_only)` | Tangency portfolio |
| `global_minimum_variance(long_only)` | GMV portfolio |
| `efficient_frontier(n_points, long_only)` | Full frontier (DataFrame) |
| `portfolio_return(w)` | $w^\top \mu$ |
| `portfolio_volatility(w)` | $\sqrt{w^\top \Sigma w}$ |
| `sharpe_ratio(w)` | $(w^\top\mu - r_f)/\sigma_p$ |

### `monte_carlo_portfolios(optimizer, n_simulations=10_000)`

Samples random portfolios from $\text{Dirichlet}(\mathbf{1})$ for feasible set visualization.

---

## Implementation Notes

| Detail | Choice |
|---|---|
| Solver | `scipy.optimize.minimize`, `method="SLSQP"` |
| Covariance | Sample covariance matrix |
| Stability | `numpy.linalg.pinv` (pseudoinverse) for near-singular $\Sigma$ |
| Long-only | $w_i \geq 0\ \forall i$ (toggleable) |
| Monte Carlo | $\text{Dirichlet}(\mathbf{1})$ ensures $w_i > 0$, $\sum w_i = 1$ |

---

## Linear Algebra Concepts

| Concept | Role in Implementation |
|---|---|
| Positive semi-definite matrix | $\Sigma$ validity check: $w^\top \Sigma w \geq 0$ |
| Pseudoinverse | Closed-form GMV and tangency under near-singularity |
| Cholesky decomposition | Correlated return simulation: $R = ZL^\top$, $\Sigma = LL^\top$ |
| Quadratic form | Portfolio variance $\sigma_p^2 = w^\top \Sigma w$ |
| KKT conditions | Optimality conditions for the constrained QP |
| Lagrange multipliers | Closed-form derivation of GMV and tangency |

---

## References

- Markowitz, H. (1952). *Portfolio Selection*. **Journal of Finance**, 7(1), 77–91.
- Michaud, R. (1989). *The Markowitz Optimization Enigma: Is Optimized Optimal?* **Financial Analysts Journal**.
- Merton, R. (1972). *An Analytic Derivation of the Efficient Portfolio Frontier*. **JFQA**.
- Boyd & Vandenberghe. *Convex Optimization*. Cambridge University Press (Ch. 4, 7).
