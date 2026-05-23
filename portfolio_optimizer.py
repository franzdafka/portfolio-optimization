"""
Portfolio Optimization via Markowitz Mean-Variance Theory
=========================================================

Mathematical foundation:
  - Minimize portfolio variance: w^T Σ w
  - Subject to: w^T μ ≥ r* (return target), 1^T w = 1, w ≥ 0

Where:
  w ∈ R^n  — weight vector
  Σ ∈ R^{n×n} — covariance matrix (symmetric positive semi-definite)
  μ ∈ R^n  — expected returns vector
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds
from dataclasses import dataclass
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────

@dataclass
class Portfolio:
    weights: np.ndarray
    expected_return: float
    volatility: float      # std dev (not variance)
    sharpe_ratio: float
    assets: list[str]

    def __str__(self):
        lines = [
            f"\n{'─'*45}",
            f"  Expected Return : {self.expected_return*100:.2f}%",
            f"  Volatility      : {self.volatility*100:.2f}%",
            f"  Sharpe Ratio    : {self.sharpe_ratio:.4f}",
            f"{'─'*45}",
        ]
        lines.append(f"  {'Asset':<12} {'Weight':>10}")
        lines.append(f"  {'─'*22}")
        for asset, w in zip(self.assets, self.weights):
            lines.append(f"  {asset:<12} {w*100:>9.2f}%")
        lines.append(f"{'─'*45}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  Core optimizer
# ─────────────────────────────────────────────

class MarkowitzOptimizer:
    """
    Markowitz Mean-Variance Portfolio Optimizer.

    Given n assets with returns matrix R ∈ R^{T×n}:
      μ = E[r]            — sample mean returns
      Σ = Cov(R)          — sample covariance matrix

    We solve:
      min   w^T Σ w
      s.t.  w^T μ = r_target
            1^T w = 1
            w_i ≥ 0  (long-only)
    """

    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns (each column = one asset, each row = one period).
            Annualized or daily — keep consistent.
        risk_free_rate : float
            Annual risk-free rate for Sharpe ratio.
        """
        self.returns = returns
        self.assets = list(returns.columns)
        self.n = len(self.assets)
        self.rf = risk_free_rate

        # ── Compute statistics ──────────────────────────
        self.mu = returns.mean().values          # μ ∈ R^n
        self.Sigma = returns.cov().values        # Σ ∈ R^{n×n}
        self.Sigma_inv = np.linalg.pinv(self.Sigma)  # Σ^{-1} (pseudoinverse for stability)

        self._validate()

    def _validate(self):
        """Σ must be positive semi-definite."""
        eigvals = np.linalg.eigvalsh(self.Sigma)
        if np.any(eigvals < -1e-8):
            raise ValueError("Covariance matrix is not positive semi-definite.")

    # ── Portfolio statistics ─────────────────────────────

    def portfolio_return(self, w: np.ndarray) -> float:
        """μ_p = w^T μ"""
        return float(w @ self.mu)

    def portfolio_variance(self, w: np.ndarray) -> float:
        """σ²_p = w^T Σ w"""
        return float(w @ self.Sigma @ w)

    def portfolio_volatility(self, w: np.ndarray) -> float:
        """σ_p = √(w^T Σ w)"""
        return float(np.sqrt(self.portfolio_variance(w)))

    def sharpe_ratio(self, w: np.ndarray) -> float:
        """S = (μ_p - r_f) / σ_p"""
        vol = self.portfolio_volatility(w)
        return (self.portfolio_return(w) - self.rf) / vol if vol > 1e-10 else 0.0

    def _make_portfolio(self, w: np.ndarray) -> Portfolio:
        return Portfolio(
            weights=w,
            expected_return=self.portfolio_return(w),
            volatility=self.portfolio_volatility(w),
            sharpe_ratio=self.sharpe_ratio(w),
            assets=self.assets,
        )

    # ── Optimization problems ────────────────────────────

    def _equal_weight(self) -> np.ndarray:
        return np.ones(self.n) / self.n

    def minimize_variance(self, target_return: Optional[float] = None,
                          long_only: bool = True) -> Portfolio:
        """
        Solve:  min  w^T Σ w
                s.t. w^T μ = target_return  (if given)
                     1^T w = 1
                     w ≥ 0  (if long_only)
        """
        w0 = self._equal_weight()

        # Objective: portfolio variance
        objective = lambda w: self.portfolio_variance(w)
        grad       = lambda w: 2 * self.Sigma @ w

        constraints = [{"type": "eq",
                        "fun": lambda w: np.sum(w) - 1,
                        "jac": lambda w: np.ones(self.n)}]

        if target_return is not None:
            constraints.append({
                "type": "eq",
                "fun": lambda w: self.portfolio_return(w) - target_return,
                "jac": lambda w: self.mu,
            })

        bounds = Bounds(lb=0.0, ub=1.0) if long_only else Bounds(lb=-np.inf, ub=np.inf)

        result = minimize(
            fun=objective, jac=grad, x0=w0,
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options={"ftol": 1e-12, "maxiter": 1000},
        )

        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        w = np.clip(result.x, 0, 1)
        w /= w.sum()
        return self._make_portfolio(w)

    def maximize_sharpe(self, long_only: bool = True) -> Portfolio:
        """
        Maximize Sharpe ratio:
          max  (w^T μ - r_f) / √(w^T Σ w)

        Equivalent to: min  -S(w)  (gradient-based)

        Closed-form (unconstrained, long-short):
          w* ∝ Σ^{-1} (μ - r_f · 1)
        """
        if not long_only:
            # Closed-form Markowitz tangency portfolio
            excess = self.mu - self.rf
            w_raw  = self.Sigma_inv @ excess
            w      = w_raw / w_raw.sum()
            return self._make_portfolio(w)

        # Constrained: numerical
        w0 = self._equal_weight()
        objective = lambda w: -self.sharpe_ratio(w)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = Bounds(lb=0.0, ub=1.0)

        result = minimize(
            fun=objective, x0=w0,
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options={"ftol": 1e-12, "maxiter": 1000},
        )

        w = np.clip(result.x, 0, 1)
        w /= w.sum()
        return self._make_portfolio(w)

    def global_minimum_variance(self, long_only: bool = True) -> Portfolio:
        """
        Global Minimum Variance (GMV) portfolio — no return target.
        Closed-form (unconstrained):
          w* = Σ^{-1} 1 / (1^T Σ^{-1} 1)
        """
        if not long_only:
            ones = np.ones(self.n)
            w_raw = self.Sigma_inv @ ones
            w = w_raw / w_raw.sum()
            return self._make_portfolio(w)
        return self.minimize_variance(long_only=True)

    # ── Efficient Frontier ───────────────────────────────

    def efficient_frontier(self, n_points: int = 200,
                           long_only: bool = True) -> pd.DataFrame:
        """
        Trace the efficient frontier:
          for r* in [r_min, r_max]:
            solve min-variance subject to E[r_p] = r*

        Returns DataFrame with columns:
          return, volatility, sharpe, weights...
        """
        r_min = self.mu.min() * 1.01
        r_max = self.mu.max() * 0.99
        targets = np.linspace(r_min, r_max, n_points)

        records = []
        for r_target in targets:
            try:
                p = self.minimize_variance(target_return=r_target,
                                           long_only=long_only)
                rec = {
                    "return":     p.expected_return,
                    "volatility": p.volatility,
                    "sharpe":     p.sharpe_ratio,
                }
                for asset, w in zip(self.assets, p.weights):
                    rec[f"w_{asset}"] = w
                records.append(rec)
            except RuntimeError:
                continue

        return pd.DataFrame(records)


# ─────────────────────────────────────────────
#  Monte Carlo simulation (for comparison)
# ─────────────────────────────────────────────

def monte_carlo_portfolios(optimizer: MarkowitzOptimizer,
                           n_simulations: int = 10_000) -> pd.DataFrame:
    """
    Sample random portfolios from the Dirichlet distribution.
    Used to visualize the feasible set and compare with the efficient frontier.
    """
    rng = np.random.default_rng(42)
    records = []

    for _ in range(n_simulations):
        # Dirichlet ensures w_i > 0 and sum(w) = 1
        w = rng.dirichlet(np.ones(optimizer.n))
        records.append({
            "return":     optimizer.portfolio_return(w),
            "volatility": optimizer.portfolio_volatility(w),
            "sharpe":     optimizer.sharpe_ratio(w),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
#  Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Synthetic data: 5 assets, 5 years daily returns ──
    np.random.seed(0)
    T, n = 1260, 5
    assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    # True parameters (annualized → daily)
    annual_returns = np.array([0.18, 0.22, 0.16, 0.20, 0.30])
    annual_vols    = np.array([0.25, 0.22, 0.20, 0.28, 0.45])
    corr = np.array([
        [1.00, 0.72, 0.68, 0.60, 0.35],
        [0.72, 1.00, 0.70, 0.62, 0.38],
        [0.68, 0.70, 1.00, 0.65, 0.32],
        [0.60, 0.62, 0.65, 1.00, 0.40],
        [0.35, 0.38, 0.32, 0.40, 1.00],
    ])

    daily_mu    = annual_returns / 252
    daily_sigma = annual_vols / np.sqrt(252)
    cov_daily   = np.diag(daily_sigma) @ corr @ np.diag(daily_sigma)
    L = np.linalg.cholesky(cov_daily)

    raw = np.random.standard_normal((T, n)) @ L.T + daily_mu
    returns_df = pd.DataFrame(raw, columns=assets)

    # ── Build optimizer ──────────────────────────────────
    opt = MarkowitzOptimizer(returns_df, risk_free_rate=0.05/252)

    print("=" * 45)
    print("   MARKOWITZ PORTFOLIO OPTIMIZER")
    print("=" * 45)

    print("\n[1] Global Minimum Variance Portfolio")
    gmv = opt.global_minimum_variance()
    print(gmv)

    print("\n[2] Maximum Sharpe Ratio Portfolio (Tangency)")
    tangency = opt.maximize_sharpe()
    print(tangency)

    print("\n[3] Efficient Frontier (200 points)")
    frontier = opt.efficient_frontier(n_points=200)
    print(f"    Computed {len(frontier)} efficient portfolios")
    print(f"    Return range : {frontier['return'].min()*252*100:.1f}% – "
          f"{frontier['return'].max()*252*100:.1f}% (annualized)")
    print(f"    Vol range    : {frontier['volatility'].min()*np.sqrt(252)*100:.1f}% – "
          f"{frontier['volatility'].max()*np.sqrt(252)*100:.1f}% (annualized)")

    print("\n[4] Equal-weight benchmark")
    ew_w = np.ones(n) / n
    ew = Portfolio(
        weights=ew_w,
        expected_return=opt.portfolio_return(ew_w),
        volatility=opt.portfolio_volatility(ew_w),
        sharpe_ratio=opt.sharpe_ratio(ew_w),
        assets=assets,
    )
    print(ew)
