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

Also includes:
  - Concentration limits (max weight per asset)
  - Black-Litterman expected return blending, to address the
    estimation-error sensitivity documented in the README
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
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
            w_i ≥ 0           (long-only)
            w_i ≤ max_weight  (concentration limit, optional)
    """

    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02,
                 max_weight: Optional[float] = None):
        """
        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns (each column = one asset, each row = one period).
            Annualized or daily — keep consistent.
        risk_free_rate : float
            Annual risk-free rate for Sharpe ratio.
        max_weight : float, optional
            Maximum allowed weight per asset (e.g. 0.30 for a 30% cap).
            If None, no concentration limit is applied (subject only to
            long-only and budget constraints).
        """
        self.returns = returns
        self.assets = list(returns.columns)
        self.n = len(self.assets)
        self.rf = risk_free_rate
        self.max_weight = max_weight

        if max_weight is not None and max_weight * self.n < 1.0:
            raise ValueError(
                f"max_weight={max_weight} is infeasible: {self.n} assets "
                f"capped at {max_weight} can sum to at most {max_weight*self.n:.2f} < 1."
            )

        # ── Compute statistics ──────────────────────────
        self.mu = returns.mean().values          # μ ∈ R^n  (sample expected returns)
        self.Sigma = returns.cov().values        # Σ ∈ R^{n×n}
        self.Sigma_inv = np.linalg.pinv(self.Sigma)  # Σ^{-1} (pseudoinverse for stability)

        self._validate()

    def _validate(self):
        """Σ must be positive semi-definite."""
        eigvals = np.linalg.eigvalsh(self.Sigma)
        if np.any(eigvals < -1e-8):
            raise ValueError("Covariance matrix is not positive semi-definite.")

    def _weight_bounds(self, long_only: bool) -> Bounds:
        """
        Build the box constraints w_i ∈ [lb, ub] used by every numerical
        optimization in this class. Centralized here so the concentration
        limit (self.max_weight) is applied consistently everywhere.
        """
        if not long_only:
            return Bounds(lb=-np.inf, ub=np.inf)
        ub = self.max_weight if self.max_weight is not None else 1.0
        return Bounds(lb=0.0, ub=ub)

    # ── Portfolio statistics ─────────────────────────────

    def portfolio_return(self, w: np.ndarray, mu: Optional[np.ndarray] = None) -> float:
        """μ_p = w^T μ"""
        mu = self.mu if mu is None else mu
        return float(w @ mu)

    def portfolio_variance(self, w: np.ndarray) -> float:
        """σ²_p = w^T Σ w"""
        return float(w @ self.Sigma @ w)

    def portfolio_volatility(self, w: np.ndarray) -> float:
        """σ_p = √(w^T Σ w)"""
        return float(np.sqrt(self.portfolio_variance(w)))

    def sharpe_ratio(self, w: np.ndarray, mu: Optional[np.ndarray] = None) -> float:
        """S = (μ_p - r_f) / σ_p"""
        vol = self.portfolio_volatility(w)
        ret = self.portfolio_return(w, mu=mu)
        return (ret - self.rf) / vol if vol > 1e-10 else 0.0

    def _make_portfolio(self, w: np.ndarray, mu: Optional[np.ndarray] = None) -> Portfolio:
        return Portfolio(
            weights=w,
            expected_return=self.portfolio_return(w, mu=mu),
            volatility=self.portfolio_volatility(w),
            sharpe_ratio=self.sharpe_ratio(w, mu=mu),
            assets=self.assets,
        )

    # ── Optimization problems ────────────────────────────

    def _equal_weight(self) -> np.ndarray:
        return np.ones(self.n) / self.n

    def minimize_variance(self, target_return: Optional[float] = None,
                          long_only: bool = True,
                          mu: Optional[np.ndarray] = None) -> Portfolio:
        """
        Solve:  min  w^T Σ w
                s.t. w^T μ = target_return  (if given)
                     1^T w = 1
                     w ≥ 0              (if long_only)
                     w ≤ max_weight     (if self.max_weight is set)

        Parameters
        ----------
        mu : np.ndarray, optional
            Expected returns to use for the return constraint. Defaults to
            the sample mean (self.mu). Pass a Black-Litterman posterior here
            to optimize against blended views instead of raw sample means.
        """
        mu_eff = self.mu if mu is None else mu
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
                "fun": lambda w: self.portfolio_return(w, mu=mu_eff) - target_return,
                "jac": lambda w: mu_eff,
            })

        bounds = self._weight_bounds(long_only)

        result = minimize(
            fun=objective, jac=grad, x0=w0,
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options={"ftol": 1e-12, "maxiter": 1000},
        )

        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        w = np.clip(result.x, 0, bounds.ub if np.isscalar(bounds.ub) else 1.0)
        w /= w.sum()
        return self._make_portfolio(w, mu=mu_eff)

    def maximize_sharpe(self, long_only: bool = True,
                        mu: Optional[np.ndarray] = None) -> Portfolio:
        """
        Maximize Sharpe ratio:
          max  (w^T μ - r_f) / √(w^T Σ w)

        Equivalent to: min  -S(w)  (gradient-based)

        Closed-form (unconstrained, long-short, no concentration limit):
          w* ∝ Σ^{-1} (μ - r_f · 1)

        Parameters
        ----------
        mu : np.ndarray, optional
            Expected returns to use. Defaults to the sample mean.
            Pass Black-Litterman posterior returns here to get the
            tangency portfolio under blended views instead of raw
            historical means.

        Note: the closed-form solution ignores `self.max_weight` — a
        concentration limit is a constraint that has no closed-form
        unconstrained solution, so when `self.max_weight` is set this
        method always falls back to the numerical (long_only) solver,
        even if long_only=False was requested.
        """
        mu_eff = self.mu if mu is None else mu

        if not long_only and self.max_weight is None:
            # Closed-form Markowitz tangency portfolio
            excess = mu_eff - self.rf
            w_raw  = self.Sigma_inv @ excess
            w      = w_raw / w_raw.sum()
            return self._make_portfolio(w, mu=mu_eff)

        # Constrained: numerical
        w0 = self._equal_weight()
        objective = lambda w: -self.sharpe_ratio(w, mu=mu_eff)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = self._weight_bounds(long_only=True)

        result = minimize(
            fun=objective, x0=w0,
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options={"ftol": 1e-12, "maxiter": 1000},
        )

        w = np.clip(result.x, 0, bounds.ub if np.isscalar(bounds.ub) else 1.0)
        w /= w.sum()
        return self._make_portfolio(w, mu=mu_eff)

    def global_minimum_variance(self, long_only: bool = True) -> Portfolio:
        """
        Global Minimum Variance (GMV) portfolio — no return target.
        Closed-form (unconstrained, no concentration limit):
          w* = Σ^{-1} 1 / (1^T Σ^{-1} 1)
        """
        if not long_only and self.max_weight is None:
            ones = np.ones(self.n)
            w_raw = self.Sigma_inv @ ones
            w = w_raw / w_raw.sum()
            return self._make_portfolio(w)
        return self.minimize_variance(long_only=True)

    # ── Efficient Frontier ───────────────────────────────

    def efficient_frontier(self, n_points: int = 200,
                           long_only: bool = True,
                           mu: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Trace the efficient frontier:
          for r* in [r_min, r_max]:
            solve min-variance subject to E[r_p] = r*

        Returns DataFrame with columns:
          return, volatility, sharpe, weights...
        """
        mu_eff = self.mu if mu is None else mu
        r_min = mu_eff.min() * 1.01
        r_max = mu_eff.max() * 0.99
        targets = np.linspace(r_min, r_max, n_points)

        records = []
        for r_target in targets:
            try:
                p = self.minimize_variance(target_return=r_target,
                                           long_only=long_only, mu=mu_eff)
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

    # ── Black-Litterman ──────────────────────────────────

    def black_litterman_posterior(
        self,
        market_caps: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        tau: float = 0.05,
        view_confidence: Optional[np.ndarray] = None,
        delta: float = 2.5,
    ) -> np.ndarray:
        """
        Compute Black-Litterman posterior expected returns.

        Rather than treating the sample mean μ as the "true" expected
        return — the assumption that drives Markowitz overconcentration —
        Black-Litterman starts from market-implied equilibrium returns and
        blends them with explicit investor views, weighted by confidence
        in each.

        Step 1 — Reverse optimization: implied equilibrium returns Π from
        market-cap weights w_mkt (the CAPM-consistent returns that would
        make the market-cap-weighted portfolio mean-variance optimal):

            Π = δ · Σ · w_mkt

        where δ (delta) is the market risk-aversion coefficient.

        Step 2 — Blend Π with views (P, Q) via:

            E[R]_BL = [(τΣ)^-1 + P^T Ω^-1 P]^-1 [(τΣ)^-1 Π + P^T Ω^-1 Q]

        where:
          P ∈ R^{k×n}  — view matrix (each row picks out the asset(s) a view is about)
          Q ∈ R^k      — view returns (the values asserted by each view)
          Ω ∈ R^{k×k}  — diagonal matrix of view uncertainty (lower = more confident)
          τ            — scalar reflecting uncertainty in the prior (Π)

        Parameters
        ----------
        market_caps : np.ndarray
            Market capitalization for each asset, used to derive the
            market-cap-weighted prior portfolio w_mkt = market_caps / sum(market_caps).
        P : np.ndarray, shape (k, n)
            View matrix. E.g. for a single absolute view on asset i,
            P = [[0, ..., 1, ..., 0]] with 1 in position i.
            For a relative view ("asset i will outperform asset j by Q"),
            the row has +1 at position i and -1 at position j.
        Q : np.ndarray, shape (k,)
            View return values, one per row of P.
        tau : float, default 0.05
            Scalar controlling how much weight is placed on the prior
            relative to the views. Common range: 0.01–0.05.
        view_confidence : np.ndarray, shape (k,), optional
            Diagonal entries of Ω (variance of each view — lower means more
            confident). If None, defaults to diag(P (τΣ) P^T), the standard
            He-Litterman convention, which scales view uncertainty to the
            prior's own uncertainty about that view.
        delta : float, default 2.5
            Market risk-aversion coefficient used to back out Π from
            market-cap weights. 2.5 is a commonly used default in the
            Black-Litterman literature (Idzorek, 2005).

        Returns
        -------
        np.ndarray, shape (n,)
            Posterior expected returns E[R]_BL, usable directly in place
            of self.mu when calling minimize_variance / maximize_sharpe /
            efficient_frontier via their `mu=` argument.
        """
        w_mkt = market_caps / market_caps.sum()
        Pi = delta * self.Sigma @ w_mkt  # implied equilibrium returns

        tau_sigma = tau * self.Sigma
        tau_sigma_inv = np.linalg.pinv(tau_sigma)

        if view_confidence is None:
            # He-Litterman convention: Omega scales with the prior's own
            # uncertainty about each view, so confident structural views
            # don't get arbitrarily overridden by a noisy prior.
            omega_diag = np.diag(P @ tau_sigma @ P.T)
            Omega = np.diag(omega_diag)
        else:
            Omega = np.diag(view_confidence)

        Omega_inv = np.linalg.pinv(Omega)

        A = tau_sigma_inv + P.T @ Omega_inv @ P
        b = tau_sigma_inv @ Pi + P.T @ Omega_inv @ Q

        posterior_mu = np.linalg.solve(A, b)
        return posterior_mu


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

    # ── Build optimizer (no concentration limit) ────────
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

    # ── [5] Max Sharpe with a 30% concentration limit ────
    print("\n[5] Maximum Sharpe Ratio Portfolio — 30% cap per asset")
    opt_capped = MarkowitzOptimizer(returns_df, risk_free_rate=0.05/252, max_weight=0.30)
    tangency_capped = opt_capped.maximize_sharpe(long_only=True)
    print(tangency_capped)
    print(f"    Uncapped max weight: {tangency.weights.max()*100:.1f}%  ->  "
          f"Capped max weight: {tangency_capped.weights.max()*100:.1f}%")

    # ── [6] Black-Litterman ───────────────────────────────
    print("\n[6] Black-Litterman posterior returns")
    # Illustrative market caps (not real data) just to demonstrate the API
    market_caps = np.array([3.2, 3.1, 2.1, 2.0, 1.1])  # in $T, roughly AAPL/MSFT/GOOGL/AMZN/TSLA scale
    # Single relative view: "AAPL will outperform GOOGL by 3% annually"
    P = np.array([[1, 0, -1, 0, 0]])  # AAPL - GOOGL
    Q = np.array([0.03 / 252])         # daily-scale, matching daily_mu units

    bl_mu = opt.black_litterman_posterior(market_caps=market_caps, P=P, Q=Q, tau=0.05)
    print("    Sample mean μ      :", np.round(opt.mu * 252, 4))
    print("    BL posterior μ     :", np.round(bl_mu * 252, 4))

    tangency_bl = opt.maximize_sharpe(long_only=True, mu=bl_mu)
    print("\n    Tangency portfolio under BL posterior:")
    print(tangency_bl)
    print(f"    Sample-mean tangency max weight : {tangency.weights.max()*100:.1f}%")
    print(f"    BL-posterior tangency max weight: {tangency_bl.weights.max()*100:.1f}%")
