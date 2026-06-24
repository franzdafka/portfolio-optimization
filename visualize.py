"""
Visualization: Efficient Frontier & Portfolio Analysis
========================================================

Dashboard comparing four portfolios on the same efficient frontier:
  - Global Minimum Variance (GMV)
  - Max Sharpe / Tangency, uncapped
  - Max Sharpe, 30% concentration cap
  - Max Sharpe under Black-Litterman posterior returns

Requires: portfolio_optimizer.py (MarkowitzOptimizer, monte_carlo_portfolios)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

from portfolio_optimizer import MarkowitzOptimizer, monte_carlo_portfolios


# ── Style ────────────────────────────────────────────────────────────────────

PALETTE = {
    "bg":       "#0d1117",
    "surface":  "#161b22",
    "border":   "#21262d",
    "frontier": "#58a6ff",
    "scatter":  "#8b949e",
    "gmv":      "#3fb950",
    "tangency": "#f78166",
    "capped":   "#e3b341",
    "bl":       "#bc8cff",
    "ew":       "#d2a8ff",
    "capital":  "#ffa657",
    "text":     "#e6edf3",
    "subtext":  "#8b949e",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["surface"],
    "axes.edgecolor":    PALETTE["border"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["subtext"],
    "ytick.color":       PALETTE["subtext"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["border"],
    "grid.alpha":        0.6,
    "font.family":       "monospace",
})


def annualize(ret, vol, periods=252):
    return ret * periods, vol * np.sqrt(periods)


def plot_efficient_frontier(optimizer: MarkowitzOptimizer,
                             market_caps: np.ndarray = None,
                             view_P: np.ndarray = None,
                             view_Q: np.ndarray = None,
                             max_weight: float = 0.30,
                             save_path: str = "efficient_frontier.png",
                             title: str = "Markowitz Mean-Variance Optimization"):
    """
    Full dashboard:
      - Efficient frontier + Monte Carlo cloud (left, tall)
      - Capital Market Line
      - Four special portfolios: GMV, Tangency (uncapped/capped/BL), Equal-weight
      - Weight distribution bar chart for each special portfolio (right column)

    If market_caps / view_P / view_Q are not provided, the Black-Litterman
    panel is skipped and only GMV / Uncapped / Capped are shown.
    """
    frontier = optimizer.efficient_frontier(n_points=300)
    mc        = monte_carlo_portfolios(optimizer, n_simulations=8000)
    gmv       = optimizer.global_minimum_variance()
    tangency  = optimizer.maximize_sharpe()
    rf        = optimizer.rf

    opt_cap       = MarkowitzOptimizer(optimizer.returns, risk_free_rate=rf, max_weight=max_weight)
    tangency_cap  = opt_cap.maximize_sharpe()

    has_bl = market_caps is not None and view_P is not None and view_Q is not None
    if has_bl:
        bl_mu        = optimizer.black_litterman_posterior(market_caps=market_caps, P=view_P, Q=view_Q)
        tangency_bl  = optimizer.maximize_sharpe(long_only=True, mu=bl_mu)

    ew_w = np.ones(optimizer.n) / optimizer.n
    ew_ret = optimizer.portfolio_return(ew_w)
    ew_vol = optimizer.portfolio_volatility(ew_w)

    # Annualize everything
    f_ret, f_vol = annualize(frontier["return"].values, frontier["volatility"].values)
    mc_ret, mc_vol = annualize(mc["return"].values, mc["volatility"].values)
    gmv_r, gmv_v   = annualize(gmv.expected_return, gmv.volatility)
    tan_r, tan_v   = annualize(tangency.expected_return, tangency.volatility)
    cap_r, cap_v   = annualize(tangency_cap.expected_return, tangency_cap.volatility)
    ew_r,  ew_v    = annualize(ew_ret, ew_vol)
    rf_annual      = rf * 252
    if has_bl:
        bl_r, bl_v = annualize(tangency_bl.expected_return, tangency_bl.volatility)

    n_right_panels = 4 if has_bl else 3

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 11), facecolor=PALETTE["bg"])
    gs = GridSpec(n_right_panels, 3, figure=fig,
                  left=0.06, right=0.97, top=0.91, bottom=0.06,
                  hspace=0.55, wspace=0.38)

    ax_main = fig.add_subplot(gs[:, :2])
    ax_gmv  = fig.add_subplot(gs[0, 2])
    ax_tan  = fig.add_subplot(gs[1, 2])
    ax_cap  = fig.add_subplot(gs[2, 2])
    ax_bl   = fig.add_subplot(gs[3, 2]) if has_bl else None

    # ── Title ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.96, title, ha="center", va="center",
              fontsize=17, fontweight="bold", color=PALETTE["text"])
    subtitle = "Efficient Frontier  ·  Capital Market Line  ·  Uncapped vs Capped vs Black-Litterman"
    fig.text(0.5, 0.935, subtitle, ha="center", va="center",
              fontsize=10, color=PALETTE["subtext"])

    # ── Main plot ─────────────────────────────────────────────────────────────
    ax = ax_main
    ax.set_facecolor(PALETTE["surface"])

    sc = ax.scatter(mc_vol * 100, mc_ret * 100,
                    c=mc["sharpe"].values, cmap="viridis",
                    s=3, alpha=0.35, linewidths=0, zorder=2)

    ax.plot(f_vol * 100, f_ret * 100,
            color=PALETTE["frontier"], lw=2.5, zorder=4, label="Efficient Frontier")

    cml_vols = np.linspace(0, max(f_vol.max(), tan_v) * 1.25, 200)
    cml_rets = rf_annual + (tan_r - rf_annual) / tan_v * cml_vols
    ax.plot(cml_vols * 100, cml_rets * 100,
            color=PALETTE["capital"], lw=1.5, ls="--", zorder=3, alpha=0.9,
            label="Capital Market Line")

    def add_point(vol, ret, color, label, marker="*", size=220):
        ax.scatter(vol * 100, ret * 100, color=color,
                   s=size, marker=marker, zorder=6, edgecolors="white",
                   linewidths=0.5, label=label)
        ax.annotate(label, xy=(vol * 100, ret * 100),
                    xytext=(7, 5), textcoords="offset points",
                    color=color, fontsize=8, fontweight="bold")

    add_point(gmv_v, gmv_r, PALETTE["gmv"],      "Min Variance",        marker="D")
    add_point(tan_v, tan_r, PALETTE["tangency"], "Max Sharpe (Uncapped)", marker="*")
    add_point(cap_v, cap_r, PALETTE["capped"],   "Max Sharpe (30% Cap)",  marker="s", size=140)
    if has_bl:
        add_point(bl_v, bl_r, PALETTE["bl"],     "Max Sharpe (Black-Litterman)", marker="P", size=160)
    add_point(ew_v, ew_r, PALETTE["ew"], "Equal Weight", marker="o", size=120)

    ax.scatter(0, rf_annual * 100, color=PALETTE["capital"], s=80, marker="o", zorder=5)
    ax.annotate(f"Risk-free: {rf_annual*100:.1f}%", xy=(0, rf_annual * 100),
                xytext=(5, -12), textcoords="offset points",
                color=PALETTE["capital"], fontsize=7.5)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.025)
    cbar.set_label("Sharpe Ratio", color=PALETTE["subtext"], fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=PALETTE["subtext"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["subtext"], fontsize=7)

    ax.set_xlabel("Annualized Volatility (%)", fontsize=10)
    ax.set_ylabel("Annualized Return (%)", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.3,
              facecolor=PALETTE["bg"], edgecolor=PALETTE["border"])
    ax.grid(True, lw=0.4)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    # ── Weight bar charts ─────────────────────────────────────────────────────
    def weight_bar(ax_, weights, title_, color, stats_text):
        ax_.set_facecolor(PALETTE["surface"])
        assets = optimizer.assets
        bars = ax_.barh(assets, weights * 100, color=color, alpha=0.85,
                        edgecolor=PALETTE["bg"], linewidth=0.5)
        for bar, w in zip(bars, weights):
            ax_.text(bar.get_width() + 1.0, bar.get_y() + bar.get_height() / 2,
                     f"{w*100:.1f}%", va="center", fontsize=7.2, color=PALETTE["text"])
        ax_.set_title(title_, fontsize=9, fontweight="bold", color=color, pad=4)
        ax_.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax_.set_xlim(0, max(weights.max() * 100 * 1.35, 12))
        ax_.grid(axis="x", lw=0.3)
        ax_.tick_params(labelsize=7)
        ax_.set_xlabel(stats_text, fontsize=6.8, color=color)

    def stats_str(r, v):
        s = (r - rf_annual) / v
        return f"μ={r*100:.1f}%  σ={v*100:.1f}%  SR={s:.2f}"

    weight_bar(ax_gmv, gmv.weights,          "Min Variance",                 PALETTE["gmv"],      stats_str(gmv_r, gmv_v))
    weight_bar(ax_tan, tangency.weights,     "Max Sharpe (Uncapped)",        PALETTE["tangency"], stats_str(tan_r, tan_v))
    weight_bar(ax_cap, tangency_cap.weights, "Max Sharpe (30% Cap)",         PALETTE["capped"],   stats_str(cap_r, cap_v))
    if has_bl:
        weight_bar(ax_bl, tangency_bl.weights, "Max Sharpe (Black-Litterman)", PALETTE["bl"], stats_str(bl_r, bl_v))

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"  Saved -> {save_path}")
    return fig


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Real S&P 500 Top-5 returns (2024-01-03 -> 2025-04-30), from analysis.ipynb
    returns_df = pd.read_csv("returns.csv", index_col=0)

    opt = MarkowitzOptimizer(returns_df, risk_free_rate=0.05 / 252)

    # Market caps ($B) as of Dec 29, 2023 (last trading day before sample start)
    # Order must match returns_df.columns: AAPL, AMZN, GOOGL, MSFT, TSLA
    market_caps = np.array([2990.0, 1570.0, 1750.0, 2790.0, 789.9])

    # Illustrative view: "MSFT will outperform GOOGL by 2% annually"
    # (disclosed as a subjective view, not a market fact -- see README)
    view_P = np.array([[0, 0, -1, 1, 0]])   # AAPL, AMZN, GOOGL, MSFT, TSLA
    view_Q = np.array([0.02 / 252])

    print("Generating efficient frontier dashboard (real S&P 500 data)...")
    plot_efficient_frontier(
        opt,
        market_caps=market_caps,
        view_P=view_P,
        view_Q=view_Q,
        max_weight=0.30,
        save_path="/mnt/user-data/outputs/efficient_frontier_real.png",
        title="Markowitz Portfolio Optimization — S&P500 Top-5 (2024-2025)",
    )
    print("Done.")
