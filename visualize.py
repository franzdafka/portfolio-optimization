"""
Visualization: Efficient Frontier & Portfolio Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

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
                             save_path: str = "efficient_frontier.png"):
    """
    Full dashboard:
      - Efficient frontier + Monte Carlo cloud
      - Capital Market Line
      - Special portfolios: GMV, Tangency, Equal-weight
      - Weight distribution (bar chart) for key portfolios
    """
    frontier   = optimizer.efficient_frontier(n_points=300)
    mc         = monte_carlo_portfolios(optimizer, n_simulations=8000)
    gmv        = optimizer.global_minimum_variance()
    tangency   = optimizer.maximize_sharpe()
    rf         = optimizer.rf

    ew_w = np.ones(optimizer.n) / optimizer.n
    ew_ret = optimizer.portfolio_return(ew_w)
    ew_vol = optimizer.portfolio_volatility(ew_w)

    # Annualize everything
    f_ret, f_vol = annualize(frontier["return"].values,
                              frontier["volatility"].values)
    mc_ret, mc_vol = annualize(mc["return"].values, mc["volatility"].values)
    gmv_r, gmv_v   = annualize(gmv.expected_return, gmv.volatility)
    tan_r, tan_v   = annualize(tangency.expected_return, tangency.volatility)
    ew_r,  ew_v    = annualize(ew_ret, ew_vol)
    rf_annual      = rf * 252

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor=PALETTE["bg"])
    gs = GridSpec(2, 3, figure=fig,
                  left=0.07, right=0.97, top=0.90, bottom=0.08,
                  hspace=0.42, wspace=0.38)

    ax_main  = fig.add_subplot(gs[:, :2])   # Frontier (left, tall)
    ax_gmv   = fig.add_subplot(gs[0, 2])    # GMV weights
    ax_tan   = fig.add_subplot(gs[1, 2])    # Tangency weights

    # ── Title ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.95, "Markowitz Mean-Variance Optimization",
             ha="center", va="center", fontsize=17, fontweight="bold",
             color=PALETTE["text"])
    fig.text(0.5, 0.92, "Efficient Frontier  ·  Capital Market Line  ·  Key Portfolios",
             ha="center", va="center", fontsize=10, color=PALETTE["subtext"])

    # ── Main plot ─────────────────────────────────────────────────────────────
    ax = ax_main
    ax.set_facecolor(PALETTE["surface"])

    # Monte Carlo cloud
    sc = ax.scatter(mc_vol * 100, mc_ret * 100,
                    c=mc["sharpe"].values, cmap="viridis",
                    s=3, alpha=0.35, linewidths=0, zorder=2)

    # Efficient frontier
    ax.plot(f_vol * 100, f_ret * 100,
            color=PALETTE["frontier"], lw=2.5, zorder=4,
            label="Efficient Frontier")

    # Capital Market Line
    cml_vols = np.linspace(0, max(f_vol.max(), tan_v) * 1.25, 200)
    cml_rets = rf_annual + (tan_r - rf_annual) / tan_v * cml_vols
    ax.plot(cml_vols * 100, cml_rets * 100,
            color=PALETTE["capital"], lw=1.5, ls="--", zorder=3, alpha=0.9,
            label="Capital Market Line")

    # Special portfolios
    def add_point(ax, vol, ret, color, label, marker="*", size=220):
        ax.scatter(vol * 100, ret * 100, color=color,
                   s=size, marker=marker, zorder=6, edgecolors="white",
                   linewidths=0.5, label=label)
        ax.annotate(label,
                    xy=(vol * 100, ret * 100),
                    xytext=(7, 5), textcoords="offset points",
                    color=color, fontsize=8.5, fontweight="bold")

    add_point(ax, gmv_v, gmv_r,  PALETTE["gmv"],      "Min Variance", marker="D")
    add_point(ax, tan_v, tan_r,  PALETTE["tangency"],  "Max Sharpe",  marker="*")
    add_point(ax, ew_v,  ew_r,   PALETTE["ew"],        "Equal Weight", marker="o", size=140)

    # Risk-free rate dot on y-axis
    ax.scatter(0, rf_annual * 100, color=PALETTE["capital"], s=80,
               marker="o", zorder=5)
    ax.annotate(f"Risk-free: {rf_annual*100:.1f}%",
                xy=(0, rf_annual * 100), xytext=(5, -12),
                textcoords="offset points", color=PALETTE["capital"], fontsize=7.5)

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.025)
    cbar.set_label("Sharpe Ratio", color=PALETTE["subtext"], fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=PALETTE["subtext"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["subtext"], fontsize=7)

    ax.set_xlabel("Annualized Volatility (%)", fontsize=10)
    ax.set_ylabel("Annualized Return (%)", fontsize=10)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.3,
              facecolor=PALETTE["bg"], edgecolor=PALETTE["border"])
    ax.grid(True, lw=0.4)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    # ── Weight charts ─────────────────────────────────────────────────────────
    def weight_bar(ax_, weights, title, color):
        ax_.set_facecolor(PALETTE["surface"])
        assets = optimizer.assets
        bars = ax_.barh(assets, weights * 100,
                        color=color, alpha=0.80, edgecolor=PALETTE["bg"], linewidth=0.5)
        for bar, w in zip(bars, weights):
            ax_.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{w*100:.1f}%", va="center", fontsize=7.5,
                     color=PALETTE["text"])
        ax_.set_title(title, fontsize=9, fontweight="bold",
                      color=color, pad=6)
        ax_.set_xlabel("Weight (%)", fontsize=7.5)
        ax_.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax_.set_xlim(0, max(weights.max() * 100 * 1.3, 10))
        ax_.grid(axis="x", lw=0.3)
        ax_.tick_params(labelsize=7.5)

    weight_bar(ax_gmv, gmv.weights,      "Min Variance Portfolio", PALETTE["gmv"])
    weight_bar(ax_tan, tangency.weights, "Max Sharpe Portfolio",   PALETTE["tangency"])

    # ── Stats annotation ──────────────────────────────────────────────────────
    def stat_box(ax_, portfolio, color, extra_y=0):
        r, v = annualize(portfolio.expected_return, portfolio.volatility)
        s = (r - rf_annual) / v
        txt = f"μ={r*100:.1f}%  σ={v*100:.1f}%  SR={s:.2f}"
        ax_.set_xlabel(txt, fontsize=7, color=color)

    stat_box(ax_gmv, gmv,      PALETTE["gmv"])
    stat_box(ax_tan, tangency, PALETTE["tangency"])

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    print(f"  Saved → {save_path}")
    return fig


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(0)
    T, n = 1260, 5
    assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

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

    opt = MarkowitzOptimizer(returns_df, risk_free_rate=0.05/252)

    print("Generating efficient frontier plot...")
    plot_efficient_frontier(opt, save_path="/mnt/user-data/outputs/efficient_frontier.png")
    print("Done.")
