import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path


# ---- DIRECTORIES ----
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures" / "risk_parity"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# RISK PARITY / EQUAL RISK CONTRIBUTION (ERC)
# ==============================================================================
# Mean-variance optimization maximizes return per unit of risk, but it is
# highly sensitive to return estimates — small errors in mu lead to wildly
# different portfolios. Risk parity takes a different approach: ignore return
# estimates entirely and focus only on equalizing risk contributions.
#
# The risk contribution of asset i to portfolio variance is:
#
#   RC_i = w_i * (Sigma * w)_i
#
# The total portfolio variance is: sigma_p^2 = sum(RC_i) = w^T * Sigma * w
#
# In an Equal Risk Contribution (ERC) portfolio:
#   RC_i = RC_j  for all i, j
#
# Equivalently: each asset contributes equally to total portfolio risk.
#
# This approach:
#   - Requires only the covariance matrix (no return estimates needed)
#   - Tends to be more diversified and stable than mean-variance
#   - Is used by many risk-focused hedge funds and asset managers
#   - Reduces concentration in high-volatility assets
# ==============================================================================


def marginal_risk_contribution(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Marginal risk contribution of each asset.
    MRC_i = (Sigma * w)_i / sqrt(w^T * Sigma * w)
    """
    port_vol = np.sqrt(weights @ cov @ weights)
    return (cov @ weights) / port_vol


def risk_contribution(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Absolute risk contribution of each asset.
    RC_i = w_i * MRC_i
    The sum of all RC_i equals portfolio volatility.
    """
    return weights * marginal_risk_contribution(weights, cov)


def risk_contribution_pct(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Percentage risk contribution of each asset.
    sum(pct_RC_i) = 1
    """
    rc = risk_contribution(weights, cov)
    return rc / rc.sum()


def equal_risk_contribution_portfolio(
    cov: np.ndarray,
    risk_budget: np.ndarray = None,
) -> dict:
    """
    Find the Equal Risk Contribution (ERC) portfolio.

    Parameters
    ----------
    cov         : covariance matrix (n x n)
    risk_budget : target risk budget per asset, shape (n,).
                  If None, defaults to equal risk (1/n per asset).
                  Can be set to any non-negative weights summing to 1.

    Returns
    -------
    dict with weights, return contributions, and risk diagnostics
    """
    n = cov.shape[0]

    if risk_budget is None:
        risk_budget = np.ones(n) / n   # equal risk budget

    def objective(w):
        # Minimize sum of squared differences in risk contributions
        rc = risk_contribution_pct(w, cov)
        return sum((rc[i] - risk_budget[i]) ** 2 for i in range(n))

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(1e-6, 1)] * n
    w0 = np.ones(n) / n

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    w = result.x
    rc_pct = risk_contribution_pct(w, cov)
    port_vol = np.sqrt(w @ cov @ w)

    return {
        "weights": w,
        "volatility": port_vol,
        "risk_contributions": rc_pct,
        "risk_budget": risk_budget,
        "converged": result.success,
    }


def naive_risk_parity(cov: np.ndarray) -> np.ndarray:
    """
    Naive risk parity: weight inversely proportional to individual asset volatility.
    This is a simple approximation — not the true ERC solution, but fast and intuitive.

    w_i = (1/sigma_i) / sum(1/sigma_j)
    """
    vols = np.sqrt(np.diag(cov))
    inv_vols = 1.0 / vols
    return inv_vols / inv_vols.sum()


def plot_risk_contributions(
    weights_dict: dict,
    cov: np.ndarray,
    asset_names: list[str],
) -> None:
    """
    Compare risk contributions (%) across different portfolio strategies.
    Shows how each strategy distributes risk across assets.
    """
    n = len(asset_names)
    strategies = list(weights_dict.keys())
    x = np.arange(n)
    width = 0.8 / len(strategies)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["steelblue", "crimson", "darkorange", "green"]

    # ---- Left: Weight allocation ----
    for i, (label, w) in enumerate(weights_dict.items()):
        axes[0].bar(x + i * width, w, width, label=label,
                    color=colors[i % len(colors)], alpha=0.8)

    axes[0].set_xticks(x + width * (len(strategies) - 1) / 2)
    axes[0].set_xticklabels(asset_names, rotation=20, ha="right", fontsize=9)
    axes[0].set_ylabel("Portfolio Weight")
    axes[0].set_title("Portfolio Weights by Strategy")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3, axis="y")

    # ---- Right: Risk contribution ----
    for i, (label, w) in enumerate(weights_dict.items()):
        rc = risk_contribution_pct(w, cov)
        axes[1].bar(x + i * width, rc, width, label=label,
                    color=colors[i % len(colors)], alpha=0.8)

    axes[1].set_xticks(x + width * (len(strategies) - 1) / 2)
    axes[1].set_xticklabels(asset_names, rotation=20, ha="right", fontsize=9)
    axes[1].set_ylabel("Risk Contribution (%)")
    axes[1].set_title("Risk Contributions by Strategy")
    axes[1].axhline(1 / n, color="black", linestyle="--",
                    linewidth=1, label=f"Equal = {1/n:.1%}")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")

    plt.suptitle("Portfolio Weights vs Risk Contributions", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "risk_contributions.png", dpi=300)
    plt.show()


def plot_diversification_ratio(
    mu: np.ndarray,
    cov: np.ndarray,
    asset_names: list[str],
    rf: float = 0.02,
) -> None:
    """
    Compare key metrics across four portfolio construction approaches:
    - Equal Weight
    - Naive Risk Parity (1/vol)
    - True ERC
    - Mean-Variance (Max Sharpe)
    """
    from mean_variance import maximum_sharpe_portfolio

    n = len(asset_names)
    vols = np.sqrt(np.diag(cov))

    # Build all portfolios
    w_ew  = np.ones(n) / n
    w_nrp = naive_risk_parity(cov)
    w_erc = equal_risk_contribution_portfolio(cov)["weights"]
    w_mvp = maximum_sharpe_portfolio(mu, cov, rf=rf)["weights"]

    strategies = {
        "Equal Weight":     w_ew,
        "Naive Risk Parity": w_nrp,
        "ERC":              w_erc,
        "Max Sharpe":       w_mvp,
    }

    metrics = {}
    for name, w in strategies.items():
        port_vol = np.sqrt(w @ cov @ w)
        port_ret = w @ mu
        weighted_vol = w @ vols   # weighted average of individual vols
        div_ratio = weighted_vol / port_vol   # diversification ratio
        metrics[name] = {
            "return":   port_ret,
            "vol":      port_vol,
            "sharpe":   (port_ret - rf) / port_vol,
            "div_ratio": div_ratio,
        }

    labels = list(metrics.keys())
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))

    for ax, key, title, color in zip(
        axes,
        ["return", "vol", "sharpe", "div_ratio"],
        ["Expected Return", "Volatility", "Sharpe Ratio", "Diversification Ratio"],
        ["steelblue", "crimson", "green", "darkorange"],
    ):
        vals = [metrics[l][key] for l in labels]
        ax.bar(labels, vals, color=color, alpha=0.8)
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    plt.suptitle("Portfolio Strategy Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "strategy_comparison.png", dpi=300)
    plt.show()

    # Print table
    print(f"\n{'Strategy':<22} {'Return':>8} {'Vol':>8} {'Sharpe':>8} {'Div Ratio':>10}")
    print("-" * 60)
    for name, m in metrics.items():
        print(f"{name:<22} {m['return']:>8.2%} {m['vol']:>8.2%} "
              f"{m['sharpe']:>8.2f} {m['div_ratio']:>10.3f}")


# ---- Example dataset (same as mean_variance.py) ----
def get_example_data():
    from mean_variance import get_example_data as _get
    return _get()


if __name__ == "__main__":
    mu, cov, asset_names = get_example_data()
    n = len(asset_names)

    # ---- Build all portfolios ----
    w_ew  = np.ones(n) / n
    w_nrp = naive_risk_parity(cov)
    erc   = equal_risk_contribution_portfolio(cov)

    print("Equal Risk Contribution Portfolio:")
    for name, w, rc in zip(asset_names, erc["weights"], erc["risk_contributions"]):
        print(f"  {name:<15}  weight={w:.1%}  risk contrib={rc:.1%}")
    print(f"  Volatility: {erc['volatility']:.2%}")
    print(f"  Converged:  {erc['converged']}")
    print()

    # ---- Plot 1: weights vs risk contributions ----
    plot_risk_contributions(
        {"Equal Weight": w_ew, "Naive Risk Parity": w_nrp, "ERC": erc["weights"]},
        cov,
        asset_names,
    )

    # ---- Plot 2: strategy comparison ----
    plot_diversification_ratio(mu, cov, asset_names, rf=0.02)