import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path


# ---- DIRECTORIES ----
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures" / "mean_variance"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# MEAN-VARIANCE OPTIMIZATION (Markowitz, 1952)
# ==============================================================================
# The core idea: given a set of risky assets, find the portfolio weights that
# minimize variance for a given level of expected return.
#
# Portfolio return:   mu_p   = w^T * mu
# Portfolio variance: sigma_p^2 = w^T * Sigma * w
#
# The set of all minimum-variance portfolios traces out the EFFICIENT FRONTIER —
# the upper portion of the minimum variance curve where no portfolio can achieve
# higher return without taking on more risk.
#
# Key portfolios on the frontier:
#   - Minimum Variance Portfolio (MVP): lowest possible variance
#   - Maximum Sharpe Portfolio (Tangency): highest return per unit of risk
#   - Any target-return portfolio: minimizes variance for a given mu target
#
# Constraints:
#   - Weights sum to 1 (fully invested)
#   - Weights >= 0 (long-only, no short selling)
# ==============================================================================


def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    """Expected portfolio return."""
    return weights @ mu


def portfolio_variance(weights: np.ndarray, cov: np.ndarray) -> float:
    """Portfolio variance."""
    return weights @ cov @ weights


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    """Portfolio volatility (std dev)."""
    return np.sqrt(portfolio_variance(weights, cov))


def portfolio_sharpe(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = 0.0,
) -> float:
    """Sharpe ratio of a portfolio."""
    ret = portfolio_return(weights, mu)
    vol = portfolio_volatility(weights, cov)
    return (ret - rf) / vol


def minimum_variance_portfolio(
    mu: np.ndarray,
    cov: np.ndarray,
) -> dict:
    """
    Find the global minimum variance portfolio.
    This is the leftmost point on the efficient frontier.
    """
    n = len(mu)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n

    result = minimize(
        lambda w: portfolio_variance(w, cov),
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return {
        "weights": result.x,
        "return": portfolio_return(result.x, mu),
        "volatility": portfolio_volatility(result.x, cov),
        "sharpe": portfolio_sharpe(result.x, mu, cov),
    }


def maximum_sharpe_portfolio(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = 0.02,
) -> dict:
    """
    Find the maximum Sharpe ratio portfolio (tangency portfolio).
    This is the portfolio on the efficient frontier that maximizes
    return per unit of risk — the optimal risky portfolio.
    """
    n = len(mu)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n

    result = minimize(
        lambda w: -portfolio_sharpe(w, mu, cov, rf),
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return {
        "weights": result.x,
        "return": portfolio_return(result.x, mu),
        "volatility": portfolio_volatility(result.x, cov),
        "sharpe": portfolio_sharpe(result.x, mu, cov, rf),
    }


def target_return_portfolio(
    mu: np.ndarray,
    cov: np.ndarray,
    target_return: float,
) -> dict:
    """
    Find the minimum variance portfolio for a given target return.
    Sweeping target_return across all feasible values traces the efficient frontier.
    """
    n = len(mu)
    constraints = [
        {"type": "eq", "fun": lambda w: w.sum() - 1},
        {"type": "eq", "fun": lambda w: portfolio_return(w, mu) - target_return},
    ]
    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n

    result = minimize(
        lambda w: portfolio_variance(w, cov),
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        return None

    return {
        "weights": result.x,
        "return": portfolio_return(result.x, mu),
        "volatility": portfolio_volatility(result.x, cov),
        "sharpe": portfolio_sharpe(result.x, mu, cov),
    }


def compute_efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the efficient frontier by sweeping target returns.

    Returns
    -------
    vols    : array of portfolio volatilities along the frontier
    returns : array of portfolio returns along the frontier
    """
    mvp = minimum_variance_portfolio(mu, cov)
    max_ret = mu.max()
    target_returns = np.linspace(mvp["return"], max_ret, n_points)

    vols, rets = [], []
    for tr in target_returns:
        p = target_return_portfolio(mu, cov, tr)
        if p is not None:
            vols.append(p["volatility"])
            rets.append(p["return"])

    return np.array(vols), np.array(rets)


def simulate_random_portfolios(
    mu: np.ndarray,
    cov: np.ndarray,
    n_portfolios: int = 5_000,
    rf: float = 0.02,
    seed: int = 42,
) -> dict:
    """
    Simulate random portfolio weights to visualize the feasible set.
    The efficient frontier lies on the upper-left boundary of this cloud.
    """
    rng = np.random.default_rng(seed)
    n = len(mu)

    port_returns, port_vols, port_sharpes = [], [], []

    for _ in range(n_portfolios):
        w = rng.random(n)
        w /= w.sum()
        port_returns.append(portfolio_return(w, mu))
        port_vols.append(portfolio_volatility(w, cov))
        port_sharpes.append(portfolio_sharpe(w, mu, cov, rf))

    return {
        "returns": np.array(port_returns),
        "vols": np.array(port_vols),
        "sharpes": np.array(port_sharpes),
    }


def plot_efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    asset_names: list[str],
    rf: float = 0.02,
) -> None:
    """
    Plot the efficient frontier with:
    - Random portfolio cloud (feasible set)
    - Efficient frontier curve
    - Minimum variance portfolio
    - Maximum Sharpe portfolio
    - Individual assets
    """
    # ---- Compute ----
    sim = simulate_random_portfolios(mu, cov, rf=rf)
    frontier_vols, frontier_rets = compute_efficient_frontier(mu, cov)
    mvp = minimum_variance_portfolio(mu, cov)
    msr = maximum_sharpe_portfolio(mu, cov, rf=rf)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(11, 7))

    # Random portfolio cloud colored by Sharpe ratio
    sc = ax.scatter(
        sim["vols"], sim["returns"],
        c=sim["sharpes"], cmap="viridis",
        alpha=0.3, s=8, label="Random portfolios"
    )
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

    # Efficient frontier
    ax.plot(frontier_vols, frontier_rets, color="white",
            linewidth=2.5, label="Efficient Frontier", zorder=5)

    # Key portfolios
    ax.scatter(mvp["volatility"], mvp["return"], color="cyan",
               s=120, zorder=10, label=f"Min Variance  (Sharpe={mvp['sharpe']:.2f})")
    ax.scatter(msr["volatility"], msr["return"], color="red",
               s=120, marker="*", zorder=10, label=f"Max Sharpe  (Sharpe={msr['sharpe']:.2f})")

    # Individual assets
    asset_vols = np.sqrt(np.diag(cov))
    for i, name in enumerate(asset_names):
        ax.scatter(asset_vols[i], mu[i], s=80, zorder=8, marker="D")
        ax.annotate(name, (asset_vols[i], mu[i]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9, color="white")

    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")

    ax.set_xlabel("Portfolio Volatility (σ)")
    ax.set_ylabel("Expected Return (μ)")
    ax.set_title("Mean-Variance Efficient Frontier")
    ax.legend(fontsize=9, facecolor="#2a2a3e", labelcolor="white")
    ax.grid(alpha=0.2, color="gray")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "efficient_frontier.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_weight_allocation(
    mu: np.ndarray,
    cov: np.ndarray,
    asset_names: list[str],
    n_points: int = 50,
) -> None:
    """
    Show how portfolio weights shift as target return increases along the frontier.
    This is the 'portfolio composition' view of the efficient frontier.
    """
    mvp = minimum_variance_portfolio(mu, cov)
    target_returns = np.linspace(mvp["return"], mu.max() * 0.95, n_points)

    weight_matrix = []
    feasible_returns = []

    for tr in target_returns:
        p = target_return_portfolio(mu, cov, tr)
        if p is not None:
            weight_matrix.append(p["weights"])
            feasible_returns.append(p["return"])

    weight_matrix = np.array(weight_matrix)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.stackplot(
        feasible_returns,
        weight_matrix.T,
        labels=asset_names,
        alpha=0.85,
    )
    ax.set_xlabel("Target Return")
    ax.set_ylabel("Portfolio Weight")
    ax.set_title("Portfolio Composition Along the Efficient Frontier")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "frontier_weights.png", dpi=300)
    plt.show()


# ---- Example dataset ----
def get_example_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    6-asset example: US Equity, Int'l Equity, Bonds, Real Estate, Commodities, Cash
    Returns and covariance matrix are illustrative annualized values.
    """
    asset_names = ["US Equity", "Intl Equity", "Bonds", "Real Estate", "Commodities", "Cash"]

    mu = np.array([0.10, 0.09, 0.04, 0.08, 0.06, 0.02])

    # Correlation matrix
    corr = np.array([
        [1.00,  0.75,  0.00,  0.55,  0.20,  0.00],
        [0.75,  1.00, -0.05,  0.45,  0.25,  0.00],
        [0.00, -0.05,  1.00,  0.10, -0.10,  0.10],
        [0.55,  0.45,  0.10,  1.00,  0.15,  0.00],
        [0.20,  0.25, -0.10,  0.15,  1.00,  0.00],
        [0.00,  0.00,  0.10,  0.00,  0.00,  1.00],
    ])

    vols = np.array([0.16, 0.18, 0.06, 0.14, 0.20, 0.01])
    cov = np.outer(vols, vols) * corr

    return mu, cov, asset_names


if __name__ == "__main__":
    mu, cov, asset_names = get_example_data()

    # ---- Key portfolios ----
    mvp = minimum_variance_portfolio(mu, cov)
    msr = maximum_sharpe_portfolio(mu, cov, rf=0.02)

    print("Minimum Variance Portfolio:")
    for name, w in zip(asset_names, mvp["weights"]):
        print(f"  {name:<15} {w:.1%}")
    print(f"  Return: {mvp['return']:.2%}  Vol: {mvp['volatility']:.2%}  Sharpe: {mvp['sharpe']:.2f}")
    print()

    print("Maximum Sharpe Portfolio:")
    for name, w in zip(asset_names, msr["weights"]):
        print(f"  {name:<15} {w:.1%}")
    print(f"  Return: {msr['return']:.2%}  Vol: {msr['volatility']:.2%}  Sharpe: {msr['sharpe']:.2f}")
    print()

    # ---- Plots ----
    plot_efficient_frontier(mu, cov, asset_names, rf=0.02)
    plot_weight_allocation(mu, cov, asset_names)