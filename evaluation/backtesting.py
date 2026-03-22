import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from data.market_data import get_market_data
from evaluation.risk_metrics import risk_report, drawdown_series

# ---- DIRECTORIES ----
FIG_DIR = Path(__file__).resolve().parent / "figures" / "backtesting"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# BACKTESTING FRAMEWORK
# ==============================================================================
# Backtesting evaluates portfolio strategies on historical data by simulating
# how they would have performed if applied in the past.
#
# This module implements a rolling-window backtest:
#   1. At each rebalancing date, estimate parameters using a lookback window
#   2. Compute optimal portfolio weights using the chosen strategy
#   3. Hold those weights until the next rebalancing date
#   4. Record realized returns over the holding period
#   5. Repeat until the end of the data
#
# Strategies compared:
#   - Equal Weight (EW): 1/n allocation, no optimization
#   - Minimum Variance (MV): minimize portfolio volatility
#   - Maximum Sharpe (MS): maximize return per unit of risk
#   - Risk Parity (ERC): equalize risk contributions
#
# The output is a time series of portfolio returns for each strategy,
# which can be evaluated using the full risk_metrics toolkit.
# ==============================================================================


def equal_weight(returns_window: pd.DataFrame, **kwargs) -> np.ndarray:
    """Equal weight: 1/n per asset. No optimization needed."""
    n = returns_window.shape[1]
    return np.ones(n) / n


def minimum_variance(returns_window: pd.DataFrame, **kwargs) -> np.ndarray:
    """Minimum variance portfolio from rolling covariance estimate."""
    from scipy.optimize import minimize
    cov = returns_window.cov().values * 252
    n = cov.shape[0]
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, 1)] * n
    result = minimize(
        lambda w: w @ cov @ w,
        np.ones(n) / n,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x if result.success else np.ones(n) / n


def maximum_sharpe(
    returns_window: pd.DataFrame,
    rf_annual: float = 0.02,
    **kwargs,
) -> np.ndarray:
    """Maximum Sharpe ratio portfolio from rolling estimates."""
    from scipy.optimize import minimize
    mu = returns_window.mean().values * 252
    cov = returns_window.cov().values * 252
    n = len(mu)
    rf = rf_annual

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        return -(ret - rf) / vol if vol > 0 else 0

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, 1)] * n
    result = minimize(
        neg_sharpe,
        np.ones(n) / n,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x if result.success else np.ones(n) / n


def risk_parity(returns_window: pd.DataFrame, **kwargs) -> np.ndarray:
    """Equal Risk Contribution portfolio from rolling covariance estimate."""
    from scipy.optimize import minimize
    cov = returns_window.cov().values * 252
    n = cov.shape[0]
    target = np.ones(n) / n

    def objective(w):
        port_vol = np.sqrt(w @ cov @ w)
        rc = w * (cov @ w) / port_vol
        rc_pct = rc / rc.sum()
        return sum((rc_pct[i] - target[i]) ** 2 for i in range(n))

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(1e-6, 1)] * n
    result = minimize(
        objective,
        np.ones(n) / n,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return result.x if result.success else np.ones(n) / n


# ---- Core Backtest Engine ----

STRATEGIES = {
    "Equal Weight":    equal_weight,
    "Min Variance":    minimum_variance,
    "Max Sharpe":      maximum_sharpe,
    "Risk Parity":     risk_parity,
}


def run_backtest(
    returns: pd.DataFrame,
    strategy_fn,
    lookback: int = 252,
    rebalance_freq: int = 21,
    rf_annual: float = 0.02,
    transaction_cost: float = 0.001,
) -> pd.Series:
    """
    Run a rolling-window backtest for a given strategy.

    Parameters
    ----------
    returns          : daily return DataFrame (T x n_assets)
    strategy_fn      : function(returns_window) -> weights array
    lookback         : number of days used to estimate parameters
    rebalance_freq   : days between rebalancing (21 ≈ monthly)
    rf_annual        : risk-free rate for Sharpe-based strategies
    transaction_cost : round-trip cost per unit of turnover

    Returns
    -------
    portfolio_returns : daily portfolio return Series
    """
    n_days, n_assets = returns.shape
    portfolio_returns = []
    dates = []
    current_weights = np.ones(n_assets) / n_assets

    for t in range(lookback, n_days):
        # Rebalance on schedule
        if (t - lookback) % rebalance_freq == 0:
            window = returns.iloc[t - lookback:t]
            new_weights = strategy_fn(window, rf_annual=rf_annual)
            # Apply transaction cost based on turnover
            turnover = np.abs(new_weights - current_weights).sum()
            tc = turnover * transaction_cost
            current_weights = new_weights
        else:
            tc = 0.0

        # Realized return for this day
        day_ret = returns.iloc[t].values @ current_weights - tc
        portfolio_returns.append(day_ret)
        dates.append(returns.index[t])

        # Drift weights with market (before next rebalance)
        asset_rets = returns.iloc[t].values
        new_vals = current_weights * (1 + asset_rets)
        if new_vals.sum() > 0:
            current_weights = new_vals / new_vals.sum()

    return pd.Series(portfolio_returns, index=dates)


def run_all_strategies(
    returns: pd.DataFrame,
    lookback: int = 252,
    rebalance_freq: int = 21,
    rf_annual: float = 0.02,
    transaction_cost: float = 0.001,
) -> dict:
    """
    Run backtest for all strategies and return dict of return series.
    """
    results = {}
    for name, fn in STRATEGIES.items():
        print(f"Running backtest: {name}...", end=" ")
        results[name] = run_backtest(
            returns, fn,
            lookback=lookback,
            rebalance_freq=rebalance_freq,
            rf_annual=rf_annual,
            transaction_cost=transaction_cost,
        )
        print("done")
    return results


# ---- Plots ----

def plot_equity_curves(results: dict, title: str = "Strategy Equity Curves") -> None:
    """Plot cumulative growth of $1 for each strategy."""
    plt.figure(figsize=(12, 6))
    for label, r in results.items():
        cum = (1 + r).cumprod()
        plt.plot(cum.index, cum, linewidth=1.5, label=label)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($1 invested)")
    plt.title(title)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "equity_curves.png", dpi=300)
    plt.show()


def plot_drawdowns(results: dict) -> None:
    """Plot drawdown series for each strategy."""
    plt.figure(figsize=(12, 5))
    for label, r in results.items():
        dd = drawdown_series(r)
        plt.plot(dd.index, dd * 100, linewidth=1.2, label=label)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.title("Strategy Drawdowns")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "strategy_drawdowns.png", dpi=300)
    plt.show()


def plot_rolling_performance(
    results: dict,
    window: int = 126,
    rf_annual: float = 0.02,
) -> None:
    """Plot rolling annualized return and Sharpe ratio side by side."""
    rf_daily = rf_annual / 252

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for label, r in results.items():
        roll_ret = r.rolling(window).mean() * 252
        excess = r - rf_daily
        roll_sharpe = (excess.rolling(window).mean() /
                       r.rolling(window).std()) * np.sqrt(252)
        axes[0].plot(roll_ret.index, roll_ret, linewidth=1.2, label=label)
        axes[1].plot(roll_sharpe.index, roll_sharpe, linewidth=1.2, label=label)

    axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[0].set_ylabel("Annualized Return")
    axes[0].set_title(f"Rolling {window}-Day Annualized Return")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[1].axhline(1, color="gray", linewidth=0.5, linestyle=":")
    axes[1].set_ylabel("Sharpe Ratio")
    axes[1].set_xlabel("Date")
    axes[1].set_title(f"Rolling {window}-Day Sharpe Ratio")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "rolling_performance.png", dpi=300)
    plt.show()


def plot_performance_summary(reports: dict) -> None:
    """Bar chart summary of key metrics across all strategies."""
    df = pd.DataFrame(reports).T
    metrics = [
        ("Ann. Return",     "steelblue",  True),
        ("Ann. Volatility", "crimson",    True),
        ("Sharpe Ratio",    "green",      False),
        ("Sortino Ratio",   "darkorange", False),
        ("Calmar Ratio",    "purple",     False),
        ("Max Drawdown",    "red",        True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for ax, (metric, color, as_pct) in zip(axes, metrics):
        vals = df[metric].astype(float)
        ax.bar(range(len(df)), vals, color=color, alpha=0.8)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df.index, rotation=20, ha="right", fontsize=8)
        ax.set_title(metric, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        if as_pct:
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v:.1%}")
            )
    plt.suptitle("Backtest Performance Summary", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "performance_summary.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # ---- Load data ----
    data = get_market_data()
    returns = data["returns"]

    # ---- Run all strategy backtests ----
    print("\nRunning backtests (lookback=252 days, rebalance=monthly)...\n")
    results = run_all_strategies(
        returns,
        lookback=252,
        rebalance_freq=21,
        rf_annual=0.02,
        transaction_cost=0.001,
    )

    # ---- Risk reports ----
    print("\n=== Backtest Performance Summary ===\n")
    reports = {name: risk_report(r, name=name) for name, r in results.items()}
    summary = pd.DataFrame(reports).T
    pd.set_option("display.float_format", "{:.4f}".format)
    print(summary.to_string())

    # ---- Plots ----
    plot_equity_curves(results)
    plot_drawdowns(results)
    plot_rolling_performance(results)
    plot_performance_summary(reports)