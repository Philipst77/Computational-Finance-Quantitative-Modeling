import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Allow imports from sibling directories
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from data.market_data import get_market_data

# ---- DIRECTORIES ----
FIG_DIR = Path(__file__).resolve().parent / "figures" / "risk_metrics"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# RISK METRICS
# ==============================================================================
# This module implements the core risk measurement toolkit used in portfolio
# management and quantitative finance.
#
# Metrics covered:
#   Return metrics:   Annualized return, CAGR
#   Risk metrics:     Volatility, downside deviation, max drawdown
#   Ratio metrics:    Sharpe, Sortino, Calmar, Information Ratio
#   Tail risk:        Value at Risk (VaR), Conditional VaR (CVaR/ES)
#   Drawdown:         Drawdown series, max drawdown, average drawdown
#
# All metrics operate on a returns Series (daily log or simple returns).
# Annualization assumes 252 trading days per year by default.
# ==============================================================================

ANN = 252   # trading days per year


# ---- Return Metrics ----

def annualized_return(returns: pd.Series, ann: int = ANN) -> float:
    """Annualized arithmetic mean return."""
    return returns.mean() * ann


def cagr(returns: pd.Series, ann: int = ANN) -> float:
    """
    Compound Annual Growth Rate.
    CAGR = (1 + r_1)(1 + r_2)...(1 + r_T)^(ann/T) - 1
    More accurate than arithmetic mean for long periods.
    """
    cumulative = (1 + returns).prod()
    n_periods = len(returns)
    return cumulative ** (ann / n_periods) - 1


def total_return(returns: pd.Series) -> float:
    """Total cumulative return over the full period."""
    return (1 + returns).prod() - 1


# ---- Risk Metrics ----

def annualized_volatility(returns: pd.Series, ann: int = ANN) -> float:
    """Annualized standard deviation of returns."""
    return returns.std() * np.sqrt(ann)


def downside_deviation(returns: pd.Series, threshold: float = 0.0, ann: int = ANN) -> float:
    """
    Downside deviation: only penalizes returns below a threshold.
    Used in the Sortino ratio. More relevant than std for skewed distributions.
    """
    downside = returns[returns < threshold]
    if len(downside) == 0:
        return 0.0
    return downside.std() * np.sqrt(ann)


# ---- Drawdown ----

def drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Compute the drawdown series: how far the portfolio is below its
    previous peak at each point in time.
    DD(t) = (Cumulative(t) - Peak(t)) / Peak(t)
    """
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    return (cum - peak) / peak


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown: the worst peak-to-trough loss."""
    return drawdown_series(returns).min()


def avg_drawdown(returns: pd.Series) -> float:
    """Average of all drawdown values (only negative periods)."""
    dd = drawdown_series(returns)
    return dd[dd < 0].mean()


def drawdown_duration(returns: pd.Series) -> dict:
    """
    Max drawdown duration (days from peak to recovery)
    and current drawdown duration.
    """
    dd = drawdown_series(returns)
    in_drawdown = dd < 0
    cur_dur, longest = 0, 0
    for val in in_drawdown:
        if val:
            cur_dur += 1
            longest = max(longest, cur_dur)
        else:
            cur_dur = 0
    return {"max_duration_days": longest, "current_duration_days": cur_dur}


# ---- Ratio Metrics ----

def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.02, ann: int = ANN) -> float:
    """
    Sharpe ratio: excess return per unit of total risk.
    Sharpe = (R_p - R_f) / sigma_p
    """
    rf_daily = rf_annual / ann
    excess = returns - rf_daily
    if returns.std() == 0:
        return np.nan
    return excess.mean() / returns.std() * np.sqrt(ann)


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.02, ann: int = ANN) -> float:
    """
    Sortino ratio: excess return per unit of DOWNSIDE risk only.
    Better than Sharpe for skewed return distributions.
    """
    rf_daily = rf_annual / ann
    excess_mean = (returns - rf_daily).mean() * ann
    dd_dev = downside_deviation(returns, threshold=rf_daily, ann=ann)
    if dd_dev == 0:
        return np.nan
    return excess_mean / dd_dev


def calmar_ratio(returns: pd.Series, ann: int = ANN) -> float:
    """
    Calmar ratio: annualized return divided by maximum drawdown.
    Measures return per unit of worst-case loss.
    """
    ann_ret = annualized_return(returns, ann)
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return np.nan
    return ann_ret / mdd


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    ann: int = ANN,
) -> float:
    """
    Information ratio: active return per unit of tracking error.
    Measures consistency of outperformance vs a benchmark.
    """
    active = returns - benchmark_returns
    te = active.std() * np.sqrt(ann)
    if te == 0:
        return np.nan
    return active.mean() * ann / te


# ---- Value at Risk ----

def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR: the loss not exceeded at the given confidence level.
    Uses empirical distribution — no normality assumption.
    Returns a positive number representing the loss magnitude.
    """
    return -np.percentile(returns, (1 - confidence) * 100)


def var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Parametric (Gaussian) VaR: assumes returns are normally distributed.
    Less accurate with fat tails, but fast and analytically tractable.
    """
    from scipy.stats import norm
    mu = returns.mean()
    sigma = returns.std()
    z = norm.ppf(1 - confidence)
    return -(mu + z * sigma)


def cvar_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR (Expected Shortfall): average loss given that the loss
    exceeds VaR. CVaR is a coherent risk measure that captures tail severity.
    Returns a positive number representing the expected tail loss.
    """
    threshold = np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= threshold]
    return -tail.mean()


# ---- Full Risk Report ----

def risk_report(
    returns: pd.Series,
    name: str = "Portfolio",
    rf_annual: float = 0.02,
    ann: int = ANN,
    confidence: float = 0.95,
) -> pd.Series:
    """Compute the full risk metrics suite for a return series."""
    return pd.Series({
        "Ann. Return":       annualized_return(returns, ann),
        "CAGR":              cagr(returns, ann),
        "Ann. Volatility":   annualized_volatility(returns, ann),
        "Downside Dev.":     downside_deviation(returns, ann=ann),
        "Sharpe Ratio":      sharpe_ratio(returns, rf_annual, ann),
        "Sortino Ratio":     sortino_ratio(returns, rf_annual, ann),
        "Calmar Ratio":      calmar_ratio(returns, ann),
        "Max Drawdown":      max_drawdown(returns),
        "Avg Drawdown":      avg_drawdown(returns),
        f"VaR ({confidence:.0%})":  -var_historical(returns, confidence),
        f"CVaR ({confidence:.0%})": -cvar_historical(returns, confidence),
        "Skewness":          returns.skew(),
        "Kurtosis":          returns.kurtosis(),
    }, name=name)


# ---- Plots ----

def plot_cumulative_returns(
    returns_dict: dict,
    title: str = "Cumulative Returns",
) -> None:
    """Plot cumulative growth of $1 invested for multiple return series."""
    plt.figure(figsize=(12, 6))
    for label, r in returns_dict.items():
        cum = (1 + r).cumprod()
        plt.plot(cum.index, cum, linewidth=1.5, label=label)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($1 invested)")
    plt.title(title)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cumulative_returns.png", dpi=300)
    plt.show()


def plot_drawdowns(returns_dict: dict, title: str = "Drawdown Series") -> None:
    """Plot drawdown series for multiple strategies."""
    plt.figure(figsize=(12, 5))
    for label, r in returns_dict.items():
        dd = drawdown_series(r)
        plt.plot(dd.index, dd * 100, linewidth=1.2, label=label)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.title(title)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "drawdowns.png", dpi=300)
    plt.show()


def plot_rolling_sharpe(
    returns_dict: dict,
    window: int = 126,
    rf_annual: float = 0.02,
    ann: int = ANN,
) -> None:
    """Plot rolling Sharpe ratio for multiple strategies."""
    rf_daily = rf_annual / ann
    plt.figure(figsize=(12, 5))
    for label, r in returns_dict.items():
        excess = r - rf_daily
        roll_sharpe = (
            excess.rolling(window).mean() / r.rolling(window).std()
        ) * np.sqrt(ann)
        plt.plot(roll_sharpe.index, roll_sharpe, linewidth=1.2, label=label)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.axhline(1, color="gray", linewidth=0.5, linestyle=":")
    plt.xlabel("Date")
    plt.ylabel("Rolling Sharpe Ratio")
    plt.title(f"Rolling {window}-Day Sharpe Ratio")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rolling_sharpe.png", dpi=300)
    plt.show()


def plot_var_comparison(
    returns: pd.Series,
    name: str = "Portfolio",
    confidence_levels: list = [0.90, 0.95, 0.99],
) -> None:
    """
    Plot return distribution with VaR and CVaR thresholds.
    Compares historical vs parametric estimates.
    """
    from scipy.stats import norm
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(returns, bins=80, density=True, alpha=0.6,
            color="steelblue", label="Return distribution")
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 300)
    ax.plot(x, norm.pdf(x, mu, sigma), color="black",
            linewidth=1.5, linestyle="--", label="Normal fit")
    colors = ["gold", "orange", "crimson"]
    for cl, color in zip(confidence_levels, colors):
        var_h = var_historical(returns, cl)
        cvar_h = cvar_historical(returns, cl)
        ax.axvline(-var_h, color=color, linewidth=1.5,
                   linestyle="-", label=f"VaR {cl:.0%} = {var_h:.4f}")
        ax.axvline(-cvar_h, color=color, linewidth=1.5,
                   linestyle=":", label=f"CVaR {cl:.0%} = {cvar_h:.4f}")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Density")
    ax.set_title(f"{name}: Return Distribution with VaR & CVaR")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "var_comparison.png", dpi=300)
    plt.show()


def plot_risk_report_comparison(reports: dict) -> None:
    """Visual bar chart comparison of key risk metrics across portfolios."""
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
    plt.suptitle("Risk Metrics Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "risk_report_comparison.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # ---- Load real market data ----
    data = get_market_data()
    returns_df = data["returns"]
    names = data["names"]

    # ---- Individual asset risk reports ----
    print("\n=== Individual Asset Risk Reports ===\n")
    reports = {}
    for col, name in zip(returns_df.columns, names):
        r = returns_df[col]
        reports[name] = risk_report(r, name=name)

    summary = pd.DataFrame(reports).T
    pd.set_option("display.float_format", "{:.4f}".format)
    print(summary.to_string())

    # ---- Plots ----
    returns_dict = {name: returns_df[col]
                    for col, name in zip(returns_df.columns, names)}

    plot_cumulative_returns(returns_dict)
    plot_drawdowns(returns_dict)
    plot_rolling_sharpe(returns_dict)
    plot_var_comparison(returns_df[returns_df.columns[0]], name=names[0])
    plot_risk_report_comparison(reports)