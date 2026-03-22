import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# yfinance for downloading real market data
try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance is required. Install with: pip install yfinance")


# ---- DIRECTORIES ----
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures" / "market_data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = BASE_DIR / "cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# MARKET DATA
# ==============================================================================
# This module handles downloading, caching, and preprocessing real market data.
# It provides clean return series, covariance matrices, and summary statistics
# that feed into risk_metrics.py, backtesting.py, and portfolio optimization.
#
# Default universe: 6 ETFs representing major asset classes
#   SPY  — US Large Cap Equity (S&P 500)
#   EFA  — International Developed Equity
#   AGG  — US Aggregate Bonds
#   VNQ  — US Real Estate (REITs)
#   GSG  — Commodities
#   SHV  — Short-Term Treasuries (cash proxy)
#
# These map directly to the illustrative assets used in portfolio_optimization.
# ==============================================================================

DEFAULT_TICKERS = ["SPY", "EFA", "AGG", "VNQ", "GSG", "SHV"]
DEFAULT_NAMES   = ["US Equity", "Intl Equity", "Bonds", "Real Estate", "Commodities", "Cash"]
DEFAULT_START   = "2010-01-01"
DEFAULT_END     = "2024-12-31"


def download_prices(
    tickers: list[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download adjusted closing prices for a list of tickers.
    Caches results to disk to avoid repeated API calls.

    Parameters
    ----------
    tickers   : list of ticker symbols
    start     : start date string (YYYY-MM-DD)
    end       : end date string (YYYY-MM-DD)
    use_cache : if True, load from cache if available

    Returns
    -------
    prices : DataFrame of adjusted close prices, shape (T, n_assets)
    """
    cache_file = DATA_DIR / f"prices_{'_'.join(tickers)}_{start}_{end}.csv"

    if use_cache and cache_file.exists():
        print(f"Loading cached prices from {cache_file.name}")
        prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return prices

    print(f"Downloading prices for {tickers} from {start} to {end}...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    prices = prices.dropna(how="all").ffill()

    if use_cache:
        prices.to_csv(cache_file)
        print(f"Cached to {cache_file.name}")

    print(f"Downloaded {len(prices)} days of data for {len(prices.columns)} assets")
    return prices


def compute_returns(
    prices: pd.DataFrame,
    frequency: str = "daily",
) -> pd.DataFrame:
    """
    Compute log returns from price series.

    Parameters
    ----------
    prices    : DataFrame of prices
    frequency : 'daily', 'weekly', or 'monthly'

    Returns
    -------
    returns : DataFrame of log returns
    """
    if frequency == "weekly":
        prices = prices.resample("W").last()
    elif frequency == "monthly":
        prices = prices.resample("ME").last()

    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


def annualization_factor(frequency: str = "daily") -> int:
    """Number of periods per year for a given frequency."""
    return {"daily": 252, "weekly": 52, "monthly": 12}[frequency]


def compute_statistics(
    returns: pd.DataFrame,
    frequency: str = "daily",
    rf_annual: float = 0.02,
    names: list[str] = None,
) -> pd.DataFrame:
    """
    Compute annualized return, volatility, Sharpe ratio, and drawdown stats
    for each asset in the returns DataFrame.

    Returns
    -------
    stats : DataFrame with one row per asset
    """
    ann = annualization_factor(frequency)
    rf_per_period = rf_annual / ann

    stats = {}
    for col in returns.columns:
        r = returns[col].dropna()
        ann_ret  = r.mean() * ann
        ann_vol  = r.std() * np.sqrt(ann)
        sharpe   = (r.mean() - rf_per_period) / r.std() * np.sqrt(ann)

        # Max drawdown
        cum = (1 + r).cumprod()
        rolling_max = cum.cummax()
        dd = (cum - rolling_max) / rolling_max
        max_dd = dd.min()

        stats[col] = {
            "Ann. Return":    ann_ret,
            "Ann. Volatility": ann_vol,
            "Sharpe Ratio":   sharpe,
            "Max Drawdown":   max_dd,
            "Skewness":       r.skew(),
            "Kurtosis":       r.kurtosis(),
        }

    df = pd.DataFrame(stats).T
    if names is not None:
        df.index = names[:len(df)]
    return df.round(4)


def compute_covariance(
    returns: pd.DataFrame,
    frequency: str = "daily",
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Compute the annualized covariance matrix from returns.
    """
    ann = annualization_factor(frequency) if annualize else 1
    return returns.cov() * ann


def compute_expected_returns(
    returns: pd.DataFrame,
    frequency: str = "daily",
    method: str = "historical",
) -> pd.Series:
    """
    Compute expected returns for each asset.

    Parameters
    ----------
    method : 'historical' — annualized sample mean
             'ewm'        — exponentially weighted mean (more weight on recent data)
    """
    ann = annualization_factor(frequency)

    if method == "historical":
        return returns.mean() * ann
    elif method == "ewm":
        return returns.ewm(halflife=63).mean().iloc[-1] * ann
    else:
        raise ValueError(f"Unknown method: {method}")


def get_market_data(
    tickers: list[str] = DEFAULT_TICKERS,
    names: list[str] = DEFAULT_NAMES,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    frequency: str = "daily",
    rf_annual: float = 0.02,
    use_cache: bool = True,
) -> dict:
    """
    Full pipeline: download → compute returns → statistics → covariance.

    Returns a dict with everything needed for portfolio optimization
    and backtesting:
        prices    : raw price DataFrame
        returns   : log return DataFrame
        mu        : annualized expected returns (numpy array)
        cov       : annualized covariance matrix (numpy array)
        stats     : summary statistics DataFrame
        tickers   : ticker list
        names     : asset name list
        frequency : data frequency
    """
    prices  = download_prices(tickers, start, end, use_cache)
    returns = compute_returns(prices, frequency)
    stats   = compute_statistics(returns, frequency, rf_annual, names)
    cov_df  = compute_covariance(returns, frequency)
    mu_s    = compute_expected_returns(returns, frequency)

    # Align names
    display_names = names[:len(tickers)] if names else tickers

    print("\n=== Asset Statistics ===")
    print(stats.to_string())

    return {
        "prices":    prices,
        "returns":   returns,
        "mu":        mu_s.values,
        "cov":       cov_df.values,
        "stats":     stats,
        "tickers":   tickers,
        "names":     display_names,
        "frequency": frequency,
    }


# ---- Plotting ----

def plot_prices(prices: pd.DataFrame, names: list[str] = None) -> None:
    """Plot normalized price series (rebased to 100)."""
    normalized = prices / prices.iloc[0] * 100
    labels = names if names else prices.columns.tolist()

    plt.figure(figsize=(12, 6))
    for col, label in zip(normalized.columns, labels):
        plt.plot(normalized.index, normalized[col], linewidth=1.2, label=label)

    plt.xlabel("Date")
    plt.ylabel("Normalized Price (base = 100)")
    plt.title("Asset Price History (Normalized)")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "normalized_prices.png", dpi=300)
    plt.show()


def plot_returns_distribution(returns: pd.DataFrame, names: list[str] = None) -> None:
    """Plot return distributions for each asset."""
    from scipy.stats import norm

    labels = names if names else returns.columns.tolist()
    n = len(returns.columns)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4))
    axes = axes.flatten()

    for ax, col, label in zip(axes, returns.columns, labels):
        r = returns[col].dropna()
        x = np.linspace(r.min(), r.max(), 300)
        ax.hist(r, bins=60, density=True, alpha=0.6, color="steelblue")
        ax.plot(x, norm.pdf(x, r.mean(), r.std()), color="crimson",
                linewidth=1.5, label="Normal fit")
        ax.set_title(label)
        ax.set_xlabel("Log Return")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for ax in axes[n:]:
        ax.axis("off")

    plt.suptitle("Return Distributions by Asset", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "return_distributions.png", dpi=300)
    plt.show()


def plot_correlation_matrix(returns: pd.DataFrame, names: list[str] = None) -> None:
    """Plot the correlation matrix as a heatmap."""
    corr = returns.corr()
    labels = names if names else returns.columns.tolist()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Correlation")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if abs(corr.values[i, j]) < 0.7 else "white")

    ax.set_title("Asset Correlation Matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_matrix.png", dpi=300)
    plt.show()


def plot_rolling_volatility(
    returns: pd.DataFrame,
    names: list[str] = None,
    window: int = 63,
    frequency: str = "daily",
) -> None:
    """Plot rolling annualized volatility for each asset."""
    ann = annualization_factor(frequency)
    labels = names if names else returns.columns.tolist()
    rolling_vol = returns.rolling(window).std() * np.sqrt(ann)

    plt.figure(figsize=(12, 6))
    for col, label in zip(rolling_vol.columns, labels):
        plt.plot(rolling_vol.index, rolling_vol[col], linewidth=1.0,
                 alpha=0.85, label=label)

    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility")
    plt.title(f"Rolling {window}-Day Annualized Volatility")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rolling_volatility.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    data = get_market_data()

    plot_prices(data["prices"], data["names"])
    plot_returns_distribution(data["returns"], data["names"])
    plot_correlation_matrix(data["returns"], data["names"])
    plot_rolling_volatility(data["returns"], data["names"])