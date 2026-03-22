import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ---- FIGURES DIRECTORY ----
BASE_DIR = Path(__file__).parent
FIG_DIR = BASE_DIR / "figures" / "volatility_models"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# GARCH(1,1) MODEL
# ==============================================================================
# GARCH = Generalized Autoregressive Conditional Heteroskedasticity
#
# The key idea: volatility is NOT constant. It clusters — high volatility
# tends to be followed by more high volatility (and vice versa).
#
# GARCH(1,1) equations:
#
#   Return:       r_t = mu + epsilon_t
#   Shock:        epsilon_t = sigma_t * z_t,   z_t ~ N(0,1)
#   Variance:     sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
#
# Parameters:
#   omega  — baseline/long-run variance contribution (must be > 0)
#   alpha  — weight on last period's shock squared (news impact)
#   beta   — weight on last period's variance (persistence)
#
# Stationarity condition:  alpha + beta < 1
# Long-run variance:       sigma_LR^2 = omega / (1 - alpha - beta)
# ==============================================================================


def simulate_garch(
    n: int = 1000,
    omega: float = 0.0001,
    alpha: float = 0.1,
    beta: float = 0.85,
    mu: float = 0.0,
    sigma0: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a GARCH(1,1) process.

    Parameters
    ----------
    n      : number of time steps
    omega  : constant term in variance equation
    alpha  : ARCH coefficient (shock impact)
    beta   : GARCH coefficient (variance persistence)
    mu     : mean return
    sigma0 : initial volatility (annualized daily vol)
    seed   : random seed for reproducibility

    Returns
    -------
    returns    : array of shape (n,) — simulated returns
    volatility : array of shape (n,) — conditional volatility (sigma_t)
    variance   : array of shape (n,) — conditional variance (sigma_t^2)
    """
    assert alpha + beta < 1, "GARCH stationarity condition violated: alpha + beta must be < 1"

    rng = np.random.default_rng(seed)

    variance = np.zeros(n)
    returns = np.zeros(n)

    # Initialize
    variance[0] = sigma0 ** 2

    for t in range(n):
        z = rng.standard_normal()
        returns[t] = mu + np.sqrt(variance[t]) * z

        if t + 1 < n:
            variance[t + 1] = omega + alpha * returns[t] ** 2 + beta * variance[t]

    volatility = np.sqrt(variance)

    long_run_vol = np.sqrt(omega / (1 - alpha - beta))
    print(f"GARCH(1,1) Parameters: omega={omega}, alpha={alpha}, beta={beta}")
    print(f"Long-run (unconditional) volatility: {long_run_vol:.6f} per period")
    print(f"Persistence (alpha + beta):           {alpha + beta:.4f}")

    return returns, volatility, variance


def plot_returns_and_volatility(
    returns: np.ndarray,
    volatility: np.ndarray,
) -> None:
    """
    Plot simulated returns alongside the conditional volatility.
    This is the classic GARCH diagnostic plot — it visually shows
    volatility clustering.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # ---- Returns ----
    axes[0].plot(returns, color="steelblue", linewidth=0.7, alpha=0.85)
    axes[0].set_ylabel("Return")
    axes[0].set_title("GARCH(1,1): Simulated Returns")
    axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[0].grid(alpha=0.3)

    # ---- Conditional Volatility ----
    axes[1].plot(volatility, color="crimson", linewidth=0.9)
    axes[1].set_ylabel("Conditional Volatility σ_t")
    axes[1].set_xlabel("Time Step")
    axes[1].set_title("GARCH(1,1): Conditional Volatility (Clustering Effect)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "garch_returns_volatility.png", dpi=300)
    plt.show()


def plot_volatility_clustering(returns: np.ndarray) -> None:
    """
    Plot |returns| and returns^2 to visually demonstrate volatility clustering.
    These are the raw signals GARCH is designed to model.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(np.abs(returns), color="darkorange", linewidth=0.7)
    axes[0].set_ylabel("|Return|")
    axes[0].set_title("Absolute Returns — Volatility Clustering Signal")
    axes[0].grid(alpha=0.3)

    axes[1].plot(returns ** 2, color="purple", linewidth=0.7)
    axes[1].set_ylabel("Return²")
    axes[1].set_xlabel("Time Step")
    axes[1].set_title("Squared Returns — Volatility Clustering Signal")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "garch_clustering_signals.png", dpi=300)
    plt.show()


def compare_persistence_levels() -> None:
    """
    Compare GARCH simulations with different persistence levels (alpha + beta).

    Low persistence  → volatility reverts to mean quickly
    High persistence → volatility shocks last a long time (like real markets)
    """
    configs = [
        {"alpha": 0.05, "beta": 0.50, "label": "Low Persistence (α+β=0.55)"},
        {"alpha": 0.10, "beta": 0.75, "label": "Medium Persistence (α+β=0.85)"},
        {"alpha": 0.10, "beta": 0.88, "label": "High Persistence (α+β=0.98)"},
    ]

    fig, axes = plt.subplots(len(configs), 1, figsize=(12, 9), sharex=True)

    for i, cfg in enumerate(configs):
        _, vol, _ = simulate_garch(
            n=1000,
            omega=0.0001,
            alpha=cfg["alpha"],
            beta=cfg["beta"],
            seed=42,
        )
        axes[i].plot(vol, linewidth=0.9)
        axes[i].set_ylabel("σ_t")
        axes[i].set_title(cfg["label"])
        axes[i].grid(alpha=0.3)

    axes[-1].set_xlabel("Time Step")
    plt.suptitle("GARCH(1,1): Effect of Persistence on Volatility Dynamics", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "garch_persistence_comparison.png", dpi=300)
    plt.show()


def plot_return_distribution(returns: np.ndarray) -> None:
    """
    Compare the GARCH return distribution to a normal distribution.
    GARCH returns exhibit fat tails — a key feature of real financial data.
    """
    from scipy.stats import norm

    mu_emp = returns.mean()
    std_emp = returns.std()
    x = np.linspace(returns.min(), returns.max(), 300)

    plt.figure(figsize=(9, 5))
    plt.hist(returns, bins=60, density=True, alpha=0.6, color="steelblue", label="GARCH Returns")
    plt.plot(x, norm.pdf(x, mu_emp, std_emp), color="crimson", linewidth=2, label="Normal Fit")
    plt.xlabel("Return")
    plt.ylabel("Density")
    plt.title("GARCH Return Distribution vs Normal — Fat Tails Visible")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "garch_return_distribution.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # ---- Simulate base GARCH(1,1) process ----
    returns, volatility, variance = simulate_garch(
        n=1000,
        omega=0.0001,
        alpha=0.10,
        beta=0.85,
        mu=0.0,
        sigma0=0.01,
        seed=42,
    )

    # ---- Plot 1: Returns + Conditional Volatility ----
    plot_returns_and_volatility(returns, volatility)

    # ---- Plot 2: Clustering signals (|r| and r^2) ----
    plot_volatility_clustering(returns)

    # ---- Plot 3: Compare different persistence levels ----
    compare_persistence_levels()

    # ---- Plot 4: Fat tails vs Normal ----
    plot_return_distribution(returns)