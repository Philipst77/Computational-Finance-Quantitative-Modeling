import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path


# ---- DIRECTORIES ----
BASE_DIR = Path(__file__).parent
FIG_DIR = BASE_DIR / "figures" / "gbm"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# GEOMETRIC BROWNIAN MOTION (GBM)
# ==============================================================================
# GBM models asset prices as a lognormal process:
#
#   dS = mu * S * dt  +  sigma * S * dW
#
# Exact solution (Ito's lemma):
#   S(t) = S0 * exp( (mu - 0.5*sigma^2) * t  +  sigma * W(t) )
#
# Parameters:
#   mu    — drift (expected return per unit time)
#   sigma — volatility (std of log returns per unit time)
#
# Key properties:
#   - S(t) is always positive (no negative prices)
#   - Log returns are normally distributed: log(S(t)/S(0)) ~ N(...)
#   - E[S(t)] = S0 * exp(mu * t)
#   - The Ito correction (-0.5*sigma^2) accounts for Jensen's inequality
# ==============================================================================


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    N: int,
    n_paths: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate Geometric Brownian Motion using the exact solution.

    Parameters
    ----------
    S0      : initial asset price
    mu      : drift (annualized)
    sigma   : volatility (annualized)
    T       : time horizon (years)
    N       : number of time steps
    n_paths : number of simulation paths
    seed    : random seed

    Returns
    -------
    t : time grid, shape (N+1,)
    S : price paths, shape (n_paths, N+1)
    """
    rng = np.random.default_rng(seed)
    dt = T / N
    t = np.linspace(0, T, N + 1)

    Z = rng.standard_normal((n_paths, N))
    dW = np.sqrt(dt) * Z

    W = np.zeros((n_paths, N + 1))
    W[:, 1:] = np.cumsum(dW, axis=1)

    S = S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W)
    return t, S


def plot_paths_2d(t: np.ndarray, S: np.ndarray) -> None:
    """Plot GBM price paths with mean path overlay."""
    plt.figure(figsize=(10, 6))
    for i in range(S.shape[0]):
        plt.plot(t, S[i], alpha=0.6, linewidth=0.8)

    # Overlay the mean path
    plt.plot(t, S.mean(axis=0), color="black", linewidth=2.0,
             linestyle="--", label="Mean path")

    plt.title("Geometric Brownian Motion Price Paths")
    plt.xlabel("Time (years)")
    plt.ylabel("Price S(t)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gbm_paths_2d.png", dpi=300)
    plt.show()


def plot_log_return_distribution(
    S0: float = 100.0,
    mu: float = 0.08,
    sigma: float = 0.2,
    T: float = 1.0,
    N: int = 252,
    n_paths: int = 20_000,
) -> None:
    """
    Show that GBM log returns are normally distributed.
    This is a core testable property: log(S(T)/S0) ~ N((mu - 0.5*sigma^2)*T, sigma^2*T)
    """
    from scipy.stats import norm

    t, S = simulate_gbm(S0=S0, mu=mu, sigma=sigma, T=T, N=N, n_paths=n_paths, seed=7)

    log_returns = np.log(S[:, -1] / S0)

    # Theoretical parameters
    mean_theory = (mu - 0.5 * sigma ** 2) * T
    std_theory = sigma * np.sqrt(T)
    x = np.linspace(log_returns.min(), log_returns.max(), 300)

    plt.figure(figsize=(9, 5))
    plt.hist(log_returns, bins=80, density=True, alpha=0.6,
             color="steelblue", label="Simulated log returns")
    plt.plot(x, norm.pdf(x, mean_theory, std_theory), color="crimson",
             linewidth=2, label=f"N({mean_theory:.3f}, {std_theory:.3f}²)")
    plt.xlabel("log(S(T) / S₀)")
    plt.ylabel("Density")
    plt.title("GBM: Log Returns are Normally Distributed")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gbm_log_return_distribution.png", dpi=300)
    plt.show()

    print(f"Empirical  mean: {log_returns.mean():.4f}  |  Theoretical: {mean_theory:.4f}")
    print(f"Empirical  std:  {log_returns.std():.4f}  |  Theoretical: {std_theory:.4f}")


def plot_volatility_sensitivity(
    S0: float = 100.0,
    mu: float = 0.08,
    T: float = 1.0,
    N: int = 500,
    n_paths: int = 30,
) -> None:
    """
    Show how sigma controls the fan-out of price paths.
    Higher sigma → wider spread, more uncertainty about terminal price.
    """
    sigmas = [0.1, 0.3, 0.6]
    colors = ["steelblue", "darkorange", "crimson"]

    fig, axes = plt.subplots(1, len(sigmas), figsize=(14, 5), sharey=False)

    for ax, sigma, color in zip(axes, sigmas, colors):
        t, S = simulate_gbm(S0=S0, mu=mu, sigma=sigma, T=T, N=N, n_paths=n_paths, seed=42)
        for i in range(n_paths):
            ax.plot(t, S[i], alpha=0.45, linewidth=0.7, color=color)
        ax.plot(t, S.mean(axis=0), color="black", linewidth=1.5, linestyle="--")
        ax.set_title(f"σ = {sigma}")
        ax.set_xlabel("Time (years)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Price S(t)")
    plt.suptitle("GBM: Effect of Volatility σ on Price Paths", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gbm_volatility_sensitivity.png", dpi=300)
    plt.show()


def plot_drift_sensitivity(
    S0: float = 100.0,
    sigma: float = 0.2,
    T: float = 1.0,
    N: int = 500,
    n_paths: int = 500,
) -> None:
    """
    Show how mu shifts the expected price path without changing the spread.
    The mean path follows E[S(t)] = S0 * exp(mu * t).
    """
    mus = [-0.1, 0.0, 0.05, 0.15]
    colors = ["crimson", "gray", "steelblue", "green"]

    plt.figure(figsize=(10, 6))

    for mu, color in zip(mus, colors):
        t, S = simulate_gbm(S0=S0, mu=mu, sigma=sigma, T=T, N=N, n_paths=n_paths, seed=42)
        mean_path = S.mean(axis=0)
        theoretical = S0 * np.exp(mu * t)
        plt.plot(t, mean_path, color=color, linewidth=1.8,
                 label=f"μ={mu:+.2f}  (simulated mean)")
        plt.plot(t, theoretical, color=color, linewidth=1.0,
                 linestyle="--", alpha=0.6)

    plt.xlabel("Time (years)")
    plt.ylabel("Average Price")
    plt.title("GBM: Effect of Drift μ on Expected Price Path\n(solid=simulated, dashed=theoretical)")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gbm_drift_sensitivity.png", dpi=300)
    plt.show()


def save_3d_plot(t: np.ndarray, S: np.ndarray) -> None:
    """Save interactive 3D GBM plot as HTML."""
    fig = go.Figure()

    for i in range(S.shape[0]):
        fig.add_trace(go.Scatter3d(
            x=t,
            y=[i] * len(t),
            z=S[i],
            mode="lines",
            line=dict(width=3),
            showlegend=False,
        ))

    fig.update_layout(
        title="3D Geometric Brownian Motion",
        scene=dict(
            xaxis_title="Time",
            yaxis_title="Path Index",
            zaxis_title="Price S(t)",
        ),
    )

    fig.write_html(FIG_DIR / "gbm_paths_3d.html")
    print(f"Saved interactive 3D plot → {FIG_DIR / 'gbm_paths_3d.html'}")


if __name__ == "__main__":
    # ---- Simulate base paths ----
    t, S = simulate_gbm(S0=100.0, mu=0.08, sigma=0.2, T=1.0, N=500, n_paths=20, seed=42)

    # ---- Plot 1: 2D paths with mean ----
    plot_paths_2d(t, S)

    # ---- Plot 2: Log returns are normal ----
    plot_log_return_distribution()

    # ---- Plot 3: Volatility sensitivity ----
    plot_volatility_sensitivity()

    # ---- Plot 4: Drift sensitivity ----
    plot_drift_sensitivity()

    # ---- Plot 5: Interactive 3D ----
    save_3d_plot(t, S)