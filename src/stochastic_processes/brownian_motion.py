import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path


# ---- DIRECTORIES ----
BASE_DIR = Path(__file__).parent
FIG_DIR = BASE_DIR / "figures" / "brownian"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# BROWNIAN MOTION (Wiener Process)
# ==============================================================================
# A Brownian motion W(t) satisfies:
#   W(0) = 0
#   Increments dW = W(t+dt) - W(t) ~ N(0, dt)
#   Increments are independent
#   Variance grows linearly: Var[W(t)] = t
#
# It is the raw noise engine — not a price model itself, but the foundation
# for GBM, Black-Scholes, and all stochastic volatility models.
# ==============================================================================


def simulate_brownian_motion(
    T: float = 1.0,
    N: int = 500,
    n_paths: int = 20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate standard Brownian motion paths.

    Parameters
    ----------
    T       : time horizon
    N       : number of time steps
    n_paths : number of independent paths
    seed    : random seed

    Returns
    -------
    t : time grid, shape (N+1,)
    W : Brownian motion paths, shape (n_paths, N+1)
    """
    rng = np.random.default_rng(seed)
    dt = T / N
    t = np.linspace(0, T, N + 1)

    Z = rng.standard_normal((n_paths, N))
    dW = np.sqrt(dt) * Z

    W = np.zeros((n_paths, N + 1))
    W[:, 1:] = np.cumsum(dW, axis=1)

    return t, W


def plot_paths_2d(t: np.ndarray, W: np.ndarray) -> None:
    """Plot standard 2D Brownian motion paths."""
    n_paths = W.shape[0]

    plt.figure(figsize=(10, 6))
    for i in range(n_paths):
        plt.plot(t, W[i], alpha=0.7, linewidth=0.9)

    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.title("Brownian Motion Paths")
    plt.xlabel("Time")
    plt.ylabel("W(t)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "brownian_motion_2d.png", dpi=300)
    plt.show()


def plot_variance_growth(T: float = 1.0, N: int = 500, n_paths: int = 2000) -> None:
    """
    Verify that Var[W(t)] = t — a fundamental property of Brownian motion.
    This is the 'linear variance growth' property, and it's what makes BM
    different from a simple random walk scaled by sqrt(t).
    """
    t, W = simulate_brownian_motion(T=T, N=N, n_paths=n_paths, seed=0)

    empirical_var = W.var(axis=0)   # variance across paths at each time step
    theoretical_var = t             # should match exactly

    plt.figure(figsize=(9, 5))
    plt.plot(t, empirical_var, label="Empirical Var[W(t)]", linewidth=1.5)
    plt.plot(t, theoretical_var, label="Theoretical: t", linestyle="--", linewidth=1.5)
    plt.xlabel("Time t")
    plt.ylabel("Variance")
    plt.title(f"Brownian Motion: Variance Growth (n={n_paths:,} paths)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "brownian_variance_growth.png", dpi=300)
    plt.show()


def plot_increment_distribution(T: float = 1.0, N: int = 500, n_paths: int = 5000) -> None:
    """
    Show that Brownian increments are normally distributed.
    This is the core assumption that justifies using BM in financial models.
    """
    from scipy.stats import norm

    t, W = simulate_brownian_motion(T=T, N=N, n_paths=n_paths, seed=1)

    # Pick increments over a fixed interval dt
    dt = T / N
    increments = np.diff(W, axis=1).flatten()

    x = np.linspace(increments.min(), increments.max(), 300)
    theoretical = norm.pdf(x, loc=0, scale=np.sqrt(dt))

    plt.figure(figsize=(9, 5))
    plt.hist(increments, bins=80, density=True, alpha=0.6, color="steelblue", label="Empirical increments")
    plt.plot(x, theoretical, color="crimson", linewidth=2, label=f"N(0, dt)  dt={dt:.4f}")
    plt.xlabel("Increment dW")
    plt.ylabel("Density")
    plt.title("Brownian Increments are Normally Distributed")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "brownian_increment_distribution.png", dpi=300)
    plt.show()


def plot_time_horizon_comparison() -> None:
    """
    Show how the spread of paths grows with time horizon.
    Longer T → wider fan-out because uncertainty accumulates as sqrt(T).
    """
    horizons = [0.25, 0.5, 1.0, 2.0]
    fig, axes = plt.subplots(1, len(horizons), figsize=(14, 4), sharey=False)

    for ax, T in zip(axes, horizons):
        t, W = simulate_brownian_motion(T=T, N=500, n_paths=30, seed=42)
        for i in range(W.shape[0]):
            ax.plot(t, W[i], alpha=0.5, linewidth=0.7)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_title(f"T = {T}")
        ax.set_xlabel("Time")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("W(t)")
    plt.suptitle("Brownian Motion: Spread Grows with Time Horizon", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "brownian_horizon_comparison.png", dpi=300)
    plt.show()


def save_3d_plot(t: np.ndarray, W: np.ndarray) -> None:
    """Save interactive 3D Brownian motion plot as HTML."""
    fig = go.Figure()

    for i in range(W.shape[0]):
        fig.add_trace(go.Scatter3d(
            x=t,
            y=[i] * len(t),
            z=W[i],
            mode="lines",
            line=dict(width=3),
            showlegend=False,
        ))

    fig.update_layout(
        title="3D Brownian Motion Paths",
        scene=dict(
            xaxis_title="Time",
            yaxis_title="Path Index",
            zaxis_title="W(t)",
        ),
    )

    fig.write_html(FIG_DIR / "brownian_motion_3d.html")
    print(f"Saved interactive 3D plot → {FIG_DIR / 'brownian_motion_3d.html'}")


if __name__ == "__main__":
    # ---- Simulate ----
    t, W = simulate_brownian_motion(T=1.0, N=500, n_paths=20, seed=42)

    # ---- Plot 1: Standard 2D paths ----
    plot_paths_2d(t, W)

    # ---- Plot 2: Variance grows linearly with t ----
    plot_variance_growth(T=1.0, N=500, n_paths=2000)

    # ---- Plot 3: Increments are normally distributed ----
    plot_increment_distribution(T=1.0, N=500, n_paths=5000)

    # ---- Plot 4: Spread grows with time horizon ----
    plot_time_horizon_comparison()

    # ---- Plot 5: Interactive 3D ----
    save_3d_plot(t, W)