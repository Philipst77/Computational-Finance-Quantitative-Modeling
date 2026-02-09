import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def simulate_gbm(S0: float, mu: float, sigma: float, T: float, N: int, n_paths: int, seed: int | None = 42):
    """
    Geometric Brownian Motion:
      S_t = S0 * exp( (mu - 0.5*sigma^2)*t + sigma*W_t )

    Returns:
      t: (N+1,)
      S: (n_paths, N+1)
      W: (n_paths, N+1)  (Brownian paths used)
    """
    if T <= 0 or N <= 0 or n_paths <= 0:
        raise ValueError("T, N, and n_paths must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    rng = np.random.default_rng(seed)
    dt = T / N
    t = np.linspace(0.0, T, N + 1)

    # Brownian increments and paths
    Z = rng.standard_normal((n_paths, N))
    dW = np.sqrt(dt) * Z
    W = np.zeros((n_paths, N + 1))
    W[:, 1:] = np.cumsum(dW, axis=1)

    # GBM price paths
    exponent = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(exponent)

    return t, S, W


def plot_gbm_3d_glassy(t, S, medium_gray=True):
    """
    3D plot:
      x = time
      y = path index
      z = price
    Rotatable by mouse (matplotlib 3D default).
    """
    n_paths, n_steps = S.shape

    fig = plt.figure(figsize=(11, 7))

    if medium_gray:
        fig.patch.set_facecolor((0.75, 0.75, 0.75))          # medium gray figure bg
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor((0.85, 0.85, 0.85, 0.85))           # lighter gray axes bg

        # Glass-like panes
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.set_facecolor((1, 1, 1, 0.25))
            axis.pane.set_edgecolor("gray")
    else:
        ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.plasma(np.linspace(0, 1, n_paths))

    for i in range(n_paths):
        ax.plot(
            t,
            np.full(n_steps, i),
            S[i],
            color=colors[i],
            linewidth=1.2,
            alpha=0.95,
        )

    ax.set_title("3D Geometric Brownian Motion (GBM) Price Paths", fontsize=15, pad=18)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Path Index")
    ax.set_zlabel("Price S(t)")

    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()


def main():
    # --- GBM parameters (edit these) ---
    S0 = 100.0      # initial price
    mu = 0.08       # drift (8% annual)
    sigma = 0.20    # volatility (20% annual)
    T = 1.0         # years
    N = 500         # steps
    n_paths = 20

    t, S, W = simulate_gbm(S0, mu, sigma, T, N, n_paths, seed=42)

    # 2D quick view (optional)
    plt.figure(figsize=(10, 6))
    for i in range(n_paths):
        plt.plot(t, S[i], linewidth=1, alpha=0.7)
    plt.title("GBM Price Paths (2D)")
    plt.xlabel("Time (years)")
    plt.ylabel("Price S(t)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 3D glassy view (rotatable)
    plot_gbm_3d_glassy(t, S, medium_gray=True)

    # Sanity checks
    print("Min price across all paths:", S.min())
    print("Mean terminal price:", S[:, -1].mean())
    print("Std terminal price:", S[:, -1].std())


if __name__ == "__main__":
    main()
