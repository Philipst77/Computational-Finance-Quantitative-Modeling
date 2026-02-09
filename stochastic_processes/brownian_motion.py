import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

def simulate_brownian_motion(T: float, N: int, n_paths: int, seed: int | None = 0):
    """
    Simulate Brownian motion paths W(t).

    Parameters
    ----------
    T : float
        Total time (e.g. 1.0)
    N : int
        Number of time steps
    n_paths : int
        Number of simulated paths
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    t : ndarray, shape (N+1,)
        Time grid
    W : ndarray, shape (n_paths, N+1)
        Brownian motion paths
    """
    if T <= 0:
        raise ValueError("T must be positive")
    if N <= 0:
        raise ValueError("N must be positive")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    rng = np.random.default_rng(seed)
    dt = T / N

    # Step 1: generate standard normal noise
    Z = rng.standard_normal((n_paths, N))

    # Step 2: scale by sqrt(dt)
    dW = np.sqrt(dt) * Z

    # Step 3: cumulative sum to build paths
    W = np.zeros((n_paths, N + 1))
    W[:, 1:] = np.cumsum(dW, axis=1)

    # Time grid
    t = np.linspace(0.0, T, N + 1)

    return t, W


def main():
    # Parameters
    T = 1.0
    N = 500
    n_paths = 20
    dt = T / N
    t = np.linspace(0, T, N + 1)

    np.random.seed(42)
    dW = np.sqrt(dt) * np.random.randn(n_paths, N + 1)
    W = np.cumsum(dW, axis=1)

    # ---- FIGURE SETUP ----
    fig = plt.figure(figsize=(11, 7))
    fig.patch.set_facecolor((0.75, 0.75, 0.75))  # medium gray background

    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor((0.85, 0.85, 0.85, 0.85))  # lighter gray, glassy look

    # Glass-like transparent panes
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1, 1, 1, 0.25))
        axis.pane.set_edgecolor("gray")

    # ---- PLOT PATHS ----
    colors = plt.cm.plasma(np.linspace(0, 1, n_paths))

    for i in range(n_paths):
        ax.plot(
            t,
            np.full(N + 1, i),
            W[i],
            color=colors[i],
            linewidth=1.2,
            alpha=0.95,
        )

    # ---- LABELS ----
    ax.set_title("3D Brownian Motion Paths", fontsize=15, pad=18)
    ax.set_xlabel("Time")
    ax.set_ylabel("Path Index")
    ax.set_zlabel("W(t)")

    # Subtle grid
    ax.grid(True, alpha=0.35)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
