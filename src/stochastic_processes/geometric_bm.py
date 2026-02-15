import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path

# ---- DIRECTORIES ----
BASE_DIR = Path(__file__).parent
FIG_DIR = BASE_DIR / "figures" / "gbm"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def simulate_gbm(S0, mu, sigma, T, N, n_paths, seed=42):
    rng = np.random.default_rng(seed)
    dt = T / N
    t = np.linspace(0, T, N + 1)

    Z = rng.standard_normal((n_paths, N))
    dW = np.sqrt(dt) * Z

    W = np.zeros((n_paths, N + 1))
    W[:, 1:] = np.cumsum(dW, axis=1)

    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    return t, S


def save_gbm_plots(t, S):
    n_paths = S.shape[0]

    # ---- STATIC PNG ----
    plt.figure(figsize=(10, 6))
    for i in range(n_paths):
        plt.plot(t, S[i], alpha=0.7)

    plt.title("GBM Price Paths (2D)")
    plt.xlabel("Time (years)")
    plt.ylabel("Price S(t)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gbm_paths_2d.png", dpi=300)
    plt.close()

    # ---- INTERACTIVE 3D ----
    fig = go.Figure()

    for i in range(n_paths):
        fig.add_trace(
            go.Scatter3d(
                x=t,
                y=[i] * len(t),
                z=S[i],
                mode="lines",
                line=dict(width=4),
            )
        )

    fig.update_layout(
        title="3D Geometric Brownian Motion (GBM)",
        scene=dict(
            xaxis_title="Time",
            yaxis_title="Path Index",
            zaxis_title="Price S(t)",
            bgcolor="rgba(200,200,200,0.85)",
        ),
        paper_bgcolor="rgba(200,200,200,1)",
    )

    fig.write_html(FIG_DIR / "gbm_paths_3d.html")


def main():
    t, S = simulate_gbm(
        S0=100,
        mu=0.08,
        sigma=0.2,
        T=1.0,
        N=500,
        n_paths=20,
    )
    save_gbm_plots(t, S)


if __name__ == "__main__":
    main()
