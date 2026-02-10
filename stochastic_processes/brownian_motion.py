import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path

# ---- DIRECTORIES ----
BASE_DIR = Path(__file__).parent
FIG_DIR = BASE_DIR / "figures" / "brownian"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def simulate_brownian_motion(T, N, n_paths, seed=42):
    rng = np.random.default_rng(seed)
    dt = T / N
    t = np.linspace(0, T, N + 1)

    Z = rng.standard_normal((n_paths, N))
    dW = np.sqrt(dt) * Z

    W = np.zeros((n_paths, N + 1))
    W[:, 1:] = np.cumsum(dW, axis=1)

    return t, W


def save_brownian_plots(t, W):
    n_paths = W.shape[0]

    # ---- STATIC PNG (for README) ----
    plt.figure(figsize=(10, 6))
    for i in range(n_paths):
        plt.plot(t, W[i], alpha=0.7)

    plt.title("Brownian Motion Paths (2D)")
    plt.xlabel("Time")
    plt.ylabel("W(t)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "brownian_motion_2d.png", dpi=300)
    plt.close()

    # ---- INTERACTIVE 3D (ROTATABLE HTML) ----
    fig = go.Figure()

    for i in range(n_paths):
        fig.add_trace(
            go.Scatter3d(
                x=t,
                y=[i] * len(t),
                z=W[i],
                mode="lines",
                line=dict(width=4),
            )
        )

    fig.update_layout(
        title="3D Brownian Motion Paths",
        scene=dict(
            xaxis_title="Time",
            yaxis_title="Path Index",
            zaxis_title="W(t)",
            bgcolor="rgba(200,200,200,0.85)",
        ),
        paper_bgcolor="rgba(200,200,200,1)",
    )

    fig.write_html(FIG_DIR / "brownian_motion_3d.html")


def main():
    t, W = simulate_brownian_motion(T=1.0, N=500, n_paths=20)
    save_brownian_plots(t, W)


if __name__ == "__main__":
    main()
