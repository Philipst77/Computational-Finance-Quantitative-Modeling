import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


# Get the directory where this script is located
BASE_DIR = Path(__file__).resolve().parent

FIG_DIR = BASE_DIR / "Figures" / "black_scholes"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ===============================
# Black-Scholes Core Functions
# ===============================

def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def call_price(S, K, T, r, sigma):
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)
    return S * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2)


def put_price(S, K, T, r, sigma):
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-D2) - S * norm.cdf(-D1)


def delta_call(S, K, T, r, sigma):
    return norm.cdf(d1(S, K, T, r, sigma))


def gamma(S, K, T, r, sigma):
    D1 = d1(S, K, T, r, sigma)
    return norm.pdf(D1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    D1 = d1(S, K, T, r, sigma)
    return S * norm.pdf(D1) * np.sqrt(T)


def theta_call(S, K, T, r, sigma):
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)
    term1 = - (S * norm.pdf(D1) * sigma) / (2 * np.sqrt(T))
    term2 = - r * K * np.exp(-r * T) * norm.cdf(D2)
    return term1 + term2



if __name__ == "__main__":

    print("Running from:", Path(__file__).resolve())
    print("Saving figures to:", FIG_DIR)

    # Parameters
    K = 100
    T = 1
    r = 0.05

    # Grid ranges
    S_vals = np.linspace(50, 150, 100)
    sigma_vals = np.linspace(0.05, 0.6, 100)

    S_grid, sigma_grid = np.meshgrid(S_vals, sigma_vals)

    # Compute Call Surface
    V = call_price(S_grid, K, T, r, sigma_grid)

    # 3D Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(S_grid, sigma_grid, V, cmap="viridis")

    ax.set_xlabel("Stock Price (S)")
    ax.set_ylabel("Volatility (σ)")
    ax.set_zlabel("Call Price")
    ax.set_title("Black-Scholes Call Option Surface")

    # ===== SAVE FIGURE =====
    output_path = FIG_DIR / "black_scholes_surface.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"Figure saved to: {output_path}")

    plt.show()