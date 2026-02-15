import os
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D



def bs_call_price(S, K, r, sigma, T):
    if T <= 0:
        return np.maximum(S - K, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)



def mc_call_price(S0, K, r, sigma, T, n_paths=100_000, seed=42):
    rng = np.random.default_rng(seed)

    Z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    payoff = np.maximum(ST - K, 0.0)
    discounted = np.exp(-r * T) * payoff

    price = discounted.mean()
    std_error = discounted.std(ddof=1) / np.sqrt(n_paths)

    return price, std_error



if __name__ == "__main__":

    # Parameters
    K = 100
    r = 0.05
    T = 1.0

    S0 = 100
    sigma = 0.2

    mc_price, se = mc_call_price(S0, K, r, sigma, T)
    bs_price = bs_call_price(S0, K, r, sigma, T)

    print("Monte Carlo Price:", mc_price)
    print("Standard Error:", se)
    print("Black–Scholes Price:", bs_price)
    print("Difference:", abs(mc_price - bs_price))


    # Surface grid
    S_values = np.linspace(60, 140, 30)
    sigma_values = np.linspace(0.1, 0.6, 30)
    S_grid, sigma_grid = np.meshgrid(S_values, sigma_values)

    time_values = np.linspace(0.01, T, 50)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()

        t_remaining = T - time_values[frame]
        surface = np.zeros_like(S_grid)

        for i in range(len(sigma_values)):
            for j in range(len(S_values)):
                surface[i, j] = bs_call_price(
                    S_values[j],
                    K,
                    r,
                    sigma_values[i],
                    t_remaining
                )

        ax.plot_surface(
            S_grid,
            sigma_grid,
            surface,
            cmap="viridis"
        )

        ax.set_title(f"Option Price Surface (T = {round(t_remaining, 2)})")
        ax.set_xlabel("Stock Price (S)")
        ax.set_ylabel("Volatility (σ)")
        ax.set_zlabel("Option Price")

    ani = FuncAnimation(fig, update, frames=len(time_values), interval=100)


    # --------------------------
    # CREATE FIGURES FOLDER
    # --------------------------
    output_dir = "Figures"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "monte_carlo_pricing.mp4")

    print("Saving animation...")
    ani.save(output_path, writer="ffmpeg", fps=15)
    print(f"Saved to {output_path}")

    plt.show()
