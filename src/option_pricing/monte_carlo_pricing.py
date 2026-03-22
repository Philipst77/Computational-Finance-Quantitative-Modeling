import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm
from pathlib import Path


# ---- DIRECTORIES ----
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "Figures" / "monte_carlo_pricing"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# MONTE CARLO OPTION PRICING
# ==============================================================================
# Monte Carlo prices options by simulating many asset price paths, computing
# the option payoff on each, and averaging the discounted result.
#
# For a European call:
#   Price = e^(-rT) * E[max(S(T) - K, 0)]
#
# This approach generalizes to any payoff function — path-dependent options,
# exotic options, and options under non-GBM dynamics.
#
# Comparison with Black-Scholes:
#   - BS is closed-form and exact (under its assumptions)
#   - MC is approximate but universally applicable
#   - MC error scales as 1/sqrt(n_paths) — more paths = more accuracy
# ==============================================================================


def bs_call_price(S, K, r, sigma, T) -> float | np.ndarray:
    """
    Black-Scholes closed-form call price.
    Used as the reference to validate Monte Carlo results.
    """
    if np.isscalar(T) and T <= 0:
        return np.maximum(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def mc_call_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int = 100_000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Monte Carlo European call price.

    Parameters
    ----------
    S0      : current asset price
    K       : strike price
    r       : risk-free rate
    sigma   : volatility
    T       : time to expiration (years)
    n_paths : number of simulation paths
    seed    : random seed

    Returns
    -------
    price     : Monte Carlo call price estimate
    std_error : standard error of the estimate (95% CI ≈ ± 1.96 * std_error)
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    S_T = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(S_T - K, 0.0)
    discounted = np.exp(-r * T) * payoffs
    price = discounted.mean()
    std_error = discounted.std(ddof=1) / np.sqrt(n_paths)
    return price, std_error


def compare_mc_vs_bs() -> None:
    """
    Compare Monte Carlo and Black-Scholes prices across a range of spot prices.
    MC should track BS closely — any divergence indicates insufficient paths.
    """
    K, r, sigma, T = 100, 0.05, 0.2, 1.0
    S_vals = np.linspace(60, 140, 30)

    mc_prices, mc_errors, bs_prices = [], [], []

    for S0 in S_vals:
        mc_p, mc_e = mc_call_price(S0, K, r, sigma, T, n_paths=50_000)
        bs_p = bs_call_price(S0, K, r, sigma, T)
        mc_prices.append(mc_p)
        mc_errors.append(mc_e * 1.96)   # 95% CI half-width
        bs_prices.append(bs_p)

    mc_prices = np.array(mc_prices)
    mc_errors = np.array(mc_errors)
    bs_prices = np.array(bs_prices)

    plt.figure(figsize=(10, 6))
    plt.plot(S_vals, bs_prices, color="steelblue", linewidth=2, label="Black-Scholes (exact)")
    plt.plot(S_vals, mc_prices, color="crimson", linewidth=1.5,
             linestyle="--", label="Monte Carlo (50k paths)")
    plt.fill_between(S_vals, mc_prices - mc_errors, mc_prices + mc_errors,
                     alpha=0.2, color="crimson", label="95% CI")
    plt.axvline(K, color="gray", linestyle=":", linewidth=1)
    plt.xlabel("Stock Price S")
    plt.ylabel("Call Price")
    plt.title("Monte Carlo vs Black-Scholes: Call Option Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "mc_vs_bs_call_price.png", dpi=300)
    plt.show()


def plot_convergence() -> None:
    """
    Show how MC price converges to the BS price as n_paths increases.
    Error shrinks as 1/sqrt(n) — the fundamental MC convergence rate.
    """
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    bs_price = bs_call_price(S0, K, r, sigma, T)

    path_counts = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
    mc_prices, mc_errors = [], []

    for n in path_counts:
        p, e = mc_call_price(S0, K, r, sigma, T, n_paths=n, seed=42)
        mc_prices.append(p)
        mc_errors.append(e * 1.96)
        print(f"n={n:>8} | MC={p:.4f} | BS={bs_price:.4f} | Error={abs(p-bs_price):.4f}")

    mc_prices = np.array(mc_prices)
    mc_errors = np.array(mc_errors)

    plt.figure(figsize=(10, 5))
    plt.fill_between(path_counts, mc_prices - mc_errors, mc_prices + mc_errors,
                     alpha=0.2, color="steelblue", label="95% CI")
    plt.plot(path_counts, mc_prices, marker="o", linewidth=1.5,
             color="steelblue", label="MC Price")
    plt.axhline(bs_price, color="crimson", linestyle="--",
                linewidth=1.5, label=f"BS Price = {bs_price:.4f}")
    plt.xscale("log")
    plt.xlabel("Number of Paths (log scale)")
    plt.ylabel("Call Price")
    plt.title("Monte Carlo Convergence to Black-Scholes Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "mc_convergence.png", dpi=300)
    plt.show()


def plot_payoff_distribution() -> None:
    """
    Show the distribution of discounted payoffs.
    Most paths expire worthless (out of the money) — the mean of this
    distribution is the option price.
    """
    S0, K, r, sigma, T = 100, 105, 0.05, 0.2, 1.0
    rng = np.random.default_rng(42)
    Z = rng.standard_normal(100_000)
    S_T = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.exp(-r * T) * np.maximum(S_T - K, 0.0)

    bs_price = bs_call_price(S0, K, r, sigma, T)
    mc_price = payoffs.mean()

    plt.figure(figsize=(9, 5))
    plt.hist(payoffs[payoffs > 0], bins=80, density=True,
             alpha=0.65, color="steelblue", label="Non-zero payoffs")
    plt.axvline(mc_price, color="crimson", linestyle="--",
                linewidth=1.8, label=f"MC Price = {mc_price:.3f}")
    plt.axvline(bs_price, color="black", linestyle=":",
                linewidth=1.5, label=f"BS Price = {bs_price:.3f}")
    pct_itm = (payoffs > 0).mean() * 100
    plt.xlabel("Discounted Payoff")
    plt.ylabel("Density (in-the-money paths only)")
    plt.title(f"Payoff Distribution — {pct_itm:.1f}% of paths expire in the money")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "mc_payoff_distribution.png", dpi=300)
    plt.show()


def save_price_surface_animation() -> None:
    """
    Animate the BS call price surface as time to expiration decreases toward 0.
    Shows how the surface collapses to the intrinsic value (hockey stick) at expiry.
    """
    K, r = 100, 0.05
    S_vals = np.linspace(60, 140, 30)
    sigma_vals = np.linspace(0.1, 0.6, 30)
    S_grid, sigma_grid = np.meshgrid(S_vals, sigma_vals)
    T_total = 1.0
    time_values = np.linspace(0.01, T_total, 50)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()
        t_remaining = T_total - time_values[frame]
        surface = bs_call_price(S_grid, K, r, sigma_grid, max(t_remaining, 1e-6))
        ax.plot_surface(S_grid, sigma_grid, surface, cmap="viridis")
        ax.set_title(f"Call Price Surface  (T remaining = {t_remaining:.2f})")
        ax.set_xlabel("Stock Price (S)")
        ax.set_ylabel("Volatility (σ)")
        ax.set_zlabel("Option Price")
        ax.set_zlim(0, 50)

    ani = FuncAnimation(fig, update, frames=len(time_values), interval=100)

    output_path = FIG_DIR / "monte_carlo_pricing.mp4"
    print("Saving animation (requires ffmpeg)...")
    ani.save(output_path, writer="ffmpeg", fps=15)
    print(f"Saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    # ---- Single price comparison ----
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    mc_p, mc_e = mc_call_price(S0, K, r, sigma, T)
    bs_p = bs_call_price(S0, K, r, sigma, T)

    print(f"Monte Carlo price: {mc_p:.4f}  (±{1.96*mc_e:.4f} 95% CI)")
    print(f"Black-Scholes:     {bs_p:.4f}")
    print(f"Difference:        {abs(mc_p - bs_p):.4f}")
    print()

    # ---- Plot 1: MC vs BS across spot prices ----
    compare_mc_vs_bs()

    # ---- Plot 2: Convergence with n_paths ----
    plot_convergence()

    # ---- Plot 3: Payoff distribution ----
    plot_payoff_distribution()

    # ---- Plot 4: Animated surface (requires ffmpeg) ----
    save_price_surface_animation()