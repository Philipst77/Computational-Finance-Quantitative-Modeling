import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path


# ---- DIRECTORIES ----
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "Figures" / "black_scholes"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# BLACK-SCHOLES MODEL
# ==============================================================================
# The Black-Scholes formula prices European options under the assumption that
# the underlying follows GBM with constant volatility.
#
# Call price:  C = S * N(d1) - K * e^(-rT) * N(d2)
# Put price:   P = K * e^(-rT) * N(-d2) - S * N(-d1)
#
# Where:
#   d1 = [log(S/K) + (r + 0.5*sigma^2)*T] / (sigma*sqrt(T))
#   d2 = d1 - sigma*sqrt(T)
#
# Parameters:
#   S     — current asset price
#   K     — strike price
#   T     — time to expiration (years)
#   r     — risk-free rate
#   sigma — volatility (annualized)
#
# Greeks measure sensitivity of the option price to each parameter:
#   Delta — sensitivity to S         (dC/dS)
#   Gamma — rate of change of Delta  (d²C/dS²)
#   Vega  — sensitivity to sigma     (dC/dsigma)
#   Theta — sensitivity to T         (dC/dT)
# ==============================================================================


# ---- Core d1 / d2 ----

def _d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def _d2(S, K, T, r, sigma):
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


# ---- Pricing ----

def call_price(S, K, T, r, sigma):
    """Black-Scholes European call price."""
    D1, D2 = _d1(S, K, T, r, sigma), _d2(S, K, T, r, sigma)
    return S * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2)


def put_price(S, K, T, r, sigma):
    """Black-Scholes European put price."""
    D1, D2 = _d1(S, K, T, r, sigma), _d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-D2) - S * norm.cdf(-D1)


# ---- Greeks ----

def delta_call(S, K, T, r, sigma):
    """Delta of a call: rate of change of call price w.r.t. S."""
    return norm.cdf(_d1(S, K, T, r, sigma))


def delta_put(S, K, T, r, sigma):
    """Delta of a put: always negative (put price falls as S rises)."""
    return norm.cdf(_d1(S, K, T, r, sigma)) - 1


def gamma(S, K, T, r, sigma):
    """Gamma: rate of change of Delta w.r.t. S. Same for calls and puts."""
    return norm.pdf(_d1(S, K, T, r, sigma)) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """Vega: sensitivity of option price to volatility."""
    return S * norm.pdf(_d1(S, K, T, r, sigma)) * np.sqrt(T)


def theta_call(S, K, T, r, sigma):
    """Theta of a call: time decay (price lost per unit time)."""
    D1, D2 = _d1(S, K, T, r, sigma), _d2(S, K, T, r, sigma)
    term1 = -(S * norm.pdf(D1) * sigma) / (2 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * norm.cdf(D2)
    return term1 + term2


def put_call_parity_check(S, K, T, r, sigma) -> None:
    """
    Verify put-call parity: C - P = S - K*e^(-rT)
    This is a model-independent relationship — a useful sanity check.
    """
    C = call_price(S, K, T, r, sigma)
    P = put_price(S, K, T, r, sigma)
    lhs = C - P
    rhs = S - K * np.exp(-r * T)
    print(f"Put-Call Parity Check:")
    print(f"  C - P = {lhs:.6f}")
    print(f"  S - K*e^(-rT) = {rhs:.6f}")
    print(f"  Difference: {abs(lhs - rhs):.2e}  {'✓ PASS' if abs(lhs - rhs) < 1e-8 else '✗ FAIL'}")


# ---- Plots ----

def plot_price_surface() -> None:
    """
    3D surface of call price as a function of stock price S and volatility sigma.
    Shows how the option price landscape changes across the parameter space.
    """
    K, T, r = 100, 1.0, 0.05
    S_vals = np.linspace(50, 150, 100)
    sigma_vals = np.linspace(0.05, 0.6, 100)
    S_grid, sigma_grid = np.meshgrid(S_vals, sigma_vals)
    V = call_price(S_grid, K, T, r, sigma_grid)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(S_grid, sigma_grid, V, cmap="viridis")
    ax.set_xlabel("Stock Price (S)")
    ax.set_ylabel("Volatility (σ)")
    ax.set_zlabel("Call Price")
    ax.set_title("Black-Scholes Call Option Price Surface")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "black_scholes_surface.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_price_vs_spot() -> None:
    """
    Call and put prices as a function of spot price S.
    Shows the asymmetric payoff profiles and how they behave around the strike.
    """
    K, T, r, sigma = 100, 1.0, 0.05, 0.2
    S_vals = np.linspace(50, 150, 300)

    calls = call_price(S_vals, K, T, r, sigma)
    puts  = put_price(S_vals, K, T, r, sigma)
    intrinsic_call = np.maximum(S_vals - K, 0)
    intrinsic_put  = np.maximum(K - S_vals, 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(S_vals, calls, color="steelblue", linewidth=2, label="Call Price")
    axes[0].plot(S_vals, intrinsic_call, color="steelblue", linewidth=1,
                 linestyle="--", alpha=0.6, label="Intrinsic Value")
    axes[0].axvline(K, color="gray", linestyle=":", linewidth=1)
    axes[0].set_title("Call Option Price vs Spot")
    axes[0].set_xlabel("Stock Price S")
    axes[0].set_ylabel("Option Price")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(S_vals, puts, color="crimson", linewidth=2, label="Put Price")
    axes[1].plot(S_vals, intrinsic_put, color="crimson", linewidth=1,
                 linestyle="--", alpha=0.6, label="Intrinsic Value")
    axes[1].axvline(K, color="gray", linestyle=":", linewidth=1)
    axes[1].set_title("Put Option Price vs Spot")
    axes[1].set_xlabel("Stock Price S")
    axes[1].set_ylabel("Option Price")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle(f"Black-Scholes Prices  (K={K}, T={T}, r={r}, σ={sigma})", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "bs_price_vs_spot.png", dpi=300)
    plt.show()


def plot_greeks() -> None:
    """
    Plot Delta, Gamma, Vega, and Theta as functions of spot price S.
    Helps build intuition for how each sensitivity behaves around the strike.
    """
    K, T, r, sigma = 100, 1.0, 0.05, 0.2
    S_vals = np.linspace(50, 150, 300)

    greeks = {
        "Delta (Call)":  delta_call(S_vals, K, T, r, sigma),
        "Delta (Put)":   delta_put(S_vals, K, T, r, sigma),
        "Gamma":         gamma(S_vals, K, T, r, sigma),
        "Vega":          vega(S_vals, K, T, r, sigma),
        "Theta (Call)":  theta_call(S_vals, K, T, r, sigma),
    }
    colors = ["steelblue", "crimson", "darkorange", "green", "purple"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, (label, values), color in zip(axes, greeks.items(), colors):
        ax.plot(S_vals, values, color=color, linewidth=1.8)
        ax.axvline(K, color="gray", linestyle=":", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(label)
        ax.set_xlabel("Stock Price S")
        ax.grid(alpha=0.3)

    axes[-1].axis("off")   # hide the unused 6th panel
    plt.suptitle(f"Black-Scholes Greeks  (K={K}, T={T}, r={r}, σ={sigma})", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "bs_greeks.png", dpi=300)
    plt.show()


def plot_time_decay() -> None:
    """
    Show how call price decays as time to expiration decreases.
    Theta (time decay) accelerates near expiration — important for options traders.
    """
    K, r, sigma, S = 100, 0.05, 0.2, 100
    S_vals = np.linspace(60, 140, 300)
    time_to_expiry = [1.0, 0.5, 0.25, 0.1, 0.01]
    colors = ["steelblue", "darkorange", "green", "crimson", "purple"]

    plt.figure(figsize=(10, 6))
    for T_val, color in zip(time_to_expiry, colors):
        prices = call_price(S_vals, K, T_val, r, sigma)
        plt.plot(S_vals, prices, color=color, linewidth=1.8, label=f"T = {T_val}")

    plt.plot(S_vals, np.maximum(S_vals - K, 0), color="black",
             linestyle="--", linewidth=1.2, label="Intrinsic value")
    plt.axvline(K, color="gray", linestyle=":", linewidth=1)
    plt.xlabel("Stock Price S")
    plt.ylabel("Call Price")
    plt.title("Black-Scholes: Time Decay of Call Option Price")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "bs_time_decay.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # ---- Sanity check ----
    put_call_parity_check(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
    print()

    # ---- Sample prices ----
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    print(f"Call price: {call_price(S, K, T, r, sigma):.4f}")
    print(f"Put  price: {put_price(S, K, T, r, sigma):.4f}")
    print(f"Delta:      {delta_call(S, K, T, r, sigma):.4f}")
    print(f"Gamma:      {gamma(S, K, T, r, sigma):.4f}")
    print(f"Vega:       {vega(S, K, T, r, sigma):.4f}")
    print(f"Theta:      {theta_call(S, K, T, r, sigma):.4f}")
    print()

    # ---- Plots ----
    plot_price_surface()
    plot_price_vs_spot()
    plot_greeks()
    plot_time_decay()