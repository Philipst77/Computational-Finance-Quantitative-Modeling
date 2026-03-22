import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ---- FIGURES DIRECTORY ----
BASE_DIR = Path(__file__).parent
FIG_DIR = BASE_DIR / "figures" / "volatility_models"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# HESTON STOCHASTIC VOLATILITY MODEL
# ==============================================================================
# In Black-Scholes, volatility is a fixed constant: sigma = constant
# In GARCH, volatility changes but is driven by past returns (discrete time)
#
# In the Heston model, volatility is ITSELF a random process (continuous time):
#
#   Asset price:   dS_t = mu * S_t * dt  +  sqrt(v_t) * S_t * dW_t^S
#   Variance:      dv_t = kappa * (theta - v_t) * dt  +  xi * sqrt(v_t) * dW_t^v
#
#   Correlation:   dW_t^S and dW_t^v are correlated with coefficient rho
#
# Parameters:
#   mu    — drift of the asset
#   v0    — initial variance (sigma0^2)
#   kappa — mean-reversion speed of variance (how fast vol reverts to theta)
#   theta — long-run mean variance (vol reverts toward sqrt(theta))
#   xi    — volatility of volatility ("vol of vol")
#   rho   — correlation between asset and variance shocks
#             rho < 0 → "leverage effect": price drops → vol spikes (realistic)
#
# Feller condition for variance to stay positive:  2 * kappa * theta > xi^2
# ==============================================================================


def simulate_heston(
    S0: float = 100.0,
    mu: float = 0.05,
    v0: float = 0.04,       # initial variance = 0.2^2
    kappa: float = 2.0,     # mean reversion speed
    theta: float = 0.04,    # long-run variance = 0.2^2
    xi: float = 0.3,        # vol of vol
    rho: float = -0.7,      # leverage effect (negative = realistic)
    T: float = 1.0,
    N: int = 252,
    n_paths: int = 1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate the Heston stochastic volatility model using Euler-Maruyama discretization.

    Parameters
    ----------
    S0      : initial asset price
    mu      : drift
    v0      : initial variance
    kappa   : mean-reversion speed of variance
    theta   : long-run mean variance
    xi      : volatility of volatility
    rho     : correlation between price and variance shocks
    T       : time horizon (years)
    N       : number of time steps
    n_paths : number of simulation paths
    seed    : random seed

    Returns
    -------
    t   : time grid, shape (N+1,)
    S   : asset price paths, shape (n_paths, N+1)
    v   : variance paths, shape (n_paths, N+1)
    """
    feller = 2 * kappa * theta
    if feller <= xi ** 2:
        print(f"Warning: Feller condition not satisfied (2κθ={feller:.4f} ≤ ξ²={xi**2:.4f})")
        print("Variance may hit zero. Consider increasing kappa or theta, or reducing xi.")

    rng = np.random.default_rng(seed)
    dt = T / N
    t = np.linspace(0, T, N + 1)

    S = np.zeros((n_paths, N + 1))
    v = np.zeros((n_paths, N + 1))

    S[:, 0] = S0
    v[:, 0] = v0

    # Cholesky decomposition to generate correlated Brownian motions
    # W^S = Z1
    # W^v = rho * Z1 + sqrt(1 - rho^2) * Z2
    for t_idx in range(N):
        Z1 = rng.standard_normal(n_paths)
        Z2 = rng.standard_normal(n_paths)

        dW_S = np.sqrt(dt) * Z1
        dW_v = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho ** 2) * Z2)

        # Variance step — clamp to 0 to avoid negative variance (full truncation)
        v_curr = np.maximum(v[:, t_idx], 0)

        v[:, t_idx + 1] = (
            v_curr
            + kappa * (theta - v_curr) * dt
            + xi * np.sqrt(v_curr) * dW_v
        )
        v[:, t_idx + 1] = np.maximum(v[:, t_idx + 1], 0)  # truncation scheme

        # Price step
        S[:, t_idx + 1] = S[:, t_idx] * np.exp(
            (mu - 0.5 * v_curr) * dt + np.sqrt(v_curr) * dW_S
        )

    print(f"Heston Model Parameters:")
    print(f"  kappa={kappa}, theta={theta}, xi={xi}, rho={rho}, v0={v0}")
    print(f"  Long-run vol = sqrt(theta) = {np.sqrt(theta):.4f}")
    print(f"  Feller condition (2κθ > ξ²): {feller:.4f} > {xi**2:.4f} → {'✓' if feller > xi**2 else '✗'}")

    return t, S, v


def plot_price_and_volatility(
    t: np.ndarray,
    S: np.ndarray,
    v: np.ndarray,
) -> None:
    """
    Plot a single Heston path: asset price on top, stochastic volatility below.
    Shows how volatility evolves randomly over time (unlike constant sigma in BS).
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(t, S[0], color="steelblue", linewidth=1.2)
    axes[0].set_ylabel("Asset Price S(t)")
    axes[0].set_title("Heston Model: Asset Price Path")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, np.sqrt(v[0]), color="crimson", linewidth=1.0)
    axes[1].axhline(np.sqrt(v[0, 0]), color="gray", linestyle="--", linewidth=0.8, label=f"Initial vol = {np.sqrt(v[0,0]):.2f}")
    axes[1].set_ylabel("Stochastic Volatility √v(t)")
    axes[1].set_xlabel("Time (years)")
    axes[1].set_title("Heston Model: Stochastic Volatility Path")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "heston_price_and_vol.png", dpi=300)
    plt.show()


def plot_multiple_vol_paths(
    t: np.ndarray,
    v: np.ndarray,
    n_display: int = 30,
) -> None:
    """
    Plot multiple variance paths to show the spread of possible volatility trajectories.
    This is the key difference from GARCH and Black-Scholes — volatility is a full
    random process with its own distribution.
    """
    plt.figure(figsize=(12, 5))

    for i in range(min(n_display, v.shape[0])):
        plt.plot(t, np.sqrt(v[i]), linewidth=0.6, alpha=0.5)

    # Long-run vol reference line
    theta = v.mean()  # approximate
    plt.axhline(np.sqrt(v[:, -1].mean()), color="black", linestyle="--",
                linewidth=1.2, label="Mean terminal vol")

    plt.xlabel("Time (years)")
    plt.ylabel("Volatility √v(t)")
    plt.title(f"Heston Model: {n_display} Stochastic Volatility Paths")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "heston_vol_paths.png", dpi=300)
    plt.show()


def compare_constant_vs_stochastic_vol(seed: int = 42) -> None:
    """
    Compare terminal price distributions under:
      1. Black-Scholes (constant volatility)
      2. Heston (stochastic volatility)

    Key insight: Heston produces fatter tails and a skewed distribution,
    which better matches real market behavior.
    """
    from scipy.stats import norm

    n_paths = 20_000
    S0, mu, T, N = 100.0, 0.05, 1.0, 252
    sigma_const = 0.20   # constant vol for BS
    v0 = sigma_const ** 2

    rng = np.random.default_rng(seed)

    # ---- Black-Scholes (constant vol GBM) ----
    dt = T / N
    Z = rng.standard_normal((n_paths, N))
    log_returns = (mu - 0.5 * sigma_const ** 2) * dt + sigma_const * np.sqrt(dt) * Z
    S_BS = S0 * np.exp(log_returns.sum(axis=1))

    # ---- Heston (stochastic vol) ----
    _, S_heston, _ = simulate_heston(
        S0=S0, mu=mu, v0=v0,
        kappa=2.0, theta=v0, xi=0.3, rho=-0.7,
        T=T, N=N, n_paths=n_paths, seed=seed,
    )
    S_H = S_heston[:, -1]

    # ---- Plot ----
    plt.figure(figsize=(10, 6))
    plt.hist(S_BS, bins=80, density=True, alpha=0.5, color="steelblue", label="Black-Scholes (constant σ)")
    plt.hist(S_H, bins=80, density=True, alpha=0.5, color="crimson", label="Heston (stochastic σ)")
    plt.xlabel("Terminal Price S(T)")
    plt.ylabel("Density")
    plt.title("Terminal Price Distribution: Black-Scholes vs Heston")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "bs_vs_heston_distribution.png", dpi=300)
    plt.show()

    print(f"\nBlack-Scholes: mean={S_BS.mean():.2f}, std={S_BS.std():.2f}")
    print(f"Heston:        mean={S_H.mean():.2f}, std={S_H.std():.2f}")


def plot_leverage_effect(seed: int = 42) -> None:
    """
    Demonstrate the leverage effect: rho < 0 means price drops are correlated
    with volatility spikes. This is a well-known empirical feature of equity markets.

    Compare rho = -0.7 (realistic) vs rho = 0.0 (no leverage effect).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, rho, label in zip(
        axes,
        [-0.7, 0.0],
        ["rho = -0.7 (Leverage Effect)", "rho = 0.0 (No Leverage)"],
    ):
        t, S, v = simulate_heston(
            rho=rho, n_paths=1, N=252, seed=seed
        )
        ax2 = ax.twinx()
        ax.plot(t, S[0], color="steelblue", linewidth=1.0, label="Price")
        ax2.plot(t, np.sqrt(v[0]), color="crimson", linewidth=0.9, alpha=0.8, label="Vol")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Price", color="steelblue")
        ax2.set_ylabel("Volatility", color="crimson")
        ax.set_title(label)
        ax.grid(alpha=0.2)

    plt.suptitle("Heston Leverage Effect: Price vs Volatility Correlation", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "heston_leverage_effect.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # ---- Simulate a single Heston path ----
    t, S, v = simulate_heston(
        S0=100.0,
        mu=0.05,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
        T=1.0,
        N=252,
        n_paths=1,
        seed=42,
    )

    # ---- Plot 1: Single price + vol path ----
    plot_price_and_volatility(t, S, v)

    # ---- Plot 2: Many vol paths ----
    t, S_many, v_many = simulate_heston(
        S0=100.0, mu=0.05, v0=0.04,
        kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        T=1.0, N=252, n_paths=100, seed=42,
    )
    plot_multiple_vol_paths(t, v_many, n_display=50)

    # ---- Plot 3: BS vs Heston terminal distribution ----
    compare_constant_vs_stochastic_vol()

    # ---- Plot 4: Leverage effect ----
    plot_leverage_effect()