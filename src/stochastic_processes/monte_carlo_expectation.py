import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from geometric_bm import simulate_gbm


# ---- FIGURES DIRECTORY ----
BASE_DIR = Path(__file__).parent
FIG_DIR = BASE_DIR / "figures" / "monte_carlo"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def monte_carlo_expectation():
    # ---- MODEL PARAMETERS ----
    S0 = 100.0
    mu = 0.05
    sigma = 0.2
    T = 1.0
    N = 252

    # Different numbers of paths to show convergence
    path_counts = [50, 100, 500, 1_000, 5_000, 10_000]
    estimates = []

    for n_paths in path_counts:
        t, S = simulate_gbm(
            S0=S0,
            mu=mu,
            sigma=sigma,
            T=T,
            N=N,
            n_paths=n_paths,
            seed=42,
        )


        # Quantity of interest: terminal price S_T
        S_T = S[:, -1]

        # Monte Carlo expectation
        estimates.append(S_T.mean())

        print(f"Paths: {n_paths:>6} | E[S_T] ≈ {S_T.mean():.4f}")

    # ---- CONVERGENCE PLOT ----
    plt.figure(figsize=(8, 5))
    plt.plot(path_counts, estimates, marker="o")
    plt.xscale("log")
    plt.xlabel("Number of Monte Carlo Paths (log scale)")
    plt.ylabel("Estimated E[S_T]")
    plt.title("Monte Carlo Expectation Convergence")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "expectation_convergence.png", dpi=300)
    plt.show()


def terminal_distribution_demo():
    # ---- MODEL PARAMETERS ----
    S0 = 100.0
    mu = 0.05
    sigma = 0.2
    T = 1.0
    N = 252
    n_paths = 20_000

    t, S, _ = simulate_gbm(
        S0=S0,
        mu=mu,
        sigma=sigma,
        T=T,
        N=N,
        n_paths=n_paths,
        seed=123,
    )

    S_T = S[:, -1]

    # ---- HISTOGRAM ----
    plt.figure(figsize=(8, 5))
    plt.hist(S_T, bins=60, density=True, alpha=0.7)
    plt.xlabel("Terminal Price S(T)")
    plt.ylabel("Density")
    plt.title("Monte Carlo Distribution of Terminal Prices")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "terminal_price_distribution.png", dpi=300)
    plt.show()

    print("E[S_T] ≈", S_T.mean())
    print("Std[S_T] ≈", S_T.std())


if __name__ == "__main__":
    monte_carlo_expectation()
    terminal_distribution_demo()
