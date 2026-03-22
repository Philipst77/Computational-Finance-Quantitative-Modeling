import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from geometric_bm import simulate_gbm


# ---- FIGURES DIRECTORY ----
BASE_DIR = Path(__file__).parent
FIG_DIR = BASE_DIR / "figures" / "monte_carlo"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# MONTE CARLO EXPECTATION
# ==============================================================================
# Monte Carlo answers: "What is the average outcome under uncertainty?"
#
# Steps:
#   1. Simulate many possible futures (paths)
#   2. Compute the quantity of interest on each path
#   3. Average across paths → Law of Large Numbers guarantees convergence
#
# Monte Carlo does NOT create new models — it EXTRACTS numerical expectations
# from whatever stochastic model you give it (GBM, GARCH, Heston, etc.)
#
# Key uses:
#   - Option pricing (average discounted payoff)
#   - Risk measurement (VaR, CVaR)
#   - Any expectation that has no closed-form solution
# ==============================================================================


def monte_carlo_expectation() -> None:
    """
    Demonstrate convergence of Monte Carlo estimate of E[S(T)].
    As n_paths increases, the estimate stabilizes — Law of Large Numbers.
    """
    S0, mu, sigma, T, N = 100.0, 0.05, 0.2, 1.0, 252
    theoretical = S0 * np.exp(mu * T)   # exact: E[S(T)] = S0 * e^(mu*T)

    path_counts = [50, 100, 500, 1_000, 5_000, 10_000]
    estimates = []

    for n_paths in path_counts:
        t, S = simulate_gbm(S0=S0, mu=mu, sigma=sigma, T=T, N=N,
                            n_paths=n_paths, seed=42)
        S_T = S[:, -1]
        estimates.append(S_T.mean())
        print(f"Paths: {n_paths:>6} | E[S_T] ≈ {S_T.mean():.4f}  (theoretical: {theoretical:.4f})")

    # ---- CONVERGENCE PLOT ----
    plt.figure(figsize=(9, 5))
    plt.plot(path_counts, estimates, marker="o", linewidth=1.5, label="MC Estimate")
    plt.axhline(theoretical, color="crimson", linestyle="--",
                linewidth=1.5, label=f"Theoretical E[S(T)] = {theoretical:.2f}")
    plt.xscale("log")
    plt.xlabel("Number of Monte Carlo Paths (log scale)")
    plt.ylabel("Estimated E[S_T]")
    plt.title("Monte Carlo Convergence: E[S(T)]")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "expectation_convergence.png", dpi=300)
    plt.show()


def terminal_distribution_demo() -> None:
    """
    Show the distribution of terminal prices from GBM simulation.
    Terminal prices follow a lognormal distribution.
    """
    from scipy.stats import lognorm

    S0, mu, sigma, T, N = 100.0, 0.05, 0.2, 1.0, 252
    n_paths = 20_000

    t, S = simulate_gbm(S0=S0, mu=mu, sigma=sigma, T=T, N=N,
                        n_paths=n_paths, seed=123)
    S_T = S[:, -1]

    # Theoretical lognormal parameters
    log_mean = np.log(S0) + (mu - 0.5 * sigma ** 2) * T
    log_std = sigma * np.sqrt(T)

    x = np.linspace(S_T.min(), S_T.max(), 300)
    theoretical_pdf = lognorm.pdf(x, s=log_std, scale=np.exp(log_mean))

    plt.figure(figsize=(9, 5))
    plt.hist(S_T, bins=80, density=True, alpha=0.6,
             color="steelblue", label="Simulated terminal prices")
    plt.plot(x, theoretical_pdf, color="crimson", linewidth=2,
             label="Theoretical lognormal")
    plt.axvline(S_T.mean(), color="black", linestyle="--",
                linewidth=1.2, label=f"MC Mean = {S_T.mean():.2f}")
    plt.xlabel("Terminal Price S(T)")
    plt.ylabel("Density")
    plt.title("Monte Carlo: Distribution of Terminal Prices (GBM)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "terminal_price_distribution.png", dpi=300)
    plt.show()

    print(f"MC   E[S_T] = {S_T.mean():.4f}  |  Theoretical = {S0 * np.exp(mu * T):.4f}")
    print(f"MC Std[S_T] = {S_T.std():.4f}")


def monte_carlo_option_pricing() -> None:
    """
    Price a European call option using Monte Carlo.

    The payoff of a call is: max(S(T) - K, 0)
    Monte Carlo price = e^(-r*T) * E[max(S(T) - K, 0)]

    We compare to the Black-Scholes closed-form price to validate.
    """
    from scipy.stats import norm

    # ---- Parameters ----
    S0, K, r, sigma, T, N = 100.0, 105.0, 0.05, 0.2, 1.0, 252
    n_paths = 50_000

    # ---- Monte Carlo ----
    t, S = simulate_gbm(S0=S0, mu=r, sigma=sigma, T=T, N=N,
                        n_paths=n_paths, seed=42)
    S_T = S[:, -1]
    payoffs = np.maximum(S_T - K, 0)
    mc_price = np.exp(-r * T) * payoffs.mean()

    # 95% confidence interval
    se = payoffs.std() / np.sqrt(n_paths)
    ci_lo = np.exp(-r * T) * (payoffs.mean() - 1.96 * se)
    ci_hi = np.exp(-r * T) * (payoffs.mean() + 1.96 * se)

    # ---- Black-Scholes closed form ----
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    bs_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    print(f"European Call Option  (S0={S0}, K={K}, r={r}, σ={sigma}, T={T})")
    print(f"  Monte Carlo price:   {mc_price:.4f}  [95% CI: {ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Black-Scholes price: {bs_price:.4f}")
    print(f"  Error: {abs(mc_price - bs_price):.4f}")

    # ---- Payoff distribution plot ----
    plt.figure(figsize=(9, 5))
    plt.hist(payoffs, bins=80, density=True, alpha=0.6, color="steelblue")
    plt.axvline(payoffs.mean(), color="crimson", linestyle="--",
                linewidth=1.5, label=f"E[Payoff] = {payoffs.mean():.2f}")
    plt.xlabel("Payoff max(S(T) - K, 0)")
    plt.ylabel("Density")
    plt.title(f"Monte Carlo: Call Option Payoff Distribution\nMC Price={mc_price:.3f} | BS Price={bs_price:.3f}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "mc_option_payoff.png", dpi=300)
    plt.show()


def monte_carlo_confidence_intervals() -> None:
    """
    Show how confidence intervals on E[S(T)] shrink as n_paths grows.
    CI width shrinks as 1/sqrt(n) — the standard Monte Carlo convergence rate.
    """
    S0, mu, sigma, T, N = 100.0, 0.05, 0.2, 1.0, 252
    theoretical = S0 * np.exp(mu * T)

    path_counts = [100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000]
    means, ci_los, ci_his = [], [], []

    for n_paths in path_counts:
        t, S = simulate_gbm(S0=S0, mu=mu, sigma=sigma, T=T, N=N,
                            n_paths=n_paths, seed=42)
        S_T = S[:, -1]
        mean = S_T.mean()
        se = S_T.std() / np.sqrt(n_paths)
        means.append(mean)
        ci_los.append(mean - 1.96 * se)
        ci_his.append(mean + 1.96 * se)

    means = np.array(means)
    ci_los = np.array(ci_los)
    ci_his = np.array(ci_his)

    plt.figure(figsize=(10, 5))
    plt.fill_between(path_counts, ci_los, ci_his, alpha=0.25,
                     color="steelblue", label="95% Confidence Interval")
    plt.plot(path_counts, means, marker="o", linewidth=1.5,
             color="steelblue", label="MC Estimate")
    plt.axhline(theoretical, color="crimson", linestyle="--",
                linewidth=1.5, label=f"Theoretical = {theoretical:.2f}")
    plt.xscale("log")
    plt.xlabel("Number of Paths (log scale)")
    plt.ylabel("E[S(T)]")
    plt.title("Monte Carlo: Confidence Intervals Shrink as 1/√n")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "mc_confidence_intervals.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # ---- 1: Convergence of E[S(T)] ----
    monte_carlo_expectation()

    # ---- 2: Terminal price distribution vs lognormal ----
    terminal_distribution_demo()

    # ---- 3: Option pricing via MC vs Black-Scholes ----
    monte_carlo_option_pricing()

    # ---- 4: Confidence intervals shrink as 1/sqrt(n) ----
    monte_carlo_confidence_intervals()