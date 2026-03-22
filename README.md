# Computational Finance & Quantitative Modeling

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NumPy](https://img.shields.io/badge/numpy-1.24+-013243.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/scipy-1.10+-8CAAE6.svg)](https://scipy.org/)

A structured implementation of core quantitative finance models and methodologies, progressing from foundational stochastic processes through option pricing, volatility modeling, and portfolio optimization. The repository emphasizes mathematical rigor, computational implementation, and empirical validation using real market data.

---

## Overview

This project builds the quantitative finance stack from first principles. Each module introduces a new layer of modeling complexity, and every concept is implemented from scratch, verified against theoretical properties, and validated on real data where applicable.

The architecture is designed as a learning framework and a demonstration of how mathematical models translate into working financial systems — from the Brownian motion that underlies all of modern derivatives theory, to a live backtesting engine that evaluates portfolio strategies on 14 years of real market data.

---

## Repository Structure

```
.
├── data/
│   └── market_data.py              # Real market data pipeline (yfinance)
├── evaluation/
│   ├── risk_metrics.py             # Sharpe, Sortino, Calmar, VaR, CVaR, drawdown
│   └── backtesting.py              # Rolling-window strategy backtest engine
├── notebooks/
│   └── experiments.ipynb           # Master end-to-end walkthrough notebook
└── src/
    ├── stochastic_processes/       # Brownian motion, GBM, Monte Carlo
    ├── option_pricing/             # Black-Scholes, Greeks, MC pricing
    ├── volatility_models/          # GARCH(1,1), Heston stochastic volatility
    └── portfolio_optimization/     # Mean-variance, risk parity, efficient frontier
```

---

## Modules

### Stochastic Processes

The mathematical foundation. All derivative pricing and portfolio simulation rests on stochastic processes.

**Brownian Motion** — Simulates standard Wiener processes and empirically verifies the core theoretical properties: Var[W(t)] = t, normally distributed increments, and continuous but nowhere differentiable paths. Includes 2D path plots, variance growth verification, increment distribution tests, and an interactive 3D HTML visualization.

**Geometric Brownian Motion** — Implements the exact GBM solution via Itô's lemma. Verifies that log returns are normally distributed with the correct theoretical mean and variance. Includes parameter sensitivity analysis for both drift μ and volatility σ, with simulated vs theoretical mean path overlays.

**Monte Carlo Expectation** — Demonstrates Monte Carlo as a numerical integration tool for stochastic models. Shows convergence of E[S(T)] to the theoretical value, confidence interval shrinkage as 1/√n, and prices a European call option numerically with BS validation.

---

### Option Pricing

Closed-form and numerical pricing of European options under the Black-Scholes framework.

**Black-Scholes** — Full implementation of the Black-Scholes pricing model including call and put prices, put-call parity verification, and all four Greeks. Includes a 3D call price surface, price vs spot plots with intrinsic value overlay, time decay visualization, and a complete Greeks dashboard.

**Monte Carlo Pricing** — Monte Carlo option pricing with confidence intervals. Validates against Black-Scholes closed-form across the full range of spot prices, demonstrates convergence as path count increases, and shows the discounted payoff distribution.

**Greeks Notebook** — Interactive exploration of Delta, Gamma, Vega, and Theta. Each Greek is shown as both a 2D curve (for intuition) and an interactive 3D Plotly surface. Includes parameter sensitivity analysis — Delta across different volatilities, Gamma and Theta across different maturities.

| Greek | Measures | Key Behavior |
|---|---|---|
| Delta (Δ) | Sensitivity to S | Ranges 0→1 for calls; hedge ratio |
| Gamma (Γ) | Rate of change of Δ | Peaks ATM; spikes near expiry |
| Vega (ν) | Sensitivity to σ | Always positive; peaks ATM |
| Theta (Θ) | Time decay | Always negative; accelerates near expiry |

---

### Volatility Models

Moving beyond constant volatility. Three frameworks with increasing modeling sophistication.

**GARCH(1,1)** — Models volatility clustering: the empirically documented tendency for high-volatility periods to persist and low-volatility periods to persist. Variance evolves as a weighted combination of a long-run baseline, past shocks, and past variance. Includes persistence analysis, fat tail verification, and comparison of low vs high persistence regimes.

```
sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
```

**Heston Stochastic Volatility** — The industry-standard continuous-time stochastic volatility model. Volatility is itself a mean-reverting random process correlated with the asset price. The leverage effect (rho < 0) produces the negative skew observed in real equity markets. Includes multi-path volatility simulation, leverage effect demonstration, and terminal distribution comparison.

```
dS_t = mu * S_t * dt + sqrt(v_t) * S_t * dW_t^S
dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW_t^v
```

**Diagnostics Notebook** — Side-by-side comparison of all three volatility regimes (constant, GARCH, Heston) using 20,000 simulated paths. Compares terminal distributions, skewness, kurtosis, and Value at Risk at 1%, 5%, and 10% confidence levels.

---

### Portfolio Optimization

Translating risk and return estimates into optimal capital allocation.

**Mean-Variance Optimization** — Full Markowitz framework implementation. Computes the efficient frontier by sweeping target returns and solving the minimum variance problem at each point. Identifies the global minimum variance portfolio and the maximum Sharpe ratio (tangency) portfolio. Includes a dark-themed efficient frontier with 5,000 randomly simulated portfolios colored by Sharpe ratio, and a portfolio composition stackplot showing how asset weights shift along the frontier.

**Risk Parity** — Equal Risk Contribution portfolio that allocates capital so every asset contributes equally to total portfolio risk. Requires only the covariance matrix — no return estimates needed. Includes naive risk parity (1/vol weighting), true ERC via optimization, and a four-strategy comparison with diversification ratios.

**Efficient Frontier Notebook** — Interactive exploration of the full portfolio optimization problem using real ETF data. Compares equal weight, minimum variance, maximum Sharpe, and ERC portfolios across return, volatility, Sharpe, Sortino, and diversification ratio.

---

### Data

**Market Data Pipeline** — Downloads and caches real price data for 6 ETFs (SPY, EFA, AGG, VNQ, GSG, SHV) via yfinance. Computes log returns, annualized statistics, covariance matrices, and expected returns at daily, weekly, or monthly frequency. Includes normalized price history, return distributions with normal fit overlay, correlation heatmap, and rolling 63-day volatility.

| Ticker | Asset Class |
|---|---|
| SPY | US Large Cap Equity |
| EFA | International Developed Equity |
| AGG | US Aggregate Bonds |
| VNQ | US Real Estate (REITs) |
| GSG | Commodities |
| SHV | Short-Term Treasuries |

---

### Evaluation

**Risk Metrics** — Comprehensive risk measurement toolkit operating on any return series.

| Metric | Description |
|---|---|
| Sharpe Ratio | Excess return per unit of total risk |
| Sortino Ratio | Excess return per unit of downside risk only |
| Calmar Ratio | Annualized return divided by max drawdown |
| Max Drawdown | Worst peak-to-trough loss |
| VaR (Historical) | Loss threshold at given confidence level |
| CVaR / ES | Expected loss given that VaR is breached |

**Backtesting** — Rolling-window backtest engine that evaluates four portfolio strategies on 14 years of real market data (2010–2024). At each monthly rebalance, parameters are re-estimated using a 252-day lookback window. Transaction costs are applied based on portfolio turnover. Output includes equity curves, drawdown series, rolling Sharpe ratios, and a full risk report for every strategy.

---

## Master Notebook

`notebooks/experiments.ipynb` is an end-to-end walkthrough that connects every module into a single pipeline:

1. Load real market data (SPY, EFA, AGG, VNQ, GSG, SHV — 2010 to 2024)
2. Simulate and verify stochastic processes
3. Price options with Black-Scholes and Monte Carlo
4. Compare volatility models (constant vs GARCH vs Heston)
5. Build efficient frontier and optimal portfolios from real data
6. Run rolling backtest across all four strategies
7. Generate final risk report with full metrics comparison

---

## Design Principles

- **Mathematical foundations first** — every model starts with the equation, not the implementation
- **Empirical verification** — theoretical properties are tested against simulation (Var[W(t)]=t, log-normality of GBM, put-call parity, ERC convergence)
- **Real data validation** — portfolio optimization and backtesting use 14 years of live ETF data
- **Modular architecture** — each module is self-contained with its own figures, explanations, and `__init__.py`
- **Progressive complexity** — the stack builds deliberately: noise → prices → derivatives → volatility → portfolios → risk

---

## Getting Started

```bash
git clone https://github.com/Philipst77/Computational-Finance-Quantitative-Modeling.git
cd Computational-Finance-Quantitative-Modeling
pip install -r requirements.txt
```

### Run a module

```bash
python src/stochastic_processes/brownian_motion.py
python src/option_pricing/black_scholes.py
python src/volatility_models/garch.py
python src/portfolio_optimization/mean_variance.py
python evaluation/risk_metrics.py
python evaluation/backtesting.py
```

### Open the master notebook

```bash
jupyter notebook notebooks/experiments.ipynb
```

---

## Dependencies

```
numpy
scipy
pandas
matplotlib
plotly
yfinance
jupyter
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.