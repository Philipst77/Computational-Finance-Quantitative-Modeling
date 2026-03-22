# Volatility Models — Comparing Constant, GARCH, and Stochastic Volatility

The central question in options pricing and risk management is not whether volatility exists, but how it behaves over time. Three fundamentally different answers to this question correspond to three distinct modeling frameworks, each with different assumptions, different outputs, and different use cases.

## The Constant Volatility World

Black-Scholes assumes that volatility is a fixed constant over the life of the option. Under this assumption, the distribution of log returns over any horizon is exactly normal, terminal asset prices are exactly lognormal, and the implied volatility of every option on the same underlying is identical regardless of strike or maturity.

This framework is mathematically elegant and analytically tractable. It produces closed-form pricing formulas for European options and clean expressions for the Greeks. Its weakness is that it is empirically wrong. Real return distributions have fat tails and negative skew, and real implied volatility surfaces are not flat. Black-Scholes remains useful as a benchmark and a quoting convention, but it is not a realistic model of how volatility actually behaves.

## The GARCH World

GARCH extends Black-Scholes by making volatility time-varying in a specific structured way. Volatility at each point in time depends on past squared returns and past volatility values. This produces volatility clustering, where high-volatility periods tend to persist and low-volatility periods tend to persist, matching one of the most well-documented features of financial return data.

GARCH operates in discrete time and generates return distributions with excess kurtosis, meaning fatter tails than a normal distribution. It improves on Black-Scholes for risk measurement and volatility forecasting. Its limitation is that volatility remains a deterministic function of past data once you condition on the observed history. There is no additional uncertainty about today's volatility beyond what the past returns imply.

## The Stochastic Volatility World

Heston and related stochastic volatility models treat volatility as a genuinely random process driven by its own Brownian motion. This additional source of randomness produces two effects that neither Black-Scholes nor GARCH can replicate. First, it generates a non-flat implied volatility surface with smile and skew patterns that match observed options markets. Second, it produces return distributions with both fat tails and negative skew simultaneously, because the negative correlation between price and variance shocks creates asymmetry in the distribution.

Stochastic volatility models are the standard framework for derivatives pricing desks and are the natural choice whenever the goal is to match or reproduce the implied volatility surface from market prices.

## How to Read the Diagnostics

When comparing the three models side by side, the key quantities to examine are the shape of the terminal price distribution, the skewness and excess kurtosis of log returns, and the tail risk as measured by Value at Risk at various confidence levels.

Black-Scholes will produce the most symmetric and thin-tailed distribution. GARCH will produce fatter tails due to volatility clustering, but roughly symmetric returns. Heston will produce the fattest tails and the most pronounced negative skew, driven by the leverage effect from negative rho.

For risk management purposes, using a Black-Scholes model will systematically underestimate tail losses. A model that ignores volatility clustering or stochasticity will produce VaR estimates that are too optimistic, leading to undercapitalized portfolios and underpriced hedges. The diagnostics notebook makes this underestimation visible by directly comparing the tail behavior across all three frameworks.

## Choosing the Right Model

No model is universally correct. Black-Scholes is appropriate for quick back-of-the-envelope estimates and as a common language for quoting option prices through implied volatility. GARCH is appropriate for volatility forecasting, risk measurement on equity portfolios, and any application where discrete-time dynamics are natural. Heston is appropriate for pricing derivatives, building volatility surfaces, and any application where matching the market's implied distribution is the goal.

Understanding the differences between these frameworks is not just an academic exercise. It determines how much capital a desk sets aside for risk, how expensive a hedge is, and whether a pricing model produces arbitrage opportunities or leaves money on the table.