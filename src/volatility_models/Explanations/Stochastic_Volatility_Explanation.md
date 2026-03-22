# Stochastic Volatility and the Heston Model

In Black-Scholes, volatility is a constant. In GARCH, volatility changes over time but is entirely determined by past returns. Stochastic volatility models take a fundamentally different approach: volatility is itself a random process, driven by its own source of randomness that is distinct from, but correlated with, the randomness driving the asset price.

This distinction matters because real markets show volatility behavior that cannot be explained by past returns alone. Implied volatility surfaces extracted from options markets show systematic patterns, such as the volatility smile and skew, that are inconsistent with any constant-volatility model. Stochastic volatility models are designed to capture these patterns from first principles.

## The Heston Model

The Heston model, introduced by Steven Heston in 1993, is the most widely used stochastic volatility model in practice. It specifies two coupled stochastic differential equations. The first governs the asset price and the second governs the variance.

The asset price follows:

    dS_t = mu * S_t * dt + sqrt(v_t) * S_t * dW_t^S

The variance follows a mean-reverting process known as the Cox-Ingersoll-Ross process:

    dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW_t^v

The two Brownian motions are correlated with coefficient rho, meaning dW_t^S and dW_t^v move together with correlation rho.

## Understanding the Parameters

Kappa is the mean-reversion speed. It controls how quickly variance returns to its long-run level after being displaced by a shock. A high kappa means volatility snaps back quickly. A low kappa means shocks persist for a long time.

Theta is the long-run mean variance. Over long time horizons, the variance process gravitates toward theta. The long-run volatility implied by the model is the square root of theta.

Xi is the volatility of volatility, often called vol-of-vol. It controls how much the variance process itself fluctuates. Higher xi produces more dramatic swings in volatility and fatter tails in the return distribution.

Rho is the correlation between price shocks and variance shocks. In equity markets, this parameter is typically negative, around negative 0.7, because falling prices tend to be accompanied by rising volatility. This is the leverage effect, a well-documented empirical feature of equity markets. A negative rho produces a left-skewed return distribution, which matches the negative skewness observed in equity index returns.

## The Feller Condition

For the variance process to remain strictly positive with probability one, the parameters must satisfy 2 * kappa * theta > xi squared. This is the Feller condition. When it is violated, the variance process can touch zero, which creates numerical difficulties in simulation. In practice, even when the Feller condition is not satisfied, a truncation scheme that clamps variance at zero is often used to keep the simulation well-behaved.

## What the Heston Model Captures

The most important consequence of stochastic volatility is that it generates a non-flat implied volatility surface. Under Black-Scholes, every option on the same underlying has the same implied volatility regardless of strike or maturity. Under Heston, different strikes and maturities produce different implied volatilities, forming the volatility smile and term structure observed in real options markets.

The negative rho generates a volatility skew, where out-of-the-money puts have higher implied volatility than out-of-the-money calls. This skew reflects the market's pricing of downside risk and is one of the most persistent features of equity options markets.

The vol-of-vol parameter xi governs the curvature of the smile. Higher xi produces a more pronounced smile shape, while lower xi produces a flatter surface closer to Black-Scholes.

## Comparison with GARCH

Both GARCH and Heston allow volatility to vary over time, but they differ in a fundamental way. GARCH operates in discrete time and makes volatility a deterministic function of past data. Heston operates in continuous time and treats volatility as a genuinely random process with its own independent source of uncertainty. This additional randomness is what allows Heston to generate the full range of volatility surface shapes observed in practice, which GARCH cannot reproduce.

The Heston model is the natural continuous-time upgrade from the GARCH framework and serves as the foundation for more advanced models used in derivatives trading and risk management.