# GARCH and Volatility Clustering

Financial markets exhibit a well-documented phenomenon called volatility clustering: large price moves tend to be followed by more large moves, and calm periods tend to persist. A quiet day is more likely to be followed by another quiet day, and a turbulent day is more likely to be followed by more turbulence. This is one of the most robust empirical features of financial return data, and the constant-volatility assumption of Black-Scholes entirely ignores it.

GARCH, which stands for Generalized Autoregressive Conditional Heteroskedasticity, is a model that captures this behavior by making volatility a function of its own past values and past shocks. Rather than treating sigma as a fixed constant, GARCH treats it as a quantity that evolves through time according to its own equation.

## The GARCH(1,1) Model

The GARCH(1,1) model is the most widely used specification. It defines variance at each time step as a weighted combination of three components. The first is a long-run baseline variance, omega, which represents the unconditional level that variance gravitates toward over time. The second is the squared return from the previous period, which captures the immediate impact of a shock. The third is the variance from the previous period, which allows the current variance to depend on the recent history of volatility.

The variance equation is written as:

    sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

Here alpha controls how strongly a single shock impacts current variance, and beta controls how persistent that impact is over time. When alpha + beta is close to one, shocks to volatility die out slowly, meaning a spike in volatility today will still be felt many periods from now. This is exactly the volatility clustering behavior observed in equity markets.

## Stationarity and the Long-Run Variance

For the model to be stationary, meaning that variance does not explode to infinity over time, the condition alpha + beta < 1 must hold. When this holds, there exists a long-run unconditional variance given by omega divided by one minus alpha minus beta. This is the level that volatility reverts to in the absence of new shocks.

The quantity alpha + beta is called persistence. Real equity markets typically exhibit persistence values of 0.95 or higher, meaning volatility shocks are very slow to decay. This is why a single bad day in markets can keep implied volatility elevated for weeks.

## What GARCH Captures That Black-Scholes Cannot

Under Black-Scholes, the distribution of returns over any horizon is normal with a fixed standard deviation. GARCH returns are not normally distributed. Because volatility itself fluctuates, the marginal distribution of returns has fatter tails than a normal distribution and exhibits excess kurtosis. Large returns occur more frequently than a Gaussian model would predict, which is precisely what is observed in real financial data.

GARCH also produces a time-varying risk environment. When markets are calm, GARCH estimates low volatility and options priced under GARCH are cheap. When markets are turbulent, GARCH estimates high volatility and options become expensive. This dynamic pricing behavior is more realistic than the fixed-sigma world of Black-Scholes.

## Limitations of GARCH

GARCH models volatility as a deterministic function of past data. Once you observe the past returns and variances, the current volatility is fully determined. There is no additional randomness in the volatility itself. This is a meaningful constraint, because in reality markets seem to have volatility that fluctuates in ways not fully explained by past returns alone. This limitation motivates the move to stochastic volatility models, where volatility is itself a random process with its own source of uncertainty.