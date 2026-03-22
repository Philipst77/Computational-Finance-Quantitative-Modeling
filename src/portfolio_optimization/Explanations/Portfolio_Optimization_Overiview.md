# Portfolio Optimization — Overview

Portfolio optimization is the process of selecting asset weights to achieve a desired investment objective. The objective can be stated in many ways — maximize return, minimize risk, maximize risk-adjusted return, or equalize risk contributions — but all approaches share a common starting point: a set of assets with estimated return and risk characteristics, and a set of constraints on how capital can be allocated.

## Why Optimization Matters

Selecting individual assets well is not sufficient for good portfolio construction. Two portfolios can hold identical assets but achieve very different outcomes depending on how capital is distributed. A portfolio concentrated in a single high-return asset will have very different risk characteristics than one that spreads capital across many assets with carefully considered correlations. The allocation decision is as important as the asset selection decision, and in many cases it matters more.

## The Role of Correlation

The most important input in portfolio optimization is not expected return or individual volatility — it is the correlation structure between assets. Correlation determines how much diversification benefit a combination of assets provides. When assets move independently, combining them reduces portfolio volatility significantly below the weighted average of individual volatilities. When assets are highly correlated, combining them provides little diversification benefit regardless of how many are included.

This is why two portfolios with the same expected return can have dramatically different volatilities. One portfolio might achieve its return with assets that partially offset each other's movements, while another might hold assets that all rise and fall together. Understanding and exploiting correlation is the central challenge in portfolio construction.

## Mean-Variance vs Risk Parity

The two main approaches in this module represent fundamentally different philosophies. Mean-variance optimization, developed by Markowitz, asks which combination of assets maximizes the Sharpe ratio given estimates of expected returns and covariances. It treats portfolio construction as a return-seeking problem subject to a risk constraint. The result is theoretically optimal when inputs are correct, but it is fragile when return estimates are imprecise.

Risk parity treats portfolio construction as a risk-balancing problem. It ignores expected returns entirely and focuses on ensuring that no single asset dominates the portfolio's risk profile. The result is a portfolio that is genuinely diversified in a risk sense, and whose quality does not depend on the accuracy of return forecasts. This makes it more robust in practice, at the cost of potentially leaving return on the table when return estimates are reliable.

## Practical Considerations

In real applications, both approaches require careful attention to input estimation. Covariance matrices estimated from historical data are noisy and can be poorly conditioned when the number of assets is large relative to the number of observations. Shrinkage methods, factor models, and regularization techniques are commonly used to produce more stable covariance estimates.

Return estimates are even more difficult. Sample mean returns from historical data are extremely noisy and have little predictive power over investment horizons of interest. The Black-Litterman model, factor-based return forecasts, and analyst estimates are all used in practice to construct more reliable inputs for mean-variance optimization.

The evaluation of portfolio optimization results should always include out-of-sample backtesting, stress testing against historical crisis periods, and analysis of turnover and transaction costs. A portfolio that looks optimal in-sample may perform poorly out-of-sample if the optimization has overfit to historical data. This is why the backtesting and risk metrics modules that follow this section are essential complements to the optimization framework.