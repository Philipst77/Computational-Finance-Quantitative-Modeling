# Mean-Variance Optimization

Modern portfolio theory, introduced by Harry Markowitz in 1952, reframes portfolio construction as a mathematical optimization problem. Rather than selecting assets based on individual merit, the theory asks how assets should be combined to produce the best possible tradeoff between expected return and risk. The central insight is that the risk of a portfolio is not simply the average of its components — correlations between assets determine how much diversification benefit the combination provides.

## The Portfolio Problem

A portfolio is defined by a vector of weights w, where each weight represents the fraction of capital allocated to a given asset. The weights must sum to one. The expected return of the portfolio is the weighted average of individual asset returns. The variance of the portfolio depends not only on individual asset variances but also on all pairwise covariances between assets.

    Portfolio return:    mu_p   = w^T * mu
    Portfolio variance: sigma_p^2 = w^T * Sigma * w

The covariance matrix Sigma captures both individual volatilities and the correlations between assets. When assets are negatively correlated, combining them reduces portfolio variance below the weighted average of individual variances. This reduction is the mathematical expression of diversification.

## The Efficient Frontier

For any given level of expected return, there exists a portfolio that achieves that return with minimum variance. The set of all such minimum-variance portfolios traces out a curve in return-volatility space called the minimum variance frontier. The upper portion of this curve, where higher return comes with higher risk, is the efficient frontier.

Every portfolio on the efficient frontier is optimal in the sense that no other portfolio can offer higher return for the same risk, or lower risk for the same return. A rational investor will always choose a portfolio on the efficient frontier. The specific choice depends on their individual risk tolerance.

## Key Portfolios

The minimum variance portfolio is the leftmost point on the efficient frontier. It achieves the lowest possible variance across all feasible weight combinations. It does not maximize return, but it minimizes risk absolutely. This portfolio is useful as a benchmark and as a conservative allocation for highly risk-averse investors.

The maximum Sharpe ratio portfolio, also called the tangency portfolio, is the point on the efficient frontier where the line from the risk-free rate is tangent to the curve. It maximizes the ratio of excess return to volatility. When an investor can combine a risky portfolio with a risk-free asset, the tangency portfolio is the optimal risky portfolio for all investors regardless of risk tolerance. Every investor should hold the same risky portfolio and adjust their overall risk by varying the allocation between the risky portfolio and the risk-free asset.

## Limitations of Mean-Variance Optimization

The framework is theoretically elegant but practically difficult. The optimization is highly sensitive to return estimates. Small errors in expected returns lead to large changes in the optimal portfolio weights. In practice, return estimates derived from historical data are noisy and unstable, which causes the optimizer to produce extreme and concentrated allocations that are not robust out of sample.

This sensitivity is sometimes called the error maximization problem: the optimizer puts the most weight on assets with the highest estimated returns, but those are precisely the estimates most likely to be wrong. The result is portfolios that look optimal in-sample but perform poorly out-of-sample.

Several approaches address this problem. Shrinkage estimators regularize the return and covariance inputs toward a prior. Black-Litterman combines market equilibrium returns with investor views to produce more stable estimates. Risk-based approaches like risk parity abandon return estimates entirely and focus solely on the covariance structure.

## Portfolio Composition Along the Frontier

As the target return increases from the minimum variance portfolio toward the maximum return portfolio, the weight allocation shifts progressively toward higher-returning assets. At low target returns, defensive assets like bonds and low-volatility equities dominate. At high target returns, high-growth assets take the largest allocations. This composition shift is sometimes called the efficient frontier path and reveals which assets are driving risk and return at each point on the curve.