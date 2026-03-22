# Risk Parity and Equal Risk Contribution

Risk parity is a portfolio construction philosophy that allocates capital based on risk rather than on expected return. Instead of asking which assets offer the best return per unit of risk, it asks how to distribute risk equally across all assets in the portfolio. The motivation is straightforward: mean-variance optimization requires reliable estimates of expected returns, and those estimates are notoriously difficult to produce. Risk parity sidesteps this requirement entirely by focusing only on the covariance structure of returns.

## The Risk Contribution Framework

In any portfolio, each asset contributes a specific amount to total portfolio risk. The risk contribution of asset i depends both on its individual volatility and on how it correlates with the rest of the portfolio. An asset with high volatility but low correlation to other holdings may contribute less to total risk than an asset with moderate volatility that moves in lockstep with the rest of the portfolio.

The marginal risk contribution of asset i is the rate at which portfolio volatility increases as the weight of asset i increases. The total risk contribution of asset i is the marginal risk contribution multiplied by the weight. A fundamental identity in portfolio theory states that the sum of all risk contributions equals total portfolio volatility.

    RC_i = w_i * (Sigma * w)_i / sqrt(w^T * Sigma * w)

The percentage risk contribution of each asset is its absolute risk contribution divided by total portfolio volatility. These percentage contributions sum to one, and they tell you how much of total portfolio risk is attributable to each position.

## Equal Risk Contribution

In an equal risk contribution portfolio, the target is for every asset to contribute the same fraction of total portfolio risk. For a portfolio of n assets, each asset should contribute exactly 1/n of total risk. The ERC portfolio is found by solving an optimization problem that minimizes the sum of squared differences between actual and target risk contributions.

This approach has several attractive properties. It produces portfolios that are genuinely diversified in a risk sense, not just a weight sense. An equal-weight portfolio may be highly concentrated in risk if some assets are much more volatile than others. The ERC portfolio corrects for this by assigning lower weights to higher-volatility assets and higher weights to lower-volatility, well-diversifying assets.

## Naive Risk Parity

A common approximation is to weight each asset inversely proportional to its individual volatility. This is sometimes called naive risk parity or the one-over-volatility weighting scheme. It is fast to compute and requires only individual asset volatilities rather than the full covariance matrix. However, it does not account for correlations between assets and therefore does not achieve true equal risk contribution when assets are correlated. It is a useful baseline but the true ERC solution will generally produce better-balanced risk contributions.

## Comparison with Mean-Variance

Mean-variance optimization and risk parity represent two poles of a spectrum in portfolio construction. Mean-variance requires both return estimates and a covariance matrix, uses all available information, and produces the highest Sharpe ratio when inputs are correct. Risk parity requires only a covariance matrix, ignores return information entirely, and produces a portfolio whose quality does not depend on return forecasts.

In practice, risk parity portfolios tend to be more stable over time because covariance estimates are more reliable than return estimates. Portfolios constructed with mean-variance optimization often require significant rebalancing as return estimates change, while risk parity portfolios change more slowly as the covariance structure evolves.

## The Diversification Ratio

One way to measure how much a portfolio benefits from diversification is the diversification ratio, defined as the weighted average of individual asset volatilities divided by portfolio volatility. A diversification ratio of one means the portfolio achieves no diversification benefit. A higher ratio means the portfolio's volatility is meaningfully lower than a naive combination of its components would suggest, indicating that correlations are providing genuine risk reduction. Risk parity portfolios tend to have high diversification ratios because they explicitly target balanced risk contributions, which naturally exploits low and negative correlations between assets.