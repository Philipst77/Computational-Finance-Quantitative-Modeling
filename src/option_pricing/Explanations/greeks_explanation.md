# Option Greeks — Complete Intuition Guide

Options are not just price instruments — they are sensitivity instruments. The Greeks measure how the option price reacts to changes in stock price, volatility, and time.

## Delta

Delta measures how much the option price changes when the stock price changes. For calls, Delta ranges from zero to one, while for puts it ranges from negative one to zero.

When an option is deep out-of-the-money, Delta is close to zero. Small stock moves barely change the option value because the option is unlikely to finish in-the-money. When an option is at-the-money, Delta is around 0.5, representing about a 50/50 chance of finishing in-the-money. Small stock moves cause large value changes here, and this is where the surface is steepest. When an option is deep in-the-money, Delta approaches one. The option behaves almost like the stock itself, and further stock moves don't change the probability much.

Delta represents the slope of the option price with respect to stock price. The option is most sensitive to stock movements near the strike price.

## Gamma

Gamma measures how Delta changes with respect to stock price. More precisely, Gamma is the second derivative of option value with respect to stock price. It measures the curvature of the option price, where Delta is the slope and Gamma is how fast that slope changes.

Near at-the-money, Gamma is highest. Delta changes rapidly from around zero to around one, and small stock moves cause large Delta changes. This is the transition zone. When options are deep out-of-the-money or deep in-the-money, Gamma is small. Delta is already near zero or one, so stock moves don't change Delta much.

Volatility has an important effect on Gamma. Low volatility creates a sharp Delta transition and high Gamma, while high volatility creates a smoother transition and lower Gamma. This happens because volatility controls the width of the probability distribution. Low volatility means a narrow distribution, which creates a sharp switch between out-of-the-money and in-the-money, resulting in large curvature. High volatility means a wide distribution, which creates a gradual switch and smaller curvature. Gamma is fundamentally about the sharpness of probability switching near the strike.

## Vega

Vega measures how much the option price changes when volatility changes. The derivative of option value with respect to volatility is always positive, meaning options benefit from higher volatility.

This happens because volatility represents uncertainty, and uncertainty means a wider range of possible outcomes. Options are convex instruments with limited downside since you can only lose the premium, but unlimited upside for calls. A wider distribution increases the expected payoff.

It's important to clarify that volatility does not determine stock direction. High volatility does not mean prices fall, and low volatility does not mean prices rise. Volatility affects dispersion, not direction. Higher volatility means bigger possible up moves and bigger possible down moves, which makes options more expensive.

Vega is highest near at-the-money because that's where the outcome is most uncertain. Small volatility changes significantly affect the probability of finishing in-the-money in this region.

## Theta

Theta measures how option value changes as time passes. For long options, Theta is usually negative, which is called time decay.

Time decay happens because options have both intrinsic value and time value. Time value exists because of uncertainty. As expiration approaches, there is less time and therefore less uncertainty. Less uncertainty means lower time value, so the option value decays toward its intrinsic value.

Theta is most negative near at-the-money because time value is largest there and uncertainty disappears fastest in that region.

## The Big Picture

Delta measures price sensitivity. Gamma measures how Delta changes with respect to stock price, representing curvature. Vega measures volatility sensitivity. Theta measures time decay.

Near the strike price, Delta changes fastest, Gamma is highest, Vega is highest, and Theta decay is strongest. This is the most sensitive region of the option, where small changes in market conditions have the greatest impact on option value.