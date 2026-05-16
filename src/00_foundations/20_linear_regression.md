# Exercise 04: Linear Regression by Gradient Descent

Generate synthetic data: `x ~ Uniform(0, 10)` with `n=200` points, and
`y = 2 * x + 1 + noise` where noise is `N(0, 0.5)`.

Fit `y = w * x + b` with `SGD(lr=0.01)` for 1000 steps. Print the learned `w`
and `b` — they should approach `2` and `1`.
