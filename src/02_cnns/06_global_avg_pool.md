# Exercise 06: Global Average Pooling

A common alternative to flattening a feature map is to average it down to one
number per channel.

Given a feature map of shape `(2, 16, 8, 8)`, use `nn.AdaptiveAvgPool2d(1)` to
turn it into `(2, 16, 1, 1)`, then squeeze the trailing dims to `(2, 16)`.
