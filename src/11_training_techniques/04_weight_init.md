# Exercise 04: Weight Initialization

Initialize a `Linear(100, 100)` layer's weights with each of:

- `nn.init.normal_(mean=0, std=0.01)`
- `nn.init.xavier_normal_`
- `nn.init.kaiming_normal_(nonlinearity='relu')`

For each, print the resulting weight tensor's std. The std reflects the
initializer's chosen scale.
