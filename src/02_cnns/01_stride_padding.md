# Exercise 01: Stride and Padding

For a `(1, 1, 7, 7)` input compute the output spatial size after:

- `Conv2d(1, 1, kernel_size=3, stride=1, padding=0)` → 5
- `Conv2d(1, 1, kernel_size=3, stride=2, padding=0)` → 3
- `Conv2d(1, 1, kernel_size=3, stride=1, padding=1)` → 7
- `Conv2d(1, 1, kernel_size=3, stride=2, padding=1)` → 4

Print the output shape from each layer to confirm.
