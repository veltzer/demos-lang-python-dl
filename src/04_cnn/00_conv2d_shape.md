# Exercise 00: Conv2d Shape

Create an `nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)`.

Push a `(2, 3, 32, 32)` input (batch=2, RGB 32x32 images) through it and print
the output shape. With `padding=1` the spatial dimensions stay the same, so you
should get `torch.Size([2, 8, 32, 32])`.
