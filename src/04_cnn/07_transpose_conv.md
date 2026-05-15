# Exercise 07: Transposed Convolution (Upsampling)

Use `nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=2, stride=2)`
to double the spatial size of a `(1, 8, 4, 4)` feature map.

Print the output shape — expect `torch.Size([1, 4, 8, 8])`.
