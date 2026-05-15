# Exercise 02: A Tiny VAE

Build a VAE with an 8-D input, 4-D latent. The encoder outputs `mu` and
`logvar`; sample `z = mu + sigma * eps`; the decoder reconstructs the input.

Loss = MSE reconstruction + KL divergence to N(0, I).

Train on random data for 1000 steps. Print the final loss.
