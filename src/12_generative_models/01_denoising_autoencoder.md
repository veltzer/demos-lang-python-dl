# Exercise 01: Denoising Autoencoder

Same architecture as the previous exercise, but during training corrupt the
input with Gaussian noise `N(0, 0.3)` before passing it through. Compare the
reconstruction loss (against the *clean* target) against the version without
the noise. The denoising variant should still reconstruct cleanly.
