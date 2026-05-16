# Exercise 05: 1-D Denoising Diffusion (Sketch)

Implement the simplest possible DDPM-style denoising model on 1-D data
sampled from N(5, 1):

- Forward process: `x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * eps` where
  `alpha_t` decreases linearly from ~1 to ~0 over `T = 20` steps.
- Train a tiny MLP `(x_t, t) -> eps_hat` to predict the noise.
- Loss is MSE between predicted and true noise.

Train 2000 steps and print the loss.
