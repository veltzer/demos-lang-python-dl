# Exercise 03: Tiny GAN

Build a 1-D toy GAN that learns to generate samples from N(5, 1):

- Generator: `Linear(1 -> 16) -> ReLU -> Linear(16 -> 1)`, input is N(0, 1) noise.
- Discriminator: `Linear(1 -> 16) -> ReLU -> Linear(16 -> 1)`, output is logit.

Train with the standard minimax: D maximizes `log D(x) + log(1 - D(G(z)))`;
G minimizes `-log D(G(z))`. After 2000 steps, print the mean and std of 1000
generated samples — they should approach `5.0` and `1.0`.
