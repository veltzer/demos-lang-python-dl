# Exercise 02: Gradient Clipping

Demonstrate `torch.nn.utils.clip_grad_norm_`.

Build any small model. Manually create a huge gradient on its first parameter
(set `.grad = ones * 1e6`). Then call `clip_grad_norm_(model.parameters(), max_norm=1.0)`.

Print the post-clip gradient norm — it should be `<= 1.0`.
