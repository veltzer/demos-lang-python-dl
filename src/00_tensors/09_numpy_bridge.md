# Exercise 09: NumPy Bridge

Build a NumPy array of `[1.0, 2.0, 3.0]`. Convert it to a PyTorch tensor with
`torch.from_numpy`. Modify the tensor in place by multiplying by 2 and show that
the underlying NumPy array changes too — they share memory.

Then convert back with `.numpy()` and verify the result.
