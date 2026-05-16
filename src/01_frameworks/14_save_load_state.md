# Exercise 04: Save and Load state_dict

Build a small model, save its `state_dict` to a temporary file with
`torch.save`, then create a fresh model and load the parameters with
`load_state_dict`.

Verify the second model produces identical output to the first on the same
input.
