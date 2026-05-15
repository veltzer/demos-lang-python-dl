# Exercise 03: Device Placement

The standard pattern is `device = "cuda" if torch.cuda.is_available() else "cpu"`.

Build a tiny model, move it to that device, generate an input on CPU, move
*it* to that device too, run the forward pass, and print the device of the output.
