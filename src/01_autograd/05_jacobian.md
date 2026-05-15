# Exercise 05: Jacobian with torch.autograd.functional

Define `f(x) = [x[0]**2, x[0]*x[1], x[1]**3]`.

Use `torch.autograd.functional.jacobian` to compute the Jacobian at
`x = [2.0, 3.0]`. The Jacobian should be a `(3, 2)` matrix.
