# Exercise 00: Dropout

Build a model with dropout between two Linear layers:

```
Linear(10, 64) -> ReLU -> Dropout(p=0.3) -> Linear(64, 2)
```

In training mode, run the same input twice and verify the outputs differ. In
eval mode, verify they're identical.
