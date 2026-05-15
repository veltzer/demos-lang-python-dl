# Exercise 06: Label Smoothing

`CrossEntropyLoss(label_smoothing=0.1)` discourages the model from assigning
extreme confidence.

Build logits `[[5.0, 0.0, 0.0]]` and target `[0]`. Compute the loss with
`label_smoothing=0.0` and `label_smoothing=0.1`. The smoothed version should be
slightly larger because the loss now expects 0.9 mass on the correct class plus
0.05 mass on each other class.
