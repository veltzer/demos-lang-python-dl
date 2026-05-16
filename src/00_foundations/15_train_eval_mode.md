# Exercise 07: train() vs eval() Mode

Some layers (Dropout, BatchNorm) behave differently in training and evaluation.
Build a model with `nn.Dropout(p=0.5)` and run the same input twice in `train()`
mode — you should get *different* outputs. Then switch to `eval()` and verify
that two runs produce *identical* outputs.
