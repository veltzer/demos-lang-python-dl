# Exercise 05: Train a CNN on Real MNIST

The previous exercise (`04_mnist_cnn`) built a CNN sized for MNIST but ran it
on random tensors. This one actually trains it on the real MNIST digits.

This is the first exercise in the repo that uses a real dataset.

## Tasks

- Download MNIST via `torchvision.datasets.MNIST(root="data", download=True,
  train=True/False, transform=torchvision.transforms.ToTensor())`.
- For speed, **subsample** the train set to 2000 examples and the test set
  to 1000 examples using `torch.utils.data.Subset(ds, indices)`. The full
  60000-train set works too but takes minutes.
- Build the same CNN as `04_mnist_cnn`:
  `Conv(1,8,3,pad=1) -> ReLU -> MaxPool -> Conv(8,16,3,pad=1) -> ReLU ->
   MaxPool -> Flatten -> Linear(16*7*7, 10)`.
- Train for 3 epochs with `CrossEntropyLoss` and `Adam(lr=1e-3)`, batch size
  64. Print epoch loss.
- Compute test-set accuracy after training. With 2000 train examples and 3
  epochs you should get **~80%** test accuracy. With the full 60k training
  set and more epochs you can push this past 99% — MNIST is easy once you
  give the model enough data.

## What to verify

- Training loss decreases each epoch.
- Final test accuracy is well above the random baseline (10% for 10 classes).
