# Exercise 05: Canonical Train Loop

Put it all together. Generate `(N=500, 4)` features and class labels (3 classes)
where the label is `argmax(features @ random_matrix)`.

Build an MLP `4 -> 32 -> 3` and train it with `CrossEntropyLoss` and Adam,
inside a `DataLoader(batch_size=32)` loop, for 10 epochs.

Print epoch number, training loss, and training accuracy after each epoch.
