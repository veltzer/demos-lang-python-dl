# Exercise 07: Fine-Tune a Classifier Head on Frozen BERT

The pretrain-then-fine-tune pattern in its simplest form: freeze a big
pretrained backbone, train only a small classifier head on top.

## Tasks

- Load `distilbert-base-uncased` (smaller and faster than `bert-base-uncased`):
  - `AutoTokenizer.from_pretrained(...)`
  - `AutoModel.from_pretrained(...)`
- **Freeze** every parameter in the backbone: `for p in model.parameters():
  p.requires_grad = False`.
- Load 64 train examples from GLUE SST-2 (`load_dataset("glue", "sst2",
  split="train").select(range(64))`).
- Tokenize them with `padding="max_length"`, `max_length=32`, `truncation=True`
  and return PyTorch tensors.
- Build a tiny classifier head: `nn.Linear(hidden_size, 2)`. (DistilBERT's hidden
  size is 768.)
- Forward pass: get the `[CLS]` embedding (`last_hidden_state[:, 0]`), feed
  it to the head.
- Train for 5 epochs with `CrossEntropyLoss` and `Adam` on just the head's
  parameters. Use the full 64 examples as a single batch.
- Print the loss each epoch. It should drop noticeably (e.g. from ~0.7 to
  under 0.5).

## What to verify

- The backbone's parameters' gradients stay `None` (frozen).
- The head's `weight.grad` is not `None` after `backward()`.
- The training loss decreases.
