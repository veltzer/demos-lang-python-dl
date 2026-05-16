# Anatomy of a Training Loop

Almost every supervised-learning exercise in this repo has the same shape.
Once you see the pattern, the variation between exercises is just which model
you build, which loss you pick, and which optimizer you use.

## The five lines that matter

```python
for xb, yb in dataloader:        # 1. get a batch
    pred = model(xb)             # 2. forward
    loss = loss_fn(pred, yb)     # 3. compute loss
    optimizer.zero_grad()        # 4. clear stale gradients
    loss.backward()              # 5. compute new gradients
    optimizer.step()             # 6. apply update
```

That's it. The rest is bookkeeping: logging, validation, checkpointing,
moving things to GPU, scheduling the learning rate. The core six lines
don't change.

## Why `zero_grad()` before `backward()`?

PyTorch *accumulates* gradients into `.grad`. If you call `backward()` twice
without clearing in between, the second gradient is added to the first.
This is occasionally what you want (gradient accumulation across micro-batches
when one large batch won't fit in memory). Far more often it's a bug — so the
canonical pattern is to zero out at the start of every iteration.

The exercise in `src/00_foundations/04_zero_grad.py` makes the accumulation
behavior explicit.

## Train vs eval mode

Some layers behave differently during training and inference:

- **Dropout** randomly zeros activations during training, does nothing during
  eval.
- **BatchNorm** uses batch statistics during training, running statistics
  during eval.

Switch with `model.train()` and `model.eval()`. Forgetting to switch to
eval mode before validation is a classic bug — your validation accuracy will
be noisy and low because dropout is still active.

The exercise in `src/00_foundations/15_train_eval_mode.py` demonstrates this.

## Validation

The honest version of the loop interleaves a validation pass:

```python
for epoch in range(num_epochs):
    model.train()
    for xb, yb in train_loader:
        # ... the six lines above ...
        ...

    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            # accumulate metrics
            ...
```

`torch.no_grad()` is important — without it, the validation forward pass
builds a gradient graph that consumes memory for no reason.

## When training goes wrong

A short checklist when your loss isn't going down:

1. **Can you overfit a single batch?** Run the model on the same 8-example
   batch for 500 steps. Loss should drop to near zero. If it doesn't, you
   have a bug in the model or loss — not in your data pipeline.
   See `src/00_foundations/29_overfit_small_batch.py`.
2. **Is the learning rate sane?** Default Adam (`lr=1e-3`) is fine for most
   things. If loss is `NaN` after a few steps, lr is too high. If loss is
   plateaued at the initial value, lr might be too low.
3. **Are gradients exploding?** Print `loss.item()` and the norm of one
   parameter's gradient after `backward()`. If gradient norms are `>1e3`,
   you need gradient clipping. See `src/00_foundations/26_grad_clipping.py`.
4. **Did you `optimizer.zero_grad()`?** Without it, gradients accumulate and
   the second step takes a wildly oversized update.
5. **Is your loss the right shape?** A common error: feeding raw logits to
   `BCELoss` instead of `BCEWithLogitsLoss`. The "WithLogits" variants take
   raw network output and do the sigmoid/softmax internally — more numerically
   stable.

## Checkpoints

In real training runs the loop also periodically saves a checkpoint:

```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss,
}, "checkpoint.pt")
```

This is what lets you resume after a crash, or roll back to the best
validation epoch. See `src/01_frameworks/14_save_load_state.py` for the
serialization mechanics.
