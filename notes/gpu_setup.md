# GPU vs CPU: When to Bother

All the exercises in this repo run on CPU. Most finish in seconds. A handful
(the LSTM training, the GAN, the Transformer LM, the BERT fine-tune) take a
minute or two on CPU and ~10x less on a recent GPU. That's the difference
between "instant feedback" and "go make coffee" — worth setting up if you
plan to iterate.

## Do I need a GPU at all?

For *learning* deep learning, no. Every exercise here is sized to run on
a laptop. You learn the API and the math the same way regardless of hardware.

For *real work* — training a model that actually solves a problem on real
data — yes. A 2024-era consumer GPU is roughly 50-200x faster than a CPU
for dense float arithmetic. That changes what experiments are practical to
run.

## What you need

### NVIDIA + CUDA (the easy path)

If you have an NVIDIA GPU, you want CUDA. Both PyTorch and TensorFlow have
first-class CUDA support. The cleanest install is via the framework's own
instructions — they pin the right CUDA toolkit version.

For PyTorch: visit `pytorch.org`, the front page has a selector
(OS / package manager / Python version / CUDA version) that gives you the
exact `pip install` line. Don't try to install CUDA separately first; modern
PyTorch wheels bundle their own CUDA runtime.

Sanity check after install:

```python
import torch
print(torch.cuda.is_available())   # True
print(torch.cuda.get_device_name(0))
```

### AMD GPUs

PyTorch supports AMD via ROCm on Linux. Setup is more involved than CUDA.
TensorFlow's AMD support is patchier. If you have an AMD card and want to
do serious DL work on it, plan for an afternoon of setup.

### Apple Silicon (M1/M2/M3)

PyTorch has a `mps` backend (Metal Performance Shaders). It works for most
ops but isn't quite as fast or as well-tested as CUDA. Use `device="mps"`
in place of `device="cuda"`.

### Cloud GPUs

If you don't want to buy hardware:

- **Colab** (Google) — free tier gives you a few hours of K80/T4 per day.
  Sufficient for any exercise in this repo. Notebook-based.
- **Kaggle Notebooks** — similar to Colab, ~30 hours/week of T4 or P100.
- **Paperspace Gradient**, **Lambda Labs**, **vast.ai** — pay-per-hour rentals
  ranging from $0.20/hr (vast.ai consumer cards) to $2-4/hr (datacenter cards).
- **AWS / GCP / Azure** — the most expensive option per hour but the most
  integrated with the rest of their stack.

## The pattern every exercise uses

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MyModel().to(device)
x = x.to(device)
```

The exercise in `src/01_frameworks/13_device_placement.py` demonstrates this.
Code written this way runs anywhere; on CPU-only machines it just runs more
slowly.

## Common pitfalls

- **"`Expected all tensors to be on the same device`"** — you forgot to move
  one tensor. Trace back through your forward pass; one input is still on CPU.
- **"`CUDA out of memory`"** — your batch is too large. Cut `batch_size` in
  half and try again. The fix is rarely "buy more GPU"; it's usually "use a
  smaller batch" or "use gradient accumulation".
- **First call after `model.to("cuda")` is slow** — CUDA kernels are compiled
  on first use. Subsequent calls are fast. Don't benchmark the first call.
- **GPU is full but I just freed the model** — Python's garbage collector hasn't
  run yet. Call `del model; torch.cuda.empty_cache()` to release immediately.
