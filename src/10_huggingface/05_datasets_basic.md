# Exercise 05: HuggingFace Datasets

Load the `glue/sst2` dataset (small sentiment dataset).

```
from datasets import load_dataset
ds = load_dataset("glue", "sst2")
```

Print the split sizes and the first training example.
