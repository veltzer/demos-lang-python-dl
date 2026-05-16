# Exercise 06: Map a Tokenizer Over a Dataset

Take the `sst2` train split, keep only the first 100 examples (use `select`),
and use `Dataset.map` to tokenize each example's `sentence` column with
`bert-base-uncased`.

Print the keys of one tokenized example.
