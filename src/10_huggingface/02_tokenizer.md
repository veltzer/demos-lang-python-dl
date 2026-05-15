# Exercise 02: AutoTokenizer

Load the BERT-base-uncased tokenizer:

```
AutoTokenizer.from_pretrained("bert-base-uncased")
```

Tokenize the sentence "Deep learning is fun." Print:
- the input IDs
- the decoded tokens (`tokenizer.convert_ids_to_tokens`)
