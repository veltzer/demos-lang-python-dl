#!/usr/bin/env python

"""Pack and unpack variable-length sequences."""

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


def main() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)

    seqs = [torch.randn(5, 2), torch.randn(3, 2), torch.randn(4, 2)]
    lengths = torch.tensor([len(s) for s in seqs])

    order = torch.argsort(lengths, descending=True)
    seqs = [seqs[i] for i in order]
    lengths = lengths[order]

    padded = pad_sequence(seqs, batch_first=True)
    packed = pack_padded_sequence(padded, lengths, batch_first=True)

    lstm = nn.LSTM(2, 8, batch_first=True)
    packed_out, _ = lstm(packed)
    out, out_lengths = pad_packed_sequence(packed_out, batch_first=True)

    print(out.shape)
    print(out_lengths.tolist())
    return out, out_lengths


if __name__ == "__main__":
    main()
