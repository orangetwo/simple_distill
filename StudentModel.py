import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec

class LstmClassification(nn.Module):
    def __init__(self, vocab_size, embed_dim, embedding=None, n_class=2, hidden_dim=None, pad_idx=1):
        super(LstmClassification, self).__init__()
        if embedding is not None:
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
            self.embed.weight.data = embedding.weight.data
        else:
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        if hidden_dim is None:
            hidden_dim = embed_dim
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)

        self.ff = nn.Linear(hidden_dim, n_class)

    def forward(self, x, lengths, **kwargs):
        # x : (batch_size X seq_len)
        x = self.embed(x)

        packed_input = pack_padded_sequence(x, lengths, batch_first=True)
        packed_output, (ht, ct) = self.lstm(packed_input)

        # packed_output.data.shape : (batch_sum_seq_len X hidden_dim)
        # output.shape : ( batch_size X seq_len X hidden_dim)
        # output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        logits = self.ff(ht[-1])

        return logits


class TextCnn(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass


if __name__ == '__main__':
    model = LstmClassification(10, 10)