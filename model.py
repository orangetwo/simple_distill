import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BertBase(BertPreTrainedModel):
    def __init__(self, config, num_labels=2):
        super(BertBase, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.init_weights()

    def forward(self, input_ids, input_mask, label_ids):
        pooled_output = self.bert(input_ids, None, input_mask).pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        pred = torch.max(logits, 1)[1]
        if label_ids is not None:
            loss_fct = CrossEntropyLoss()
            return loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1)), pred
        return logits, pred


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


class BertTextCNN(BertPreTrainedModel):
    def __init__(self, config, hidden_size=128, num_labels=2):
        super(BertTextCNN, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.conv1 = nn.Conv2d(1, hidden_size, (3, config.hidden_size))
        self.conv2 = nn.Conv2d(1, hidden_size, (4, config.hidden_size))
        self.conv3 = nn.Conv2d(1, hidden_size, (5, config.hidden_size))
        self.classifier = nn.Linear(hidden_size * 3, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, input_mask, label_ids):
        sequence_output, _ = self.bert(input_ids, None, input_mask, output_all_encoded_layers=False)
        out = self.dropout(sequence_output).unsqueeze(1)
        c1 = torch.relu(self.conv1(out).squeeze(3))
        p1 = F.max_pool1d(c1, c1.size(2)).squeeze(2)
        c2 = torch.relu(self.conv2(out).squeeze(3))
        p2 = F.max_pool1d(c2, c2.size(2)).squeeze(2)
        c3 = torch.relu(self.conv3(out).squeeze(3))
        p3 = F.max_pool1d(c3, c3.size(2)).squeeze(2)
        pool = self.dropout(torch.cat((p1, p2, p3), 1))
        logits = self.classifier(pool)
        pred = torch.max(logits, 1)[1]
        if label_ids is not None:
            loss_fct = CrossEntropyLoss()
            return loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1)), pred
        return logits, pred
