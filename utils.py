from typing import List
import numpy as np
import torch
from torch import nn


def get_w2v(word2vec_path):
    """
    load the word embedding.
    """
    for line in open(word2vec_path, 'r', encoding="utf-8").read().strip().split('\n'):
        line = line.strip().split()
        if not line:  # or line[0] not in tokens:
            continue
        yield line[0], np.array(list(map(float, line[1:])))


def save_vocab(save_path: str, vocab: List[str]):
    with open(save_path, 'w', encoding='utf-8') as fw:
        for token in vocab:
            fw.writelines(token + '\n')


def convert_w2v_to_embedding(w2v: dict, token_to_index: dict):
    dim = len(w2v['我'])
    embed = nn.Embedding(len(token_to_index), dim, padding_idx=token_to_index['<pad>'])
    print(embed.weight.shape)

    _,dim = embed.weight.shape
    for token in token_to_index.keys():
        if token == '<pad>':
            continue
        if token in w2v:
            assert len(w2v[token]) == dim, 'embedding dim error!'
            embed.weight.data[token_to_index[token]] = torch.FloatTensor(w2v[token])
            # torch.tensor(w2v[token], dtype=torch.FloatTensor)




if __name__ == '__main__':
    w2v = dict(get_w2v('./data/cache/word2vec'))
    print(len(w2v))
