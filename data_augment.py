import collections

import jieba
import numpy as np
import torch

from Tokenize import tokenizer,Tokenizer


def ngram_sampling(words, p_ng=0.25, ngram_range=(2, 6)):
    """
    n-gram sampling
    """
    if np.random.rand() < p_ng:
        ngram_len = np.random.randint(ngram_range[0], ngram_range[1] + 1)
        ngram_len = min(ngram_len, len(words))
        start = np.random.randint(0, len(words) - ngram_len + 1)
        words = words[start:start + ngram_len]

    return words


def data_augmentation(dataPath, p_mask=0.75, p_ng=0.25, ngram_range=(2, 6), n_iter=20, tokenizer=jieba.lcut,
                      min_length=2):
    """
    generate data augment train set.
    """
    input4student_input4bert_label = []

    with open(dataPath, 'r', encoding='utf-8') as fr:
        for line in fr:
            tmp = line.strip().split('\t', 1)
            assert len(tmp) == 2, 'error line!'

            label, text = tmp
            input4student = ' '.join(tokenizer(text))
            input4teacher = text
            if len(input4student) <= min_length or len(text) <= min_length:
                continue

            input4student_input4bert_label.append((input4student, input4teacher, int(label)))
            added_inputs = {input4teacher}

            # 数据增强,每句话增强的次数
            for i in range(n_iter):
                # 1. Masking
                tokens = [x if np.random.rand() < p_mask else "[MASK]" for x in tokenizer(text)]
                # 2. n-gram sampling
                tokens = ngram_sampling(tokens, p_ng, ngram_range)

                input4teacher = ''.join(tokens)
                # 防止重复加入
                if input4teacher not in added_inputs:
                    input4student = ' '.join(tokens)
                    input4student_input4bert_label.append((input4student, input4teacher, int(label)))

    return input4student_input4bert_label


def get_w2v():
    """
    load the word embedding.
    """
    for line in open('data/cache/word2vec', encoding="utf-8").read().strip().split('\n'):
        line = line.strip().split()
        if not line:  # or line[0] not in tokens:
            continue
        yield line[0], np.array(list(map(float, line[1:])))


if __name__ == '__main__':
    path = './data/hotel/test.txt'
    # x = data_augmentation(path, n_iter=10)
    # for y in x:
    #     print(y)

    tokenizer = tokenizer

    counter = collections.Counter()
    vocab = Tokenizer(tokenizer=tokenizer, counter=counter)
    texts = [' '.join(jieba.cut(line.split('\t', 1)[1].strip())) for line in open(path, encoding="utf-8",).read().strip().split('\n')]

    from torch import nn

    embeds = nn.Embedding(2, 5)  # 2 个单词，维度 5

    print(embeds.weight.data)
    print(type(embeds.weight.data))
    embeds.weight.data[0] = torch.FloatTensor([1.1820, 0.0601, 0.1449, 0.4057, 1.3008])

    print(embeds.weight.data)

    print({0: '<unk>', 1: '不错', 2: '，', 3: '。', 4: '也', 5: '下次', 6: '还', 7: '考虑', 8: '入住', 9: '交通', 10: '方便', 11: '在',
           12: '餐厅', 13: '吃', 14: '的'})
    print({'<unk>': 0, '不错': 1, '，': 2, '。': 3, '也': 4, '下次': 5, '还': 6, '考虑': 7, '入住': 8, '交通': 9, '方便': 10, '在': 11,
           '餐厅': 12, '吃': 13, '的': 14})
    # print(texts)
    vocab.counter_sequences(texts)

    # print(vocab.index_to_token)
    # print(vocab.token_to_index)

    print(vocab.convert_sentences_to_indices('不错 ， 下次 还 考虑 入住 。 交通 也 方便 ， 在 餐厅 吃 的 也 不错 。'))
    indices = [23, 2, 94, 22, 424, 28, 4, 130, 14, 46, 2, 15, 114, 115, 3, 14, 23, 4]
    print(vocab.convert_indices_to_sentences(indices))

    print(len(vocab))
