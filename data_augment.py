import jieba
import numpy as np


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


def data_augmentation(dataPath, p_mask, p_ng, ngram_range=(2,6), n_iter=20,tokenizer=jieba.lcut, min_length=2):

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
        if not line:
            continue
        yield line[0], np.array(list(map(float, line[1:])))
