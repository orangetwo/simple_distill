import jieba
import collections
import numpy as np
from Tokenize import Tokenizer
from typing import List, Tuple
from utils import save_vocab, convert_w2v_to_embedding, get_w2v, char_tokenizer, seg


def ngram_sampling(words: List[str], p_ng=0.25, ngram_range=(4, 10)) -> List[str]:
    """
    n-gram sampling
    """
    if np.random.rand() < p_ng:
        ngram_len = np.random.randint(ngram_range[0], ngram_range[1] + 1)
        ngram_len = min(ngram_len, len(words))
        start = np.random.randint(0, len(words) - ngram_len + 1)
        words = words[start:start + ngram_len]

    return words


def data_augmentation(dataPath, p_mask=0.75, p_ng=0.25, ngram_range=(4, 10), n_iter=20, tokenizer=jieba.lcut,
                      min_length=2) -> List[Tuple[str, str, int]]:
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


if __name__ == '__main__':
    """
        1. Builds the dictionary on the specified text, paying attention to the implementation of the word \
            segmentation 
        2. Convert each sentence to the corresponding index in the dictionary. It is important to \
            note that the word segmentation in this step is the same as in the first step. 
    """

    path = './data/hotel/train.txt'
    counter = collections.Counter()
    n_iter = 10

    # If you want to use word-level, you can use the following code.
    # Or you can define the word segmentation instead of jieba.lcut.
    # x = data_augmentation(path, n_iter=n_iter, tokenizer=jieba.lcut)
    # vocab = Tokenizer(tokenizer=jieba.lcut, counter=counter)

    # Now we use char-level
    x = data_augmentation(path, n_iter=n_iter, tokenizer=char_tokenizer)
    vocab = Tokenizer(tokenizer=char_tokenizer, counter=counter)

    texts = [line.split('\t', 1)[1].strip() for line in open(path, encoding="utf-8", ).read().strip().split('\n')]

    vocab.counter_sequences(texts)


    # check
    indices = vocab.convert_sentences_to_indices('不错，下次还考虑入住。交通也方便，在餐厅吃的也不错。')
    print(vocab.convert_indices_to_sentences(indices))

    tmp = '不 错 [MASK] [MASK] [MASK] 还 考 [MASK] 入 住 。 交 通 也 方 便 [MASK] 在 餐 厅 吃 的 [MASK] [MASK] 错 。'
    indices = vocab.convert_sentences_to_indices(sentences=tmp, seg=seg)
    print(indices)

    print(vocab.convert_indices_to_sentences(indices))

    print(x[2])
    # save_vocab('./vocab.txt', vocab.index_to_token)




