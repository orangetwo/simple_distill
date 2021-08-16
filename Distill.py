import collections
from functools import partial

from transformers import BertTokenizer

from DataAugment import data_augmentation
from Tokenize import TokenizerX
from utils import char_tokenizer, convert_sample_to_indices, prepare_data, Mydataset

path = './data/hotel/DA.txt'
counter = collections.Counter()
n_iter = 10

# construct vocab
vocab = TokenizerX(tokenizer=char_tokenizer, counter=counter)
texts = [line.split('\t', 1)[1].strip() for line in open(path, encoding="utf-8", ).read().strip().split('\n')]
vocab.counter_sequences(texts)

# update vocab
vocab.update_vocab(add_tokens={}, discard_tokens={'[MASK]'})

path = './data/hotel/train.txt'
train_raw = data_augmentation(path, n_iter=n_iter, tokenizer=char_tokenizer)

path = './data/hotel/test.txt'
test_raw = data_augmentation(path, n_iter=0, tokenizer=char_tokenizer)

bertTokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
func = partial(convert_sample_to_indices, tokenizer4student=vocab, tokenizer4teacher=bertTokenizer)

train = prepare_data(train_raw, func=func)
test = prepare_data(test_raw, func=func)

train = Mydataset(train)
test = Mydataset(test)


class Dataloader(object):
    pass


train_iter = Dataloader(train,batch_size =16,shuffle=True,collate_fn=collate_fn)
