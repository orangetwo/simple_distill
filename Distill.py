import collections
from functools import partial

from torch.utils.data import DataLoader
from transformers import BertTokenizer

from DataAugment import data_augmentation
from Tokenize import TokenizerX
from utils import char_tokenizer, convert_sample_to_indices, prepare_data, Mydataset, collate_fn

train_path = './data/train.txt'
test_path = './data/test.txt'
counter = collections.Counter()
n_iter = 10

# construct vocab
vocab = TokenizerX(tokenizer=char_tokenizer, counter=counter)
texts = [line.split('\t', 1)[1].strip() for line in open(train_path, encoding="utf-8", ).read().strip().split('\n')]
vocab.counter_sequences(texts)

# if you want to update the vocab by the test set you can use following code.
# texts = [line.split('\t', 1)[1].strip() for line in open(test_file_path, encoding="utf-8",
#                                                          ).read().strip().split('\n')]
# vocab.counter_sequences(texts)

# if '[MASK] in vocab, delete it.'
vocab.update_vocab(add_tokens={}, discard_tokens={'[MASK]'})


train_raw = data_augmentation(train_path, n_iter=n_iter, tokenizer=char_tokenizer)

path = './data/test.txt'
test_raw = data_augmentation(path, n_iter=0, tokenizer=char_tokenizer)

bertTokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
func = partial(convert_sample_to_indices, tokenizer4student=vocab, tokenizer4teacher=bertTokenizer)

train = prepare_data(train_raw, func=func)
test = prepare_data(test_raw, func=func)

train = Mydataset(train)
test = Mydataset(test)

train_iter = DataLoader(train, batch_size=16, shuffle=True, collate_fn=collate_fn)
