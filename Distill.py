import collections
from functools import partial

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from torch.utils.data import DataLoader
from transformers import BertTokenizer

from DataAugment import data_augmentation
from StudentModel import LstmClassification
from Tokenize import TokenizerX
from model import BertBase
from utils import char_tokenizer, convert_sample_to_indices, prepare_data, Mydataset, collate_fn, get_w2v, \
    convert_w2v_to_embedding, save_vocab

train_path = './data/train.txt'
test_path = './data/test.txt'
counter = collections.Counter()
n_iter = 10

# construct vocab
vocab = TokenizerX(tokenizer=char_tokenizer, counter=counter)
texts = [line.split('\t', 1)[1].strip() for line in open(train_path, encoding="utf-8", ).read().strip().split('\n')]
vocab.counter_sequences(texts)

# if you want to update the vocab by the test set you can use the following code.
# texts = [line.split('\t', 1)[1].strip() for line in open(test_file_path, encoding="utf-8",
#                                                          ).read().strip().split('\n')]
# vocab.counter_sequences(texts)

# if '[MASK]' in vocab, delete it.'
vocab.update_vocab(add_tokens={}, discard_tokens={'[MASK]'})

save_vocab('./vocab.txt', vocab.index_to_token)

train_raw = data_augmentation(train_path, n_iter=n_iter, tokenizer=char_tokenizer)

path = './data/test.txt'
test_raw = data_augmentation(path, n_iter=0, tokenizer=char_tokenizer)

bertTokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
func = partial(convert_sample_to_indices, tokenizer4student=vocab, tokenizer4teacher=bertTokenizer)

train = prepare_data(train_raw, func=func)
test = prepare_data(test_raw, func=func)

train = Mydataset(train)
test = Mydataset(test)

print(vocab['<pad>'])
collate = partial(collate_fn, student_padding_value=vocab['<pad>'])
train_iter = DataLoader(train, batch_size=4, shuffle=True, collate_fn=collate)
collate = partial(collate_fn, student_padding_value=vocab['<pad>'], output_teacher=False)
test_iter = DataLoader(test, batch_size=4, shuffle=False, collate_fn=collate)

n_epochs = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
word_embedding = False
embedding_dim = 100

if word_embedding:
    w2v = get_w2v('./data/wordembedding/word2vec')
    embed = convert_w2v_to_embedding(w2v, vocab)

    model = LstmClassification(embed.weight.data.shape[0],
                               embed.weight.data.shape[1],
                               embedding=embed)
else:

    model = LstmClassification(len(vocab), embedding_dim)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# 和 FineTuningBert初始化参数保持一致
teacherModel = BertBase('./bert-base-chinese', cache_dir=None, num_labels=2)
teacherModel.load_state_dict(torch.load('./model/model.pt'))
teacherModel.to(device)
teacherModel.eval()

ce_loss = nn.CrossEntropyLoss()  # 交叉熵损失函数
mse_loss = nn.MSELoss()  # 均方误差损失函数

total_iter = 0
alpha = 0.25

for num in range(n_epochs):
    for i, bacth in enumerate(train_iter):

        total_iter += 1
        model.train()

        optimizer.zero_grad()

        student_input, student_lengths, input_ids, attn_mask, labels = tuple(element.to(device) for element in bacth)

        with torch.no_grad():
            # logits : [batch size, n_class]
            teacher_logits, _ = teacherModel(input_ids, attn_mask)

        logits = model(student_input, student_lengths)

        loss = alpha * ce_loss(logits, labels) + (
                1 - alpha) * mse_loss(logits, teacher_logits)

        loss.backward()
        optimizer.step()

        if total_iter % 100 == 0:
            # TODO: attention to save model.
            model.eval()

            preds = []
            turth = []
            for test_batch in test_iter:
                student_input, student_lengths, labels = tuple(t.to(device) for t in test_batch)

                with torch.no_grad():
                    logits = model(student_input, student_lengths)

                    preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
                    turth.extend(labels.detach().cpu().tolist())

            acc = accuracy_score(turth, preds)
            print(f"test accuracy : {acc}")
