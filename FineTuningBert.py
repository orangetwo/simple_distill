# -*- coding: utf-8 -*-

import argparse
import os
from functools import partial

import torch


from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup, BertTokenizer, AdamW

from model import BertBase

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def args():
    """
    Configuration parameters.
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--max_length', type=int, default=100, help='the max length should be less than or '
                                                                    'equal to 510')
    parser.add_argument('--num_epochs', type=int, default=2, help='the num of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--model', type=str, default='bert-base-chinese')
    parser.add_argument('--train', type=str, default='./data/train.txt')
    parser.add_argument('--test', type=str, default='./data/test.txt')
    parser.add_argument('--cache_dir', type=str, default=None, help='Model cache path')
    parser.add_argument('--warmup_ratio', type=float, default=0.1)

    parser.add_argument('--bert_model', type=str, default='./bert-base-chinese')

    args = parser.parse_args()

    def print_args(argx):
        print('--------args----------')
        for k in list(vars(argx).keys()):
            print('%s: %s' % (k, vars(argx)[k]))
        print('--------args----------\n')

    print_args(args)

    return args


class MyData(object):
    """
    读取train test文件, 并把文本转化为索引。返回list,如[(indices,label)]
    """

    def __init__(self, arg, tokenizer):

        assert arg.max_length <= 510, 'max length more than 510, error!'
        self.max_seq = arg.max_length
        self.tokenizer = tokenizer
        self.labels = set()
        if arg.train:
            self.train = self.get_examples(arg.train, add_label=True)
        else:
            raise ValueError('no train set!')
        if arg.test:
            self.test = self.get_examples(arg.test)

    def get_examples(self, path, add_label=False):
        examples = []
        func = partial(MyData.convert_text_to_indices, max_seq=self.max_seq, tokenizer=self.tokenizer)
        with open(path, 'r', encoding="utf-8") as f:
            for i, line in enumerate(f):
                label, text = line.strip().split('\t', 1)
                idxs = func(text)
                if not idxs:
                    continue
                examples.append((idxs, int(label)))
                if add_label:
                    self.labels.add(int(label))

        return examples

    @staticmethod
    def convert_text_to_indices(text, max_seq, tokenizer):
        """
        convert text to indices
        """

        tokens = tokenizer.tokenize(text)

        # or
        # tokens = tokens[:max_seq]
        # indices = tokenizer.encode(tokens, add_special_tokens=True)

        tokens = ["[CLS]"] + tokens[:max_seq] + ["[SEP]"]
        indices = tokenizer.convert_tokens_to_ids(tokens)

        return indices


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def collate_fn(examples, padding_idx=0):
    """
    process the batch samples
    """
    masks = [torch.tensor(len(ex[0]) * [1]) for ex in examples]
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)

    inputs = pad_sequence(inputs, batch_first=True, padding_value=padding_idx)
    # In transformer library, 1 denote not masked, 0 denote masked
    masks = pad_sequence(masks, batch_first=True, padding_value=0)

    assert inputs.shape == masks.shape, 'inputs shape not equals to masks!!!'
    return inputs, masks, targets


def main():
    arg = args()
    tokenizer = BertTokenizer.from_pretrained(arg.bert_model, do_lower_case=True)
    mydata = MyData(arg, tokenizer)
    print(f"训练集规模 :{len(mydata.train)}")
    train = MyDataset(mydata.train)
    test = MyDataset(mydata.test)

    labels = mydata.labels

    model = BertBase.from_pretrained(arg.bert_model,
                                               cache_dir=arg.cache_dir, num_labels=len(labels))

    # You can choose this model.
    # model = BertTextCNN.from_pretrained(bert_model,
    #                                     cache_dir=arg.cache_dir, num_labels=len(labels))

    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not \
            any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if \
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.00}]
    print('train...')
    func = partial(collate_fn, padding_idx=tokenizer.vocab['[PAD]'])
    train_dataloader = DataLoader(train, batch_size=arg.batch_size, collate_fn=func, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=arg.batch_size, collate_fn=func, shuffle=False)

    total_steps = len(train_dataloader) * arg.num_epochs

    optimizer = AdamW(optimizer_grouped_parameters, lr=arg.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps * arg.warmup_ratio,
                                                num_training_steps=total_steps)

    if not os.path.exists('./model/'):
        os.makedirs('./model')

    total_step = 0
    best = float('-inf')
    #
    # for step, batch in enumerate(tqdm(train_dataloader, desc=f'Training Epoch')):
    #     input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
    #     print(input_ids.shape)

    print(f"start ->>> ")
    for _ in range(arg.num_epochs):

        train_loss = []
        for step, batch in enumerate(
                train_dataloader):  # enumerate(tqdm(train_dataloader, desc=f'Training Epoch {_}')):

            model.train()
            optimizer.zero_grad()

            total_step = total_step + 1
            input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
            # print(f"input_ids shape : {input_ids.shape}")
            loss, pred = model(input_ids, input_mask, label_ids)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            train_loss.append(loss.item())

            if total_step % 20 == 0:
                # print(f"accuracy : {accuracy_score(label_ids.tolist(), pred)}")
                print(f"this batch loss :{loss.item()}")

            if total_step % 100 == 0:
                model.eval()
                preds = []
                turth = []
                for batch in test_dataloader:
                    input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
                    with torch.no_grad():
                        logits, pred = model(input_ids, input_mask, None)
                        preds.extend(pred.detach().cpu().tolist())
                        turth.extend(label_ids.detach().cpu().tolist())

                acc = accuracy_score(turth, preds)
                print(f"test set accuracy : {acc}")

                if acc > best:
                    best = acc
                    print(f"save the model ->>> /model/model.pt")
                    torch.save(model.state_dict(), './model/model.pt')
        print(f'avg train_loss : {sum(train_loss) / len(train_loss)}')


if __name__ == '__main__':
    main()
