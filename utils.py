from typing import List, NoReturn, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from Tokenize import TokenizerX
# from torch.utils.data import dataset
from torch.utils.data import Dataset


def get_w2v(word2vec_path: str):
    """
    load the word embedding.
    """
    for line in open(word2vec_path, 'r', encoding="utf-8").read().strip().split('\n'):
        line = line.strip().split()
        if not line:  # or line[0] not in tokens:
            continue
        yield line[0], np.array(list(map(float, line[1:])))


def save_vocab(save_path: str, vocab: List[str]) -> NoReturn:
    """
    save vocab
    """
    with open(save_path, 'w', encoding='utf-8') as fw:
        for idx, token in enumerate(vocab):
            if idx == len(vocab) - 1:
                fw.writelines(token)
            else:
                fw.writelines(token + '\n')


def convert_w2v_to_embedding(w2v: dict, token_to_index: dict):
    """
    copy the vectors from word2vec file to nn.embedding
    """
    dim = len(w2v['我'])
    embed = nn.Embedding(len(token_to_index), dim, padding_idx=token_to_index['<pad>'])
    # print(embed.weight.shape)

    _, dim = embed.weight.shape
    for token in token_to_index.keys():
        if token == '<pad>':
            continue
        if token in w2v:
            assert len(w2v[token]) == dim, 'embedding dim error!'
            embed.weight.data[token_to_index[token]] = torch.FloatTensor(w2v[token])
            # torch.tensor(w2v[token], dtype=torch.FloatTensor)

    return embed


def char_tokenizer(sentence: str) -> List[str]:
    if sentence == '':
        return []
    else:
        return [token for token in sentence]


def seg(sentence: str) -> List[str]:
    return sentence.split(' ')


def convert_sample_to_indices(sample: Tuple[str, str, int], tokenizer4student, tokenizer4teacher, max_seq=510) -> Tuple[
    List[int], List[int], int]:
    """
    The input of the student model and the input of the teacher model should be processed separately, and the length
    should be less than 510.
    tokenizer4teacher： BertTokenizer
    tokenizer4student:
    """
    assert len(sample) == 3, 'The length of sample is error!'
    student_indices = tokenizer4student.convert_sentences_to_indices(sample[0], seg=TokenizerX.seg)
    student_indices = student_indices[:max_seq]

    teacher_tokens = tokenizer4teacher.tokenize(sample[1])
    teacher_tokens = teacher_tokens[:max_seq]
    teacher_indices = tokenizer4teacher.encode(teacher_tokens)

    return student_indices, teacher_indices, sample[2]


def prepare_data(samples: List[Tuple[str, str, int]], func) -> List[Tuple[List[int], List[int], int]]:
    result = []
    for sample in samples:
        result.append(func(sample))

    return result


class Mydataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def collate_fn(samples: List[Tuple[List[int], List[int], int]],
               teacher_padding_value=0, student_padding_value=1, output_teacher=True):
    # sample : (student inputs ids, teacher input ids, label)

    # labels
    labels = torch.LongTensor([sample[2]] for sample in samples)

    # process student inputs
    student_lengths = torch.LongTensor([len(sample[0]) for sample in samples])
    student_input = [torch.LongTensor([sample[0]]) for sample in samples]
    student_input = pad_sequence(student_input, padding_value=student_padding_value, batch_first=True)

    # process teacher inputs
    if output_teacher:
        teacher_lengths = [torch.LongTensor(len(sample[1]) * [1]) for sample in samples]
        teacher_inputs = [torch.LongTensor(sample[1]) for sample in samples]
        attn_mask = pad_sequence(teacher_lengths, padding_value=0, batch_first=True)
        input_ids = pad_sequence(teacher_inputs, padding_value=teacher_padding_value, batch_first=True)

        return student_input, student_lengths, input_ids, attn_mask, labels
    else:
        return student_input, student_lengths, labels


if __name__ == '__main__':
    # w2v = dict(get_w2v('./data/wordembedding/word2vec'))
    # print(len(w2v))

    print(char_tokenizer('依兰爱情故事'))

    bertTokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
    x = bertTokenizer.tokenize
    print(x('什么哇'))

    embed = nn.Embedding(3, 4)
    print(embed.weight.data.shape[1])
