from collections import Counter
from typing import Union, List, Tuple, Set


class TokenizerX:
    def __init__(self, tokenizer, counter, max_size=None, min_freq=1,
                 filter_token=(), specials=(), lower=True,
                 unk='<unk>', pad='<pad>'):
        """
        Args:
            max_size: The maximum size of the vocabulary, or None for no
                    maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary. Default: ['<unk'>, '<pad>'].
            tokenizer: tokenizer function, Default: None.
            filter_token: filter the unuseful token. Default: ().
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
        """

        self.filter_token = filter_token
        self.min_freq = max(min_freq, 1)
        self.tokenizer = tokenizer
        self.counter = counter
        self.index_to_token = list()

        # self.index_to_token.extend(list(specials))
        self.token_to_index = dict()

        self.unk = unk
        self.pad = pad

        self.specials_tokens = specials + (self.unk, self.pad )
        self.max_size = max_size
        self.lower = lower

    def convert_sequence_to_tokens(self, sequence: str) -> list:
        """
        Args:
            sequence: Input sequence.
            if self.tokenizer is None, then using char-level.
        """

        if self.tokenizer is not None:
            if self.lower:
                return [token.lower() for token in self.tokenizer(sequence) if token not in self.filter_token]
            else:
                return [token for token in self.tokenizer(sequence) if token not in self.filter_token]
        else:
            if self.lower:
                return [token.lower() for token in sequence if token not in self.filter_token]
            else:
                return [token for token in sequence if token not in self.filter_token]

    def counter_sequences(self, sequences: List[str], save_words_and_frequencies=False):
        """
        Args:
            sequences: The list of sequence.
            save_words_and_frequencies: whether save the words & frequencies
        """
        if not isinstance(sequences, list):
            raise ValueError('The input sequences should be a list of str!')

        for sequence in sequences:
            assert type(sequence) == str, f'sequence should be str type, not ' \
                                          f'{type(sequence)} !!!'
            self.counter.update(self.convert_sequence_to_tokens(sequence))

        self.generate_vocab(save_words_and_frequencies=save_words_and_frequencies)

    def generate_vocab(self, save_words_and_frequencies=False):
        # sort by frequency, then alphabetically
        for tok in self.specials_tokens:
            if tok in self.counter:
                print(f"special token '{tok}' in your vocab!"
                      f"Please check your special tokens!!!")

        words_and_frequencies = sorted(self.counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        # TODO:
        if save_words_and_frequencies:
            pass

        self.index_to_token = list(self.specials_tokens)

        for word, freq in words_and_frequencies:
            if self.max_size is not None:
                if freq < self.min_freq or len(self.index_to_token) == (self.max_size + len(self.specials_tokens)):
                    break
                self.index_to_token.append(word)
            else:
                # if the token freq less than self.min_freq, discard the token
                if freq < self.min_freq:
                    break
                self.index_to_token.append(word)

        self.token_to_index = {tok: i for i, tok in enumerate(self.index_to_token)}

    def convert_sentences_to_indices(self, sentences: Union[str, List[str]], unk='<unk>', seg=None):

        if seg is not None:
            tokenizer_sent = seg
        else:
            tokenizer_sent = self.tokenizer

        if isinstance(sentences, str):
            return [self.token_to_index.get(token, self.token_to_index[unk]) for token in tokenizer_sent(sentences)]

        elif isinstance(sentences, list):
            result = []

            for sentence in sentences:
                result.append(
                    [self.token_to_index.get(token, self.token_to_index[unk]) for token in tokenizer_sent(sentence)])

            return result
        else:
            raise ValueError('The input is neither a list nor a string!')

    def convert_indices_to_sentences(self, indices: Union[List[int], List[List[int]]]):

        assert len(indices) >= 1, 'The length of indices is less than 1!'

        if isinstance(indices[0], int):
            return [self.index_to_token[index] for index in indices]
        elif isinstance(indices[0], list):
            result = []
            for indie in indices:
                result.append([self.index_to_token[index] for index in indie])
            return result
        else:
            raise ValueError('The input is neither a list of int nor a list of list!')

    def update_vocab(self, add_tokens: Set[str], discard_tokens: Set[str]):
        if add_tokens:
            self.index_to_token.extend([token for token in add_tokens if token not in self.token_to_index])
            self.token_to_index = {token: idx for idx, token in enumerate(self.index_to_token)}

            for token in add_tokens:
                if token not in self.counter:
                    self.counter[token] = -1

        if discard_tokens:
            self.index_to_token = [token for token in self.index_to_token if token not in discard_tokens]
            self.token_to_index = {token: idx for idx, token in enumerate(self.index_to_token)}

            for token in discard_tokens:
                if token in self.counter:
                    del self.counter[token]

    @staticmethod
    def seg(sentence: str) -> List[str]:

        return sentence.split(' ')

    def __getitem__(self, token):

        assert len(self.token_to_index) > 2, f'The current dictionary size is {len(self.token_to_index)}!!!'

        if '<unk>' not in self.token_to_index:
            print(f"Please check the '<unk> token'!")

        return self.token_to_index.get(token, self.token_to_index.get(self.unk))

    def __len__(self):
        return len(self.index_to_token)


if __name__ == '__main__':
    counter = Counter()


    def charX(sentence: str) -> List[str]:
        if sentence == '':
            return []
        else:
            return [token for token in sentence]


    vocab = TokenizerX(tokenizer=charX, counter=counter)

    vocab.counter_sequences(['不错，下次还考虑入住。交通也方便，在餐厅吃的也不错。'])
    print(f"\nupdate vocab:")
    print(vocab.index_to_token)
    print(vocab.counter)
    print(vocab.token_to_index)

    vocab.counter_sequences(['不错，下次还考虑入住。交通也方便，在餐厅吃的也不错。嘀嘀咕咕'])
    print(f"\nupdate vocab:")
    print(vocab.index_to_token)
    print(vocab.counter)
    print(vocab.token_to_index)

    # vocab.update_vocab(add_tokens={'wx'}, discard_tokens={})
    # print(f"\nupdate vocab:")
    # print(vocab.index_to_token)
    # print(vocab.counter)
    # print(vocab.token_to_index)

    tmp = '不 错 [MASK] [MASK] [MASK] 还 考 [MASK] 入 住 。 交 通 也 方 便 [MASK] 在 餐 厅 吃 的 [MASK] [MASK] 错 。'
    indices = vocab.convert_sentences_to_indices(sentences=tmp, seg=TokenizerX.seg)
    print(indices)
    print(vocab.convert_indices_to_sentences(indices))


    print(vocab[vocab.pad])

