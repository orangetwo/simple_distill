from collections import Counter
from typing import Union,List


class Tokenizer:
    def __init__(self, tokenizer, counter, max_size=None, min_freq=1,
                 filter_token=(), specials=('<unk>', '<pad>'), lower=True):
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
            collections.Counter object holding the frequencies of
                each value found in the data.
        """

        self.filter_token = filter_token
        self.min_freq = max(min_freq, 1)
        self.tokenizer = tokenizer
        self.counter = counter
        self.index_to_token = list()
        self.specials = specials

        self.index_to_token.extend(list(specials))
        self.token_to_index = dict()

        self.unk = specials[0]
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
        for tok in self.specials:
            del self.counter[tok]

        words_and_frequencies = sorted(self.counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        # TODO:
        if save_words_and_frequencies:
            pass

        for word, freq in words_and_frequencies:
            if self.max_size is not None:
                if freq < self.min_freq or len(self.itos) == (self.max_size + len(self.specials)):
                    break
                self.index_to_token.append(word)
            else:
                # if the token freq less than self.min_freq, discard the token
                if freq < self.min_freq:
                    break
                self.index_to_token.append(word)

        self.token_to_index.update({tok: i for i, tok in enumerate(self.index_to_token)})

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

    def __getitem__(self, token):

        assert len(self.token_to_index) > 2, f'The current dictionary size is {len(self.token_to_index)}!!!'

        if '<unk>' not in self.token_to_index:
            print(f"Please check the '<unk> token'!")

        return self.token_to_index.get(token, self.token_to_index.get(self.unk))

    def __len__(self):
        return len(self.index_to_token)


def tokenizer(sentence):
    return [token for token in sentence.split(' ')]


if __name__ == '__main__':
    counter = Counter()

    vocab = Tokenizer(tokenizer=tokenizer, counter=counter)

    vocab.counter_sequences(['不错 ， 下次 还 考虑 入住 。 交通 也 方便 ， 在 餐厅 吃 的 也 不错 。'])
    # vocab.generate_vocab()

    print(vocab.index_to_token)
    print(vocab.counter)
    print(vocab.token_to_index)
