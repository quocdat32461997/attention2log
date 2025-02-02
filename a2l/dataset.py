# dataset.py

import os
import torch

from a2l.utils import read_vocab


class LogTokenizer:
    def __init__(self, vocab):
        self.vocab = read_vocab(vocab)
        self.i_vocab = {v: k for k,v in self.vocab.items()}

    def get_vocab_size(self):
        return len(self.vocab)

    def decode(self, inputs):
        if not isinstance(inputs, list):
            inputs = list(inputs)
        return [self.i_vocab[x] for x in inputs]

    def encode(self, inputs):
        if not isinstance(inputs, list):
            inputs = list(inputs)
        return [self.vocab[x] for x in inputs]

    def __call__(self, log):
        return [self.vocab['[CLS]']] + self.encode(log) + [self.vocab['[SEP]']]


class LogDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size, max_len, tokenizer):
        super(LogDataset, self).__init__()
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.max_len = max_len

        # read data
        self.inputs, self.labels = self._read_data(data)

    def _read_data(self, data):
        # Function to read HFDS data
        # Args:
        #   - data: str
        #       Path to data
        #   - testing: bool
        #       Flag to indicate if data is a testing dataset
        #       If a testing dataset, return inputs only.

        # TODO: handle testing datasets when testing=True
        assert os.path.exists(data), 'Provided data {} does not exists'.format(data)

        num_sessions = 0
        inputs, labels = [], []
        with open(data, 'r') as file:
            for line in file.readlines():
                num_sessions += 1
                line = tuple(map(str, line.strip().split()))
                #line = tuple(map(lambda n: n - 1, map(str, line.strip().split())))
                for i in range(len(line) - self.window_size):
                    inputs.append(line[i:i + self.window_size])
                    labels.append(line[i + self.window_size])

        # print dataset stats
        self.num_session = num_sessions
        self.num_seq = len(inputs)
        print('Printing dataset statistics: {}'.format(data))
        print('Number of sessions({}): {}'.format(data, self.num_session))
        print('Number of seqs({}): {}'.format(data, self.num_seq))

        return inputs, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # extract inputs and labels
        inputs = self.tokenizer(self.inputs[idx])
        label = self.tokenizer.vocab[self.labels[idx]]
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(label)

class MaskedLogDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size, max_len, tokenizer):
        super(MaskedLogDataset, self).__init__()
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.max_len = max_len

        # read data
        self.inputs, self.labels = self._read_data(data)

    def _read_data(self, data):
        # Function to read HFDS data
        # Args:
        #   - data: str
        #       Path to data
        #   - testing: bool
        #       Flag to indicate if data is a testing dataset
        #       If a testing dataset, return inputs only.

        # TODO: handle testing datasets when testing=True
        assert os.path.exists(data), 'Provided data {} does not exists'.format(data)

        num_sessions = 0
        inputs, labels = [], []
        with open(data, 'r') as file:
            for line in file.readlines():
                num_sessions += 1
                line = tuple(map(str, line.strip().split()))
                #line = tuple(map(lambda n: n - 1, map(str, line.strip().split())))
                for i in range(len(line) - self.window_size):
                    inputs.append(line[i:i + self.window_size])
                    labels.append(line[i + self.window_size])

        # print dataset stats
        self.num_session = num_sessions
        self.num_seq = len(inputs)
        print('Printing dataset statistics: {}'.format(data))
        print('Number of sessions({}): {}'.format(data, self.num_session))
        print('Number of seqs({}): {}'.format(data, self.num_seq))

        return inputs, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # extract inputs and labels
        inputs = list(self.inputs[idx])
        targets = self.labels[idx]

        # format to masked language inputs & targets
        targets = self.tokenizer(inputs + [targets])
        inputs = targets[:-2] + [self.tokenizer.vocab['[MASK]']] + targets[-1:]
        return torch.tensor(inputs, dtype=torch.long),\
               torch.tensor(targets, dtype=torch.long)