# utils.py

import torch

is_cuda = torch.cuda.is_available()


def read_vocab(file):
    # Function to read a text file of log keys and returns a dict
    # Args:
    #   - file: str

    # read vocab
    with open(file) as file:
        vocabs = file.read().split('\n')

    # split pair of enumerated_event_id and raw_event_id
    # parse to dict
    return {k: i for i, k in enumerate(vocabs)}


def to_cuda(x):
    if is_cuda:
        x = x.cuda(0)
    return x