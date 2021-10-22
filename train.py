# train.py

import math
import time
import torch
from torch.utils.data import DataLoader

from parser import get_model_args
from a2l.dataset import *
from a2l.models import *
from a2l.utils import to_cuda


def main(args, configs):
    # initialize datasets and convert to dataloaders
    train_dataset = LogDataset(args.data,
                               window_size=configs['window_size'],
                               vocab=args.vocab)
    train_dataloader = DataLoader(train_dataset,
                               batch_size=configs['batch_size'],
                               shuffle=configs['shuffle'],
                               pin_memory=True)

    if args.eval_data:
        eval_dataset = LogDataset(args.eval_data,
                                  window_size=configs['window_size'],
                                  vocab=args.vocab)
        eval_dataloader = DataLoader(eval_dataset,
                                  batch_size=configs['batch_size'],
                                  shuffle=configs['shuffle'],
                                  pin_memory=True)

    # initialize model
    model = LogTransformer(num_class=configs['num_class'],
                           vocab_size=train_dataset.get_vocab_size(),
                           hidden_size=configs['hidden_size'],
                           decoder_hidden_size=configs['decoder_hidden_size'],
                           max_len=configs['max_len'],
                           num_layer=configs['num_layer'],
                           num_head=configs['num_head'],
                           dropout=configs['dropout'])
    print(model)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # training pipeline
    for epoch in range(configs['epochs']):
        batch_idx, batch_loss = 0, 0
        start_time = time.time()

        for inputs, labels in train_dataloader:
            inputs, labels = to_cuda(inputs), to_cuda(labels) # convert to cuda tensors
            optimizer.zero_grad()

            # forward
            loss = model(inputs, labels)
            batch_loss += loss.item()

            # log loss
            log_interval = int(train_dataset.__len__() / configs['batch_size'])
            if batch_idx %  log_interval == 0 and batch_idx > 0:
                cur_loss = batch_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d} batches | '
                      'lr {:02.6f} | {:5.2f} ms | '
                      'loss {:5.5f} | ppl {:8.2f}'.format(
                    epoch, batch_idx, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
                batch_loss = 0
                start_time = time.time()

            # backprop
            loss.backward()
            optimizer.step()

            batch_idx += 1


if __name__ == '__main__':
    # get argument arguments
    args, configs = get_model_args()
    # the training pipeline with given args
    main(args, configs)
