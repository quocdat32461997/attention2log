# train.py

import math
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from parser import get_model_args
from a2l.dataset import *
from a2l.models import *
from a2l.utils import to_cuda


def main(args, configs):

    # initialize tokenizer
    tokenizer = LogTokenizer(vocab=args.vocab)

    # initialize datasets and convert to dataloaders
    train_dataset = MaskedLogDataset(args.data,
                               window_size=configs['window_size'],
                               max_len=configs['max_len'],
                               tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset,
                               batch_size=configs['batch_size'],
                               shuffle=configs['shuffle'],
                               pin_memory=True)

    if args.eval_data:
        eval_dataset = LogDataset(args.eval_data,
                                  window_size=configs['window_size'],
                                  max_len=configs['max_len'],
                                  tokenizer=tokenizer)
        eval_dataloader = DataLoader(eval_dataset,
                                  batch_size=configs['batch_size'],
                                  shuffle=configs['shuffle'],
                                  pin_memory=True)

    # initialize model
    model = MaskedLogTransformer(num_class=tokenizer.get_vocab_size(),
                           vocab_size=tokenizer.get_vocab_size(),
                           hidden_size=configs['hidden_size'],
                           decoder_hidden_size=configs['decoder_hidden_size'],
                           max_len=configs['max_len'],
                           num_layer=configs['num_layer'],
                           num_head=configs['num_head'],
                           dropout=configs['dropout'])
    model = to_cuda(model)
    print(model)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)

    # training pipeline
    writer = SummaryWriter(log_dir=configs['saved'])
    for epoch in range(configs['epochs']):
        batch_loss, batch_acc, batch_f1 = 0, 0, 0
        start_time = time.time()

        for inputs, labels in train_dataloader:
            inputs, labels = to_cuda(inputs), to_cuda(labels) # convert to cuda tensors

            # forward
            optimizer.zero_grad()
            loss, acc, f1 = model(inputs, labels)

            # backprop
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
            batch_acc += acc
            batch_f1 += f1

        # log loss
        log_interval = int(train_dataset.__len__() / configs['batch_size'])
        #if batch_idx %  log_interval == 0 and batch_idx > 0:
        cur_loss = batch_loss / log_interval
        cur_acc = batch_acc / log_interval
        cur_f1 = batch_f1 / log_interval
        elapsed = time.time() - start_time
        print('| epoch {:3d} | '
              'lr {:02.6f} | {:5.2f} ms | '
              'loss {:5.5f} | acc {:.2f} | f1 {:.2f}'.format(
                epoch, scheduler.get_lr()[0],
                elapsed * 1000 / log_interval,
                cur_loss, cur_acc, cur_f1))

        # write logs
        writer.add_scalar('Loss', cur_loss, epoch)
        writer.add_scalar('Acc', cur_acc, epoch)
        writer.add_scalar('F1', cur_f1, epoch)

        # update lr
        scheduler.step()

    # stop writing logs
    writer.flush()
    writer.close()

    # save model
    torch.save(model.state_dict(),
                 path=os.path.join(configs['saved'], 'model.pt'))


if __name__ == '__main__':
    # get argument arguments
    args, configs = get_model_args()
    # the training pipeline with given args
    main(args, configs)
