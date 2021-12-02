# berts.py

import math
import torch
from torch.nn import functional as F
from torchmetrics import Accuracy, F1, Precision, Recall
from a2l.utils import to_cuda


class PositionEncoding(torch.nn.Module):
    def __init__(self, hidden_size, max_len=5000, name='PositionEncoding'):
        # PositionEncoding from https://github.com/oliverguhr/transformer-time-series-prediction/blob/570d39bc0bbd61c823626ec9ca8bb5696c068187/transformer-singlestep.py#L25
        super(PositionEncoding, self).__init__()
        self.pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_team = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        self.pe[:, 0::2] = torch.sin(position * div_team)
        self.pe[:, 1::2] = torch.cos(position * div_team)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, inputs):
        return self.pe[:, inputs.size(1)]


class LogClassifier(torch.nn.Module):
    def __init__(self, input_hidden_size, hidden_size, num_class, num_layer=2, dropout=0.0, name='LogClassifier'):
        super(LogClassifier, self).__init__()

        # initialize hidden layers
        self.classifier = []
        for _ in range(num_layer):
            self.classifier.extend([
                torch.nn.Linear(input_hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout)
            ])
            input_hidden_size = hidden_size

        # final layer
        self.classifier.append(torch.nn.Linear(hidden_size, num_class))

        # convert list ot ModuleList
        self.classifier = torch.nn.ModuleList(self.classifier)

    def forward(self, inputs):
        outputs = inputs
        for layer in self.classifier:
            outputs = layer(outputs)
        return outputs


class LogTransformer(torch.nn.Module):
    # A Transformer-based encoder for abnormally detection on logs
    def __init__(self, num_class, vocab_size, hidden_size, num_layer, num_head,
                 dropout, decoder_hidden_size, max_len, name='LogTransformer'):
        super(LogTransformer, self).__init__()

        # initialize embedding
        self.num_class = num_class
        self.log_embed = torch.nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=hidden_size)

        # initialize encoder
        self.pos_encoder = PositionEncoding(hidden_size, max_len=max_len)
        self.transformer_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_size,
                                                                  nhead=num_head,
                                                                  dropout=dropout)
        self.encoder = torch.nn.TransformerEncoder(self.transformer_layer,
                                                   num_layers=num_layer)

        # initialize decoder
        self.decoder = LogClassifier(input_hidden_size=hidden_size,
                                    hidden_size=decoder_hidden_size,
                                    num_class=num_class,
                                    num_layer=2,
                                    dropout=dropout)

    def predict(self, inputs):
        # forward
        outputs = self.forward()

        # compute loss
        loss = self.compute_loss(outputs, labels)

        # softmax
        outputs = F.softmax(outputs, dim=-1)

        # compute metrics
        acc = Accuracy()(outputs, labels)
        f1 = F1(num_classes=self.num_class)(outputs, labels)
        return outputs, loss, acc, f1

    def compute_loss(self, outputs, labels):
        # Function to compute loss
        return F.cross_entropy(outputs, labels)

    def _forward(self, inputs):

        # embed logs-positions and logs
        pos_features = self.pos_encoder(inputs)
        log_features = self.log_embed(inputs)
        features = to_cuda(pos_features) + log_features

        # encode
        features = self.encoder(features)

        # get features of CLS (at the beginning)
        features = features[:, 0]

        # predict next log entries
        outputs = self.decoder(features)

        return outputs

    def forward(self, inputs, labels):
        # forward step
        outputs = self._forward(inputs)

        # compute loss
        loss = self.compute_loss(outputs, labels)

        # compute metrics
        outputs = F.softmax(outputs, dim=-1)
        acc = Accuracy()(outputs, labels)
        f1 = F1(num_classes=self.num_class)(outputs, labels)

        return loss, acc, f1


class LogForecast(torch.nn.Module):
    # A Transformer-based encoder for abnormally detection on logs
    def __init__(self, num_class, vocab_size, hidden_size, num_layer, num_head,
                 dropout, decoder_hidden_size, max_len, name='LogTransformer'):
        super(LogForecast, self).__init__()

        # initialize embedding
        self.log_embed = torch.nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=hidden_size)

        # initialize encoder
        self.pos_encoder = PositionEncoding(hidden_size, max_len=max_len)
        self.transformer_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_size,
                                                                  nhead=num_head,
                                                                  dropout=dropout)
        self.encoder = torch.nn.TransformerEncoder(self.transformer_layer,
                                                   num_layers=num_layer)

        # initialize decoder
        self.decoder = LogClassifier(input_hidden_size=hidden_size,
                                     hidden_size=decoder_hidden_size,
                                     num_class=num_class,
                                     num_layer=2,
                                     dropout=dropout)

    def predict(self, inputs):
        # forward
        outputs = self.forward()

        # softmax
        outputs = F.softmax(outputs, dim=-1)
        return outputs

    def compute_loss(self, outputs, labels):
        # Function to compute loss
        return F.cross_entropy(outputs, labels)

    def _forward(self, inputs):
        # embed logs-positions and logs
        pos_features = self.pos_encoder(inputs)
        log_features = self.log_embed(inputs)
        features = to_cuda(pos_features) + log_features

        # encode
        features = self.encoder(features)

        # get features of CLS (at the beginning)
        features = features[:, 0]

        # predict next log entries
        outputs = self.decoder(features)

        return outputs

    def forward(self, inputs, labels):
        # forward step
        outputs = self._forward(inputs)

        # compute loss
        loss = self.compute_loss(outputs, labels)

        return loss
