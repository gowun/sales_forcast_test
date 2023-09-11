import logging
import os
import yaml
import torch
import torch.nn as nn
import pandas as pd
from argparse import ArgumentParser, Namespace
from transformers import (
    ElectraModel,
    ElectraTokenizer,
    AutoConfig,
    pipeline
)
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

MODEL_PATH = 'sales_seq_electra-small_model'


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_seq, drop_prob=0.2, use_bert=False):
        super(GRUNet, self).__init__()
        
        if use_bert:
            config = AutoConfig.from_pretrained(MODEL_PATH, return_dict=True)
            hidden_size = 2 * hidden_dim
            self.bert_layer = nn.Linear(config.hidden_size, hidden_dim)
        else:
            hidden_size = hidden_dim
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_seq, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_size, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, h, bert_out=None):
        out, h = self.gru(x, h)
        if bert_out is None:
            out = out[:, -1]
        else:
            bert_out = self.bert_layer(bert_out)
            out = torch.cat([x[:, -1], bert_out[:, -1]], -1)
        out = self.fc(self.relu(out))
        return out, h
    
    def init_hidden(self, batch_size):
        device = self.gru.device
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_seq, drop_prob=0.2, use_bert=False):
        super(LSTMNet, self).__init__()
        
        if use_bert:
            config = AutoConfig.from_pretrained(MODEL_PATH, return_dict=True)
            hidden_size = 2 * hidden_dim
            self.bert_layer = nn.Linear(config.hidden_size, hidden_dim)
        else:
            hidden_size = hidden_dim
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_seq, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_size, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, h, bert_out=None):
        out, h = self.lstm(x, h)
        if bert_out is None:
            out = out[:, -1]
        else:
            bert_out = self.bert_layer(bert_out)
            out = torch.cat([x[:, -1], bert_out[:, -1]], -1)
        out = self.fc(self.relu(out))
        return out, h
    
    def init_hidden(self, batch_size):
        device = self.lstm.device
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden