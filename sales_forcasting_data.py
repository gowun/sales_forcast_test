import warnings
import torch
from datasets import load_dataset
import pytorch_lightning as pl
import numpy as np
from torch.utils.data.dataloader import DataLoader
from transformers import (
    ElectraModel,
    ElectraTokenizer
)

MODEL_PATH = 'sales_seq_electra-small_model'
TOK_PATH = 'koelectra_tokenizer'

class SalesDataModule(pl.LightningDataModule):
    def __init__(self, item, train_file, val_file, test_file,
                 workers, overwrite_cache, batch_size, max_length=32, use_bert=False):
        super().__init__()
        self.item = str(item)
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.workers = workers
        self.overwrite_cache = overwrite_cache
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_bert = use_bert

        if self.use_bert:
            self.tokenizer = ElectraTokenizer.from_pretrained(TOK_PATH)
            self.bert = ElectraModel.from_pretrained(MODEL_PATH)
            
        datasets = load_dataset('json', data_files={
            'train': self.train_file,
            'val': self.val_file,
        })
        testdatasets = load_dataset('json', data_files={'test': self.test_file})

        tokenized_datasets = datasets.map(
            self.process_file,
            batched=True,
            num_proc=self.workers,
            load_from_cache_file=not self.overwrite_cache,
        )
        tokenized_testdatasets = testdatasets.map(
            self.process_file,
            batched=True,
            num_proc=1,
            load_from_cache_file=not self.overwrite_cache,
        )
        
        self.train_dataset = tokenized_datasets["train"]
        self.val_dataset = tokenized_datasets["val"]
        self.test_dataset = tokenized_testdatasets["test"]
        
    def process_file(self, data):
        item_data = data[self.item]
        out_dict = {
            'input': np.array(list(map(lambda x: item_data[x]['input'], item_date.keys())), dtype=np.float16),
            'output': np.array(list(map(lambda x: item_data[x]['output'], item_date.keys())), dtype=np.float16),
            'total': np.array(list(map(lambda x: item_data[x]['total'], item_date.keys())), dtype=np.float16)
        }
        if self.use_bert:
            input_ids = self.tokenizer.batch_encode_plus(
                list(map(lambda x: item_data[x]['store_seq'], item_date.keys())),
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).input_ids
            with torch.no_grad():
                out_dict['bert_out'] = self.bert(input_ids).cpu().detach().numpy().astype(np.float16)
        return out_dict 

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=1,
        )