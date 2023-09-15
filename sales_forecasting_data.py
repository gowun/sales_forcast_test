import warnings
import pickle
from datasets import load_dataset
import pytorch_lightning as pl
import numpy as np
from torch.utils.data.dataloader import DataLoader
from transformers import (
    ElectraModel,
    ElectraTokenizer
)

TOK_PATH = 'koelectra_tokenizer'

class SalesDataModule(pl.LightningDataModule):
    def __init__(self, train_file, val_file, test_file,
                 workers, overwrite_cache, batch_size, 
                 max_length=32, use_bert=False, add_total=False, transformer_path=''):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.workers = workers
        self.overwrite_cache = overwrite_cache
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_bert = use_bert
        self.add_total = add_total

        if self.use_bert:
            self.tokenizer = ElectraTokenizer.from_pretrained(TOK_PATH)
        if len(transformer_path) > 0:
            self.transformer = pickle.load(open(transformer_path, 'rb'))
        else:
            self.transformer = None
        
        remove_cols = ['input', 'store_seq', 'total', 'date']
        datasets = load_dataset('json', data_files={
            'train': self.train_file,
            'val': self.val_file,
        })
        datasets = datasets.map(
            self.process_file,
            batched=True,
            num_proc=self.workers,
            remove_columns=remove_cols,
            load_from_cache_file=not self.overwrite_cache,
        )
        testdatasets = load_dataset('json', data_files={'test': self.test_file})
        testdatasets = testdatasets.map(
            self.process_file,
            batched=True,
            num_proc=1,
            remove_columns=remove_cols,
            load_from_cache_file=not self.overwrite_cache,
        )
        
        self.train_dataset = datasets["train"]
        self.val_dataset = datasets["val"]
        self.test_dataset = testdatasets["test"]
        
    def scale(self, arr_list):
        arr_list = np.array(arr_list)
        sizes = list(arr_list.shape)

        if len(sizes) == 2:
            sizes += [1]
            arr_list = np.vstack(arr_list.reshape(*sizes))
        else:
            arr_list = np.concatenate(np.vstack(arr_list)).reshape(-1, 1)
        return np.array(self.transformer.transform(arr_list).reshape(*sizes), dtype=np.float16)
        
    def bert_model(self, seq_data):
        bert_input = []
        for seq_list in seq_data:
            token_list = []
            for seqs in seq_list:
                token_list.append(self.tokenizer.batch_encode_plus(
                    seqs,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).input_ids.numpy())
            bert_input.append(np.stack(token_list))
        return np.stack(bert_input)
    
    def process_file(self, batch):
        if self.transformer is None:
            results = {
                'input': batch['input'],
                'tr_output': batch['output'],
            }
        else:
            results = {
                'input': self.scale(batch['input']),
                'tr_output': self.scale(batch['output']),
            }
            
        if self.add_total:
            results['total'] = self.scale(batch['total'])
        
        if self.use_bert:
            results['bert_input'] = self.bert_model(batch['store_seq'])
        return results

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