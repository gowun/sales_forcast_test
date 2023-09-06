import warnings
from datasets import load_dataset
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

class LMDataModule(pl.LightningDataModule):
    def __init__(self, model_name_or_path, target, train_file, val_file, test_file,
                 workers, overwrite_cache, max_length, mlm_probability,
                 batch_size):
        super().__init__()
        self.target = target
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.model_name_or_path = model_name_or_path
        self.workers = workers
        self.overwrite_cache = overwrite_cache
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.batch_size = batch_size

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        datasets = load_dataset('csv', data_files={
            'train': self.train_file,
            'val': self.val_file,
        })
        testdatasets = load_dataset('csv', data_files={'test': self.test_file})
        remove_cols = datasets['train'].column_names
        
        def tokenize_function(data):
            return tokenizer.batch_encode_plus(
                data[self.target],
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=self.workers,
            remove_columns=remove_cols,
            load_from_cache_file=not self.overwrite_cache,
        )
        tokenized_testdatasets = testdatasets.map(
            tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=remove_cols,
            load_from_cache_file=not self.overwrite_cache,
        )
        
        self.train_dataset = tokenized_datasets["train"]
        self.val_dataset = tokenized_datasets["val"]
        self.test_dataset = tokenized_testdatasets["test"]
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=self.mlm_probability)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.workers,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=1,
        )