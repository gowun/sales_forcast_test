import logging
import os
import yaml
import torch
import torch.nn as nn
import pandas as pd
from argparse import ArgumentParser, Namespace
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoConfig,
    pipeline
)
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from pretraining_data import LMDataModule


class MLMModel(nn.Module):
    def __init__(self, model_name_or_path):
        super(MLMModel, self).__init__()

        config = AutoConfig.from_pretrained(model_name_or_path, return_dict=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, config=config)

    def forward(self, inputs):
        device = self.model.device
        return self.model(input_ids=inputs['input_ids'].to(device), 
                          attention_mask=inputs['attention_mask'].to(device), 
                          labels=inputs['labels'].to(device))
        
    def fill_mask(self, sentence):
        device = self.model.device
        fill_mask = pipeline(
            "fill-mask",
            model=self.model.cpu(),
            tokenizer=self.tokenizer
        )
        #self.model.to(device)
        return self.fill_mask(sentence)[0]['sequence'][6:-6]
    
    def compare_input_output(self, input_ids, labels):
        input_ids = input_ids.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        for j, (i, l) in enumerate(zip(input_ids, labels)):
            l[l == -100] = i[l == -100]
            labels[j, :] = l
        input_sentences = self.tokenizer.batch_decode(list(input_ids), skip_special_tokens=True)
        label_sentences = self.tokenizer.batch_decode(list(labels), skip_special_tokens=True)
        pred_sentences, tfs = [], []
        for i, l in zip(input_sentences, label_sentences):
            pred = self.fill_mask(i)
            pred_sentences.append(pred)
            tfs.append(pred == l)
        acc = sum(tfs) / len(tfs)
        return acc, pred_sentences
    
class Base(pl.LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super(Base, self).__init__()
        self.save_hyperparameters(hparams)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('-f', default=None)
        parser.add_argument('--lr', type=float, default=5e-5)
        parser.add_argument('--warmup_ratio', type=float, default=0.1)
        parser.add_argument('--seed', type=int, default=1234)
        parser.add_argument('--model_name_or_path', type=str, default='distilbert-base-uncased')
        parser.add_argument('--train_file', type=str, default="")
        parser.add_argument('--val_file', type=str, default="")
        parser.add_argument('--test_file', type=str, default="")
        parser.add_argument('--target', type=str, default='')
        parser.add_argument('--workers', type=int, default=4)
        parser.add_argument('--overwrite_cache', action='store_true', default=False)
        parser.add_argument('--max_length', type=int, default=512)
        parser.add_argument('--mlm_probability', type=float, default=0.15)
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--gpu_list', type=str, default='')
        
        return parser
    
    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_workers = self.hparams.num_nodes if self.hparams.num_nodes is not None else 1
        data_len = len(self.train_dataloader().dataset)
        logging.info(f'number of workers {num_workers}, data length {data_len}')
        num_train_steps = int(data_len / (self.hparams.batch_size * num_workers) * self.hparams.max_epochs)
        logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]
    
    
class Train_MLMModel(Base):
    def __init__(self, hparams, **kwargs):
        super(Train_MLMModel, self).__init__(hparams, **kwargs)
        
        self.save_hyperparameters(hparams)
        
        self.model = MLMModel(hparams.model_name_or_path)
        self.model.model.train()
        
        self.epoch = -1
        self.log_filepath = self.hparams.log_filepath
        self.log_file = None
        self.prior_test = False
        
        if self.hparams.gpus is None:
            self.agg_device = torch.device('cpu')
        elif self.hparams.gpus == -1:
            self.agg_device = torch.device('cuda:0')
        else:
            min_dv = min(self.hparams.gpus)
            self.agg_device = torch.device(f'cuda:{min_dv}')
    
    def forward(self, inputs):
        return self.model(inputs)
        
    def training_step(self, batch, batch_idx):
        loss = self(batch).loss.to(self.agg_device)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch).loss.to(self.agg_device)
        #accuracy, _ = self.model.compare_input_output(batch['input_ids'], batch['labels'])
        return loss
    
    def validation_epoch_end(self, outputs):
        losses = []
        for i in outputs:
            losses.append(i)
        loss = torch.stack(losses).mean()
        self.log('val_loss', loss, prog_bar=True)
        print(f'epoch={self.epoch}, val_loss={float(loss)}')
        self.log_file = open(self.log_filepath, 'a')
        if self.epoch == -1:
            if not self.prior_test:
                self.log_file.write(f'epoch,val_loss\n')
        else:
            self.log_file.write(f'{self.epoch},{float(loss)}\n')
        self.log_file.close()
        self.epoch += 1
        
    def test_step(self, batch, batch_idx):
        loss = self(batch).loss.to(self.agg_device)
        #accuracy, _ = self.model.compare_input_output(batch['input_ids'], batch['labels'])
        return loss
    
    def test_epoch_end(self, outputs):
        losses = []
        for i in outputs:
            losses.append(i)
        loss = torch.stack(losses).mean()
        self.log('test_loss', loss, prog_bar=True)
        self.log_file = open(self.log_filepath, 'a')
        if self.epoch == -1:
            self.log_file.write(f'epoch,val_loss\n')
            self.prior_test = True
        self.log_file.write(f'-1,{float(loss)}\n')
        self.log_file.close()


def pretrain(dict_args={}):
    # ------------
    # args
    # ------------
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Base.add_model_specific_args(parser)
    args = parser.parse_args()
    user = os.environ['USER']
    
    if len(dict_args) > 0:
        for k, v in dict_args.items():
            setattr(args, k, v)
    
        if args.gpu_list == '':
            if args.gpus > -1:
                setattr(args, 'gpus', str(args.gpus))
        else:
            gpu_list = list(map(lambda x: int(x), args.gpu_list.split(';')))
            setattr(args, 'gpus', gpu_list)
        
        setattr(args, 'root_dir', f'/home/{user}/logs_{args.model_name_or_path.split("/")[-1]}_{args.target}_{args.mlm_probability}_{args.seed}')
        setattr(args, 'log_filepath', os.path.join(args.root_dir, 'performance_log.txt'))
        logging.info(args)
        
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    data_module = LMDataModule(
        model_name_or_path=args.model_name_or_path,
        target=args.target,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        workers=args.workers,
        overwrite_cache=args.overwrite_cache,
        max_length=args.max_length,
        mlm_probability=args.mlm_probability,
        batch_size=args.batch_size
    )

    # ------------
    # model
    # ------------
    lmmodel = Train_MLMModel(args)

    # ------------
    # training
    # ------------
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                       dirpath=args.root_dir,
                                                       filename='model_chp/{epoch:02d}-{train_loss:.3f}-{val_loss:.3f}',
                                                       verbose=True,
                                                       save_last=False,
                                                       mode='min',
                                                       save_top_k=10)
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.root_dir, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10)
    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger, callbacks=[checkpoint_callback, lr_logger, early_stop_callback])
    trainer.test(lmmodel, test_dataloaders=data_module.test_dataloader())
    trainer.fit(lmmodel, data_module)
    trainer.test(ckpt_path="best", test_dataloaders=data_module.test_dataloader())
    
    
def evaluate_models(log_dir_path):
    last_version = os.popen(f'ls {log_dir_path}/tb_logs/default/').readlines()[-1][:-1]
    
    with open(f'{log_dir_path}/tb_logs/default/{last_version}/hparams.yaml', 'r') as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
    args = Namespace(**hparams)
    model_names = os.popen(f'ls {log_dir_path}/model_chp/').readlines()
    
    perf_df = pd.read_csv(f'{log_dir_path}/performance_log.txt', sep=',', header=0)
    perf_pairs = list(perf_df.loc[perf_df['epoch'] > -1].values)
    best_epoch = sorted(perf_pairs, key=lambda x: x[1])[0][0]
    last_epoch = perf_pairs[-1][0]
    model_files = dict()
    for i, epoch in enumerate([best_epoch, last_epoch]):
        if i == 0:
            tag = 'best'
        else:
            tag = 'last'
        matched = list(filter(lambda x: x.startswith('epoch={0:02d}'.format(int(epoch))), model_names))
        if len(matched) == 1:
            model_files[tag] = f'{log_dir_path}/model_chp/{matched[0][:-1]}'
    print(model_files)
    
    data_module = LMDataModule(
        model_name_or_path=args.model_name_or_path,
        target=args.target,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        workers=args.workers,
        overwrite_cache=args.overwrite_cache,
        max_length=args.max_length,
        mlm_probability=args.mlm_probability,
        batch_size=args.batch_size
    )
    
    trainer = pl.Trainer.from_argparse_args(args)
    model_result = dict()
    for k, v in model_files.items():
        lmmodel = Train_MLMModel.load_from_checkpoint(checkpoint_path=v, hparams=args)
        model_result[k] = trainer.test(lmmodel, test_dataloaders=data_module.test_dataloader())
    return model_result

def load_model(log_dir_path, mode='best'):
    last_version = os.popen(f'ls {log_dir_path}/tb_logs/default/').readlines()[-1][:-1]
    
    with open(f'{log_dir_path}/tb_logs/default/{last_version}/hparams.yaml', 'r') as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
    args = Namespace(**hparams)
    model_names = os.popen(f'ls {log_dir_path}/model_chp/').readlines()
    
    perf_df = pd.read_csv(f'{log_dir_path}/performance_log.txt', sep=',', header=0)
    perf_pairs = list(perf_df.loc[perf_df['epoch'] > -1].values)
    if mode == 'best':
        epoch = sorted(perf_pairs, key=lambda x: x[1])[0][0]
    elif mode == 'last':
        epoch = perf_pairs[-1][0]
    matched = list(filter(lambda x: x.startswith('epoch={0:02d}'.format(int(epoch))), model_names))
    if len(matched) == 1:
        file_name = f'{log_dir_path}/model_chp/{matched[0][:-1]}'
        print(file_name)
        return Train_MLMModel.load_from_checkpoint(checkpoint_path=file_name, hparams=args)
    else:
        return None
